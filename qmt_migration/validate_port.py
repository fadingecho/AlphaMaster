"""Read-only checks against AlphaMaster; writes only this migration's artifacts."""
import ast
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from model_core.features import MT5FeatureEngineer as FE
from model_core.vocab import FORMULA_VOCAB
from model_core.vm import StackVM
from model_core.ops import _winsorize, _decay_linear, _ts_arg_min

torch.set_num_threads(2)
spec = importlib.util.spec_from_file_location('port', HERE / 'ALPHAMASTER_D1_BACKTEST.py')
port = importlib.util.module_from_spec(spec)
spec.loader.exec_module(port)
frame = pd.read_csv(ROOT / 'demo/data/market_snapshot.csv', dtype={'date': str})
raw = {k: torch.tensor(frame[k if k != 'volume' else 'tick_volume'].to_numpy()[None],
                       dtype=torch.float32) for k in ('open', 'high', 'low', 'close', 'volume')}
raw['time'] = torch.tensor(frame.time.to_numpy()[None], dtype=torch.int64)
features = torch.zeros(1, FORMULA_VOCAB.feature_count, len(frame))
refs = {}
for name, func, key in [('REL_RET20', FE._c_rel_ret20, 'rel_ret20'),
                       ('VWAP_DEV', FE._c_vwap_dev, 'vwap_dev'),
                       ('DONCHIAN_POS_20', FE._c_donchian_pos_20, 'donchian')]:
    value = func(raw)
    features[:, FORMULA_VOCAB.feature_names.index(name), :] = value
    refs[key] = value[0].numpy()
refs['condition'] = _decay_linear(_winsorize(torch.from_numpy(refs['rel_ret20'][None])), 5).abs()[0].numpy()
refs['argmin'] = _ts_arg_min(torch.from_numpy(refs['vwap_dev'][None]), 5)[0].numpy()
refs['selected'] = np.where(refs['condition'] > 0, refs['argmin'], refs['donchian'])
refs['factor'] = StackVM().execute(port.FORMULA_TOKENS, features)[0].numpy()
refs['position'] = np.maximum(np.tanh(refs['factor']), 0)
refs['position'][refs['position'] < port.MIN_EXPOSURE] = 0
args = [frame[k].values for k in ('high', 'low', 'close', 'tick_volume')]
actual = port.calculate_signal(*args, diagnostics=True)
errors = {k: float(np.max(np.abs(actual[k] - refs[k]))) for k in refs}
print('MAX_ABS_ERRORS', json.dumps(errors), flush=True)
for key in ('argmin', 'selected'):
    np.testing.assert_allclose(actual[key], refs[key], atol=1e-6, rtol=1e-6)
for key in ('factor', 'position'):
    np.testing.assert_allclose(actual[key], refs[key], atol=2e-5, rtol=2e-5)
np.testing.assert_array_equal(actual['position'] > 0, refs['position'] > 0)

checks = ['Python 3.6 syntax', 'source formula identity', 'full-series signal parity',
          'long/flat decision parity', 'prefix causality', 'backtest-mode guard',
          'lot/cash/T+1/limit constraints', 'QMT callback harness']
ast.parse((HERE / 'ALPHAMASTER_D1_BACKTEST.py').read_text(), feature_version=(3, 6))
source_strategy = json.loads((ROOT / 'strategies/best_DEMO_000001.SZ.json').read_text())
assert source_strategy['formula'] == port.FORMULA_TOKENS
assert source_strategy['train_steps'] == 80
for stop in (500, 600, 682, 720, len(frame)):
    fac, pos = port.calculate_signal(*(x[:stop] for x in args))
    np.testing.assert_allclose(fac[-1], actual['factor'][stop-1], atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(pos[-1], actual['position'][stop-1], atol=2e-5, rtol=2e-5)

for mode in (False, 0, 'False', 'True', None):
    try:
        port._require_backtest(type('Context', (), {'do_back_test': mode})())
    except RuntimeError:
        pass
    else:
        raise AssertionError('Unsafe mode accepted: %r' % mode)
assert port.plan_order(0, 0, 1000, 0, 10, 10)[0] == 0
assert port.plan_order(0, 0, 1000, 500, 10, 10)[0] == -500
assert port.plan_order(1, 10000, 0, 0, 11, 10)[0] == 0
assert port.plan_order(0, 0, 1000, 1000, 9, 10)[0] == 0
assert port.plan_order(1, 1000, 0, 0, 10, 10)[0] == 0
assert port.plan_order(1, 2000, 0, 0, 10, 10)[0] == 100


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class Context:
    do_back_test = True
    stockcode, market, period = '000001', 'SZ', '1d'

    def __init__(self):
        self.barpos = 682
        self.cash = 100000.0
        self.shares = 0
        self.orders = []

    def set_universe(self, universe):
        assert universe == [port.SYMBOL]

    def set_commission(self, kind, values):
        assert kind == 0 and values == [0, .0005, .0003, .0003, 0, 5]

    def set_slippage(self, kind, value):
        assert kind == 2 and value == .0002

    def get_bar_timetag(self, i):
        return frame.date.iloc[i]

    def get_market_data_ex(self, fields, symbols, **kw):
        assert symbols == [port.SYMBOL]
        assert kw['subscribe'] is False and kw['fill_data'] is False
        # Deliberately expose future signal data; the adapter must filter it.
        if kw['dividend_type'] == 'front':
            out = frame.rename(columns={'tick_volume': 'volume'}).set_index('date')
        else:
            i = frame.index[frame.date == kw['end_time']][0]
            row = frame.iloc[i]
            out = pd.DataFrame([dict(open=row.raw_open, close=row.raw_close,
                                     preClose=row.pre_close, volume=row.tick_volume,
                                     suspendFlag=0)], index=[row.date])
        return {port.SYMBOL: out[fields]}


ctx = Context()
port.timetag_to_datetime = lambda value, fmt: value + '150000'


def trade_details(account_id, kind, detail):
    assert account_id == 'alphamaster_backtest_only' and kind == 'STOCK'
    if detail == 'ACCOUNT':
        return [Obj(m_dAvailable=ctx.cash)]
    return [Obj(m_strInstrumentID='000001', m_strExchangeID='SZ',
                m_nVolume=ctx.shares, m_nCanUseVolume=ctx.shares)] if ctx.shares else []


def order_shares(symbol, qty, style, price, C, account):
    assert C.do_back_test is True and symbol == port.SYMBOL and style == 'FIX'
    assert qty % 100 == 0 and ctx.shares + qty >= 0
    amount = abs(qty) * price
    cost = max(5, amount * .0003) + (amount * .0005 if qty < 0 else 0) + amount * .0002
    ctx.cash -= qty * price + cost
    ctx.shares += qty
    assert ctx.cash >= -1e-6
    ctx.orders.append((frame.date.iloc[ctx.barpos], qty))


port.get_trade_detail_data = trade_details
port.order_shares = order_shares
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    port.init(ctx)
    for i in range(682, len(frame)):
        ctx.barpos = i
        decision, fac, pos, bars = port._previous_signal(ctx, frame.date.iloc[i])
        assert decision == frame.date.iloc[i-1]
        assert bars == i
        np.testing.assert_allclose(pos, refs['position'][i-1], atol=2e-5, rtol=2e-5)
        port.handlebar(ctx)
        n = len(ctx.orders)
        port.handlebar(ctx)  # duplicate callbacks must not create extra orders
        assert len(ctx.orders) == n
assert len(port.g.done) == 120 and not port.g.halted
assert ctx.orders
(HERE / 'callback_harness.log').write_text(captured.getvalue())
with (HERE / 'reference_signals.csv').open('w', newline='') as out:
    writer = csv.writer(out)
    writer.writerow(['date', 'torch_factor', 'numpy_factor', 'torch_target', 'numpy_target'])
    writer.writerows(zip(frame.date, refs['factor'], actual['factor'], refs['position'], actual['position']))
summary = dict(checks=checks, bars=len(frame), holdout_bars=120,
               max_abs_errors=errors, callback_sessions=120,
               callback_order_requests=len(ctx.orders),
               mode='offline mock of QMT APIs; NOT a QMT-native backtest',
               source_strategy_sha256=hashlib.sha256((ROOT / 'strategies/best_DEMO_000001.SZ.json').read_bytes()).hexdigest(),
               port_sha256=hashlib.sha256((HERE / 'ALPHAMASTER_D1_BACKTEST.py').read_bytes()).hexdigest())
(HERE / 'validation.json').write_text(json.dumps(summary, indent=2))
print('VALIDATION_PASSED', json.dumps(summary), flush=True)
