#coding:gbk
"""AlphaMaster frozen formula, ported to QMT built-in Python 3.6 + NumPy.

Historical BACKTEST only. No trained neural network or external service needed.
Main chart: 000001.SZ, daily, unadjusted. Click BACKTEST, not RUN.
"""
import math
from decimal import Decimal, ROUND_HALF_UP

import numpy as np


SYMBOL = '000001.SZ'
HISTORY_START = '20230406'
START_DATE = '20260127'
END_DATE = '20260728'
INITIAL_CASH = 100000.0
ACCOUNT_ID = 'alphamaster_backtest_only'
FORMULA_TOKENS = [18, 122, 102, 70, 20, 101, 51, 126]
FORMULA_TEXT = ('REL_RET20 WINSORIZE DECAY_LINEAR_5 ABS '
                'VWAP_DEV TS_ARG_MIN_5 DONCHIAN_POS_20 IF_GT')
MIN_EXPOSURE = 0.05
COMMISSION = 0.0003
MIN_COMMISSION = 5.0
SLIPPAGE = 0.0002
STAMP_DUTY = 0.0005  # This fixed example is entirely after 2023-08-28.


class State(object):
    pass


g = State()


def _clean(x):
    return np.where(np.isfinite(x), x, np.float32(0)).astype(np.float32)


def _windows(x, width):
    x = np.asarray(x, dtype=np.float32)
    padded = np.concatenate((np.zeros(width - 1, dtype=np.float32), x))
    return np.lib.stride_tricks.as_strided(
        padded, shape=(len(x), width),
        strides=(padded.strides[0], padded.strides[0]), writeable=False)


def _robust_norm(x, width=200):
    # torch.median selects the LOWER middle element for an even-sized window.
    x = np.asarray(x, dtype=np.float32)
    if len(x) < width:
        return np.zeros_like(x)
    windows = _windows(x, width)
    middle = (width - 1) // 2
    med = np.partition(windows, middle, axis=1)[:, middle]
    deviations = np.abs(windows - med[:, None])
    mad = np.partition(deviations, middle, axis=1)[:, middle] + np.float32(1e-6)
    out = _clean(np.clip((x - med) / mad, -5.0, 5.0))
    out[:width - 1] = 0.0
    return out


def _winsorize(x):
    windows = _windows(x, 20)
    lower = np.percentile(windows, 5, axis=1).astype(np.float32)
    upper = np.percentile(windows, 95, axis=1).astype(np.float32)
    flat = (upper - lower) < np.float32(1e-9)
    return _clean(np.clip(x, np.where(flat, x, lower), np.where(flat, x, upper)))


def _normalize_output(x):
    # Preserve StackVM's sample standard deviation and 500-bar warm-up.
    if len(x) < 2 or np.std(x, ddof=1) < 1e-6:
        return x.copy()
    if len(x) < 500:
        count = np.arange(1, len(x) + 1, dtype=np.float32)
        mean = np.cumsum(x) / count
        var = np.cumsum(x * x) / count - mean * mean
        return np.clip((x - mean) / np.sqrt(np.maximum(var, 1e-8)), -3, 3)
    windows = _windows(x, 500)
    mean = windows.mean(axis=1)
    std = np.maximum(windows.std(axis=1, ddof=1), np.float32(1e-8))
    out = np.clip((x - mean) / std, -3, 3)
    out[:499] = 0.0
    return out.astype(np.float32)


def calculate_signal(high, low, close, volume, diagnostics=False):
    """Compute the exact selected SINGLE-STOCK formula on chronological bars.

    Input prices are front-adjusted. A multi-stock universe changes REL_RET20's
    original semantics; this port deliberately remains a one-stock experiment.
    """
    high, low, close, volume = [np.asarray(x, dtype=np.float32)
                               for x in (high, low, close, volume)]
    if len(close) < 21 or not all(len(x) == len(close) for x in (high, low, volume)):
        raise ValueError('Need aligned OHLCV arrays with at least 21 bars')
    if not all(np.isfinite(x).all() for x in (high, low, close, volume)):
        raise ValueError('Non-finite market data')
    if np.any(low <= 0) or np.any(close <= 0) or np.any(high < low) or np.any(volume < 0):
        raise ValueError('Invalid OHLCV values')
    eps = np.float32(1e-9)
    ret20 = np.zeros_like(close)
    ret20[20:] = np.log(close[20:] / (close[:-20] + eps))
    rel_ret20 = _robust_norm(ret20)
    winsor = _winsorize(rel_ret20)
    weights = np.arange(1, 6, dtype=np.float32) / np.float32(15)
    condition = np.abs((_windows(winsor, 5) * weights).sum(axis=1))

    typical = (high + low + close) / np.float32(3)
    vwap = (_windows(typical * volume, 20).sum(axis=1)
            / (_windows(volume, 20).sum(axis=1) + eps))
    vwap_dev = _robust_norm((close - vwap) / (vwap + eps))
    argmin = _windows(vwap_dev, 5).argmin(axis=1).astype(np.float32) / np.float32(4)
    highest = _windows(high, 20).max(axis=1)
    lowest = _windows(low, 20).min(axis=1)
    donchian = np.clip((close - lowest) / (highest - lowest + eps), 0, 1)
    selected = np.where(condition > 0, argmin, donchian).astype(np.float32)
    factor = _normalize_output(selected)
    position = np.maximum(np.tanh(factor), np.float32(0))
    position[position < MIN_EXPOSURE] = 0.0
    if diagnostics:
        return dict(rel_ret20=rel_ret20, vwap_dev=vwap_dev, condition=condition,
                    argmin=argmin, donchian=donchian, selected=selected,
                    factor=factor, position=position)
    return factor, position


def _require_backtest(C):
    # Never set do_back_test ourselves; require the platform's actual mode.
    mode = getattr(C, 'do_back_test', False)
    if not isinstance(mode, (bool, np.bool_, int)) or mode != True:
        raise RuntimeError('BACKTEST ONLY: click the Backtest button in QMT')


def _date_key(value):
    digits = ''.join(ch for ch in str(value) if ch.isdigit())
    if len(digits) >= 8 and digits[:2] in ('19', '20'):
        return digits[:8]
    raise ValueError('Unexpected QMT daily index: %r' % value)


def _read_frame(C, fields, date, adjustment, start='', count=-1):
    values = C.get_market_data_ex(
        fields, [SYMBOL], period='1d', start_time=start, end_time=date,
        count=count, dividend_type=adjustment, fill_data=False, subscribe=False)
    frame = (values or {}).get(SYMBOL)
    if frame is None or len(frame.index) == 0:
        raise RuntimeError('Download local daily history for ' + SYMBOL)
    frame = frame.sort_index()
    if frame.index.duplicated().any():
        raise RuntimeError('Duplicate daily market data')
    return frame


def _previous_signal(C, date):
    frame = _read_frame(C, ['high', 'low', 'close', 'volume'], date,
                        'front', start=HISTORY_START)
    keys = np.array([_date_key(x) for x in frame.index])
    # Even if QMT returns the whole backtest range, future/current data is removed.
    mask = (keys >= HISTORY_START) & (keys < date)
    frame = frame.loc[mask]
    keys = keys[mask]
    if len(frame) < 500:
        raise RuntimeError('Need at least 500 prior daily bars, from ' + HISTORY_START)
    if keys[0] != HISTORY_START:
        raise RuntimeError('Missing history start ' + HISTORY_START + '; download daily data')
    factor, position = calculate_signal(
        frame['high'].values, frame['low'].values,
        frame['close'].values, frame['volume'].values)
    return keys[-1], float(factor[-1]), float(position[-1]), len(frame)


def _execution_bar(C, date):
    frame = _read_frame(C, ['open', 'close', 'preClose', 'volume', 'suspendFlag'],
                        date, 'none', count=1)
    if _date_key(frame.index[-1]) != date:
        return None  # Missing/suspended bar: do not trade using a stale quote.
    row = frame.iloc[-1]
    op = float(row['open'])
    prev = float(row['preClose'])
    if not (math.isfinite(op) and math.isfinite(prev) and op > 0 and prev > 0):
        raise RuntimeError('Invalid raw opening/reference price')
    if float(row['volume']) <= 0 or int(row.get('suspendFlag', 0)) != 0:
        return None
    # FIX is interpreted in the chart's price basis. Reject incompatible charts.
    chart = _read_frame(C, ['open'], date, 'follow', count=1)
    if _date_key(chart.index[-1]) != date or abs(float(chart.iloc[-1]['open']) - op) > 0.0005:
        raise RuntimeError('Select an UNADJUSTED main chart for raw opening-price fills')
    return op, prev


def _account_position():
    accounts = get_trade_detail_data(ACCOUNT_ID, 'STOCK', 'ACCOUNT') or []
    if not accounts:
        raise RuntimeError('QMT did not return its virtual backtest account')
    cash = float(accounts[0].m_dAvailable)
    positions = get_trade_detail_data(ACCOUNT_ID, 'STOCK', 'POSITION') or []
    shares, sellable = 0, 0
    for p in positions:
        symbol = p.m_strInstrumentID + '.' + p.m_strExchangeID
        if symbol == SYMBOL:
            shares += int(p.m_nVolume)
            sellable += int(p.m_nCanUseVolume)
        elif int(p.m_nVolume) != 0:
            raise RuntimeError('Unexpected holding in this single-stock backtest')
    if not math.isfinite(cash) or cash < -0.01 or shares < 0 or sellable < 0:
        raise RuntimeError('Invalid account/position returned by QMT')
    return max(0.0, cash), shares, min(shares, sellable)


def plan_order(target, cash, shares, sellable, op, prev):
    """Determine a whole-lot order without shorts, borrowing or unavailable sales."""
    if not math.isfinite(target) or not 0 <= target <= 1:
        raise ValueError('Target must be in [0, 1]')
    desired = int((cash + shares * op) * target / op / 100) * 100
    delta = desired - shares
    up = float(Decimal(str(prev * 1.10)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
    down = float(Decimal(str(prev * 0.90)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
    if (delta > 0 and op >= up - .005) or (delta < 0 and op <= down + .005):
        return 0, 'opening_limit_estimate'
    if delta > 0:
        qty = delta
        while qty > 0:
            notional = qty * op
            reserve = notional * (1 + SLIPPAGE) + max(MIN_COMMISSION, notional * COMMISSION)
            # Reserve a little for separately charged transfer costs as well.
            reserve += notional * 0.00001
            if reserve <= cash:
                break
            qty -= 100
        return qty, 'buy' if qty else 'insufficient_cash'
    if delta < 0:
        qty = min(-delta, shares, sellable)
        qty = int(qty // 100) * 100
        return -qty, 'sell' if qty else 'no_sellable_position'
    return 0, 'hold'


def init(C):
    _require_backtest(C)
    C.start = START_DATE[:4] + '-' + START_DATE[4:6] + '-' + START_DATE[6:] + ' 00:00:00'
    C.end = END_DATE[:4] + '-' + END_DATE[4:6] + '-' + END_DATE[6:] + ' 23:59:59'
    C.capital = INITIAL_CASH
    C.set_universe([SYMBOL])
    C.set_commission(0, [0, STAMP_DUTY, COMMISSION, COMMISSION, 0, MIN_COMMISSION])
    C.set_slippage(2, SLIPPAGE)
    g.done = set()
    g.requests = 0
    g.halted = False
    print('AM_INIT|%s|%s|%s|BACKTEST_ONLY' % (SYMBOL, START_DATE, END_DATE))
    print('AM_FORMULA|' + FORMULA_TEXT)
    print('AM_COSTS|commission=0.0003|min=5|sell_tax=0.0005|slippage=0.0002|transfer=not_separate')


def handlebar(C):
    _require_backtest(C)
    if g.halted:
        return
    date = timetag_to_datetime(C.get_bar_timetag(C.barpos), '%Y%m%d%H%M%S')[:8]
    if date < START_DATE or date > END_DATE or date in g.done:
        return
    try:
        chart_symbol = str(C.stockcode) + '.' + str(C.market)
        if chart_symbol != SYMBOL or str(C.period).lower() not in ('1d', 'd1', 'day'):
            raise RuntimeError('Main chart must be 000001.SZ, daily')
        execution = _execution_bar(C, date)
        if execution is None:
            g.done.add(date)
            print('AM_SKIP|%s|missing_or_suspended' % date)
            return
        decision_date, factor, target, bars = _previous_signal(C, date)
        op, prev = execution
        cash, shares, sellable = _account_position()
        delta, reason = plan_order(target, cash, shares, sellable, op, prev)
        print('AM_SIGNAL|%s|%s|%.9f|%.9f|%s|%s' %
              (date, decision_date, factor, target, bars, reason))
        # Mark before submitting: an exception must not cause a duplicate retry.
        g.done.add(date)
        if delta:
            _require_backtest(C)
            order_shares(SYMBOL, int(delta), 'FIX', op, C, ACCOUNT_ID)
            g.requests += 1
            print('AM_ORDER_REQUEST|%s|%d|%.6f' % (date, delta, op))
        print('AM_PRETRADE|%s|%.4f|%d|%d' % (date, cash, shares, sellable))
        if date == END_DATE:
            print('AM_END|%s|sessions=%d|order_requests=%d' %
                  (date, len(g.done), g.requests))
    except Exception as exc:
        g.halted = True
        print('AM_ERROR|%s|%s' % (date, str(exc)))
        raise
