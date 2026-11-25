# coding: utf-8
"""Verify portability with QMT's own Python; no QMT account/API calls."""
import csv
import importlib.util
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    import numpy as np
    import pandas as pd
    spec = importlib.util.spec_from_file_location('port', os.path.join(HERE, 'ALPHAMASTER_D1_BACKTEST.py'))
    port = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(port)
    # Old Windows pandas cannot reliably open a non-ASCII filename itself.
    with open(os.path.join(HERE, '..', 'demo', 'data', 'market_snapshot.csv'), encoding='utf-8') as stream:
        frame = pd.read_csv(stream, dtype={'date': str})
    with open(os.path.join(HERE, 'reference_signals.csv'), encoding='utf-8') as stream:
        expected = pd.read_csv(stream)
    fac, pos = port.calculate_signal(*[frame[k].values for k in ('high', 'low', 'close', 'tick_volume')])
    np.testing.assert_allclose(fac, expected.torch_factor.values, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(pos, expected.torch_target.values, atol=2e-5, rtol=2e-5)
    np.testing.assert_array_equal(pos > 0, expected.torch_target.values > 0)
    class Context(object):
        def get_market_data_ex(self, fields, symbols, **kw):
            data = frame.rename(columns={'tick_volume': 'volume'}).set_index('date')
            return {port.SYMBOL: data[fields]}
    decision, factor, target, bars = port._previous_signal(Context(), '20260127')
    assert decision == '20260126' and bars == 682
    np.testing.assert_allclose(target, expected.torch_target.values[681], atol=2e-5, rtol=2e-5)
    return dict(status='passed', python=sys.version, executable=sys.executable,
                numpy=np.__version__, pandas=pd.__version__, bars=len(frame),
                max_position_error=float(np.max(np.abs(pos - expected.torch_target.values))),
                scope='QMT bundled Python only; NOT the QMT native backtest engine')


try:
    result = main()
except Exception:
    result = dict(status='failed', error=traceback.format_exc())
with open(os.path.join(HERE, 'windows_runtime_validation.json'), 'w', encoding='utf-8') as out:
    json.dump(result, out, ensure_ascii=False, indent=2)
