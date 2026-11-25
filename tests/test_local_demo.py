"""Checks for the independent daily ledger used by the local demo report."""
import numpy as np
import pandas as pd
import pytest

from demo_experiment import fee, simulate


def market(opens=(10., 10., 10.), closes=(10., 10., 10.)):
    return pd.DataFrame({
        "date": ["20260105", "20260106", "20260107"],
        "raw_open": opens, "raw_close": closes, "pre_close": [10., 10., 10.],
    })


def test_signal_is_executed_next_session_and_t_plus_one():
    result = simulate(market(), np.array([1., 0., 0.]), 0, 3, {})
    assert [(x["date"], x["side"]) for x in result["fills"]] == [
        ("20260106", "buy"), ("20260107", "sell")]
    assert result["ending_shares"] == 0
    assert result["return_pct"] < 0  # Flat market, actual commissions and slippage.
    assert all(x["quantity"] % 100 == 0 for x in result["fills"])


def test_future_signals_do_not_change_past_orders():
    a = simulate(market(), np.array([1., 0., 0.]), 0, 2, {})
    b = simulate(market(), np.array([1., 1., 1.]), 0, 2, {})
    assert a == b


def test_opening_limit_and_no_shorting():
    result = simulate(market(opens=(10., 11., 10.)), np.array([1., -1., 0.]), 0, 3, {})
    assert result["blocked_orders"] == 1
    assert result["orders"] == 0
    assert result["ending_cash"] == 100000.


def test_dividend_paid_only_to_preexisting_shares():
    frame = market()
    target = np.ones(3)
    base = simulate(frame, target, 0, 3, {})
    same_day_buy = simulate(frame, target, 0, 3, {"20260106": .5})
    assert base == same_day_buy
    credited = simulate(frame, target, 0, 3, {"20260107": .5})
    assert credited["equity"][-1] > base["equity"][-1]


def test_historical_stamp_duty_and_minimum_commission():
    assert fee(1000., False, "20260105") == pytest.approx(5.01)
    assert fee(100000., True, "20230827") - fee(100000., True, "20230828") == pytest.approx(50.)


def test_buy_and_hold_keeps_dividends_in_cash():
    result = simulate(market(), np.ones(3), 1, 3, {"20260107": 1.}, buy_once=True)
    assert result["orders"] == 1
    assert result["ending_cash"] > 9000.
