"""Reproducible offline example. Read SQLite -> train -> frozen holdout report.

The upstream reward is a fast research approximation. This report separately
uses a long-only daily cash/share ledger, next-open fills, lots and explicit fees.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import sqlite3
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

os.environ["ALPHAMASTER_DEMO"] = "1"
os.environ.setdefault("ALPHAMASTER_CPU_THREADS", "2")
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "demo/mpl-cache"))
os.chdir(ROOT)

import numpy as np
import pandas as pd

DB = Path(os.getenv("ALPHAMASTER_DEMO_DB", str(ROOT.parent / "a-share-ai/data/trading.db"))).expanduser().resolve()
DATA = ROOT / "demo/data"
REPORT = ROOT / "demo/report"
TAG = "DEMO_000001.SZ"
DEV = DATA / "development" / f"{TAG}_D1.parquet"
FULL = DATA / "full" / f"{TAG}_D1.parquet"
STRATEGY = ROOT / "strategies" / f"best_{TAG}.json"
SEED = 20261025


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def prepare():
    if not DB.is_file():
        raise FileNotFoundError("Set ALPHAMASTER_DEMO_DB to the existing a-share-ai trading.db")
    for folder in (DEV.parent, FULL.parent, REPORT):
        folder.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True) as con:
        con.execute("PRAGMA query_only=ON")
        con.execute("BEGIN")
        bars = pd.read_sql_query(
            "SELECT bar_time,dividend_type,open,high,low,close,volume,amount,pre_close "
            "FROM market_bars WHERE source_id='gjqmt-full' AND symbol='000001.SZ' "
            "AND period='1d' ORDER BY bar_time", con)
        calendar = pd.read_sql_query(
            "SELECT bar_time FROM market_bars WHERE source_id='gjqmt-full' "
            "AND symbol='000300.SH' AND dividend_type='none' AND period='1d' "
            "ORDER BY bar_time", con).bar_time.tolist()
        events = pd.read_sql_query(
            "SELECT ex_date,cash_dividend_per_share,stock_dividend_per_share,"
            "capitalization_per_share,rights_issue_per_share FROM market_corporate_actions "
            "WHERE source_id='gjqmt-full' AND symbol='000001.SZ' ORDER BY ex_date", con)
    for col in ("open", "high", "low", "close", "volume", "amount", "pre_close"):
        bars[col] = pd.to_numeric(bars[col], errors="raise")
    f = bars[bars.dividend_type == "front"].set_index("bar_time")
    n = bars[bars.dividend_type == "none"].set_index("bar_time")
    dates = sorted(set(f.index) & set(n.index) & set(calendar))
    expected = [d for d in calendar if dates[0] <= d <= dates[-1]]
    if dates != expected:
        raise ValueError("The selected stock has missing index-calendar sessions; choose another sample.")
    f, n = f.loc[dates], n.loc[dates]
    frame = pd.DataFrame({"date": dates})
    frame["time"] = (pd.to_datetime(frame.date, format="%Y%m%d")
                     .dt.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)).astype("int64") // 10**9
    for col in ("open", "high", "low", "close"):
        frame[col] = f[col].to_numpy()
        frame["raw_" + col] = n[col].to_numpy()
    frame["tick_volume"] = n.volume.to_numpy()
    frame["pre_close"] = n.pre_close.to_numpy()
    assert len(frame) >= 600 and frame.time.is_monotonic_increasing
    assert np.isfinite(frame.drop(columns="date").to_numpy()).all()
    for prefix in ("", "raw_"):
        assert (frame[prefix + "high"] >= frame[[prefix + "open", prefix + "close"]].max(axis=1)).all()
        assert (frame[prefix + "low"] <= frame[[prefix + "open", prefix + "close"]].min(axis=1)).all()
    split = len(frame) - 120
    cols = ["time", "open", "high", "low", "close", "tick_volume"]
    frame[cols].to_parquet(FULL, index=False)
    frame.iloc[:split][cols].to_parquet(DEV, index=False)
    frame.to_csv(DATA / "market_snapshot.csv", index=False)
    events = events[(events.ex_date >= dates[0]) & (events.ex_date <= dates[-1])].copy()
    for col in events.columns[1:]:
        events[col] = pd.to_numeric(events[col], errors="raise")
    if (events[["stock_dividend_per_share", "capitalization_per_share", "rights_issue_per_share"]] != 0).any().any():
        raise ValueError("This small demo ledger only supports cash dividends, not stock/rights actions.")
    dump(DATA / "cash_dividends.json", dict(zip(events.ex_date, events.cash_dividend_per_share)))
    manifest = {
        "symbol": "000001.SZ", "name": "平安银行", "source": str(DB),
        "source_id": "gjqmt-full", "seed": SEED, "total_bars": len(frame),
        "start": dates[0], "end": dates[-1], "development_bars": split,
        "development_end": dates[split-1], "holdout_start": dates[split], "holdout_bars": 120,
        "development_sha256": hashlib.sha256(DEV.read_bytes()).hexdigest(),
        "full_sha256": hashlib.sha256(FULL.read_bytes()).hexdigest(),
        "volume": "QMT volume preserved under tick_volume for loader compatibility; not tick counts",
        "price_views": "Stored front-adjusted prices for signals; none/raw prices for ledger fills",
    }
    dump(DATA / "manifest.json", manifest)
    dump(ROOT / "web_settings.json", {
        "last_data_file": str(DEV), "last_strategy_file": str(STRATEGY),
        "debug_mode": False, "ai_api_key": "", "realtime_watches": [],
        "feishu_enabled": False, "feishu_webhook_url": "", "feishu_secret": "",
        "tqsdk_user": "", "tqsdk_password": "", "bt_commission_pct": .03,
        "bt_slippage_pct": .02,
    })
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def train():
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    from train_file import train_from_file
    engine = train_from_file(str(DEV), from_scratch=True)
    if engine is None or engine.best_formula is None:
        raise RuntimeError("Training did not produce a formula; no result will be fabricated.")
    engine.save_checkpoint(engine.train_steps)
    print("TRAINING_COMPLETE", engine._decode_formula(engine.best_formula), flush=True)


def fee(notional, sell, date):
    commission = max(5.0, notional * 0.0003)
    transfer = notional * 0.00001
    stamp = notional * (0.001 if str(date) < "20230828" else 0.0005) if sell else 0.0
    return commission + transfer + stamp


def rounded_price(price):
    return float(Decimal(str(price)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def simulate(frame, targets, start, stop, dividends, buy_once=False):
    """Independent segment, flat initial holdings, previous-close decisions.

    One rebalance per session: today's buy cannot be sold again today (T+1).
    At an opening limit price, block the corresponding trade conservatively.
    Gross cash dividends are credited on ex-date as an explicit approximation.
    """
    cash, shares = 100000.0, 0
    entered = False
    equity, fills, blocked = [], [], []
    for i in range(start, stop):
        row = frame.iloc[i]
        date = str(row.date)
        cash += shares * float(dividends.get(date, 0.0))
        op, close, prev = float(row.raw_open), float(row.raw_close), float(row.pre_close)
        target = float(np.clip(targets[i-1], 0, 1)) if i else 0.0
        value = cash + shares * op
        desired = int(value * target / op / 100) * 100
        delta = desired - shares
        if buy_once and entered:
            delta = 0
        limit_up, limit_down = rounded_price(prev * 1.10), rounded_price(prev * .90)
        if (delta > 0 and op >= limit_up - .005) or (delta < 0 and op <= limit_down + .005):
            blocked.append({"date": date, "reason": "opening_limit", "requested_shares": delta})
            delta = 0
        if delta > 0:
            price = op * 1.0002
            qty = delta
            while qty > 0 and qty * price + fee(qty * price, False, date) > cash:
                qty -= 100
            if qty > 0:
                cost = fee(qty * price, False, date)
                cash -= qty * price + cost
                shares += qty
                entered = True
                fills.append({"date": date, "side": "buy", "quantity": qty, "price": price, "fee": cost})
        elif delta < 0:
            qty = min(shares, -delta)
            price = op * .9998
            cost = fee(qty * price, True, date)
            cash += qty * price - cost
            shares -= qty
            fills.append({"date": date, "side": "sell", "quantity": qty, "price": price, "fee": cost})
        assert cash >= -1e-6 and shares >= 0 and shares % 100 == 0
        equity.append(cash + shares * close)
    curve = np.array([100000., *equity])
    returns = curve[1:] / curve[:-1] - 1
    std = returns.std(ddof=1) if len(returns) > 1 else 0
    return {
        "return_pct": float((curve[-1] / curve[0] - 1) * 100),
        "max_drawdown_pct": float((curve / np.maximum.accumulate(curve) - 1).min() * 100),
        "sharpe": float(returns.mean() / std * np.sqrt(244)) if std > 1e-12 else 0.0,
        "orders": len(fills), "fees": sum(x["fee"] for x in fills),
        "ending_shares": shares, "ending_cash": cash, "blocked_orders": len(blocked),
        "equity": equity, "fills": fills,
    }


def report():
    import torch
    from data_pipeline.parquet_manager import ParquetDataManager
    from model_core.vm import StackVM
    from strategy_manager.signal import compute_target_positions_stateless
    from model_core.vocab import FORMULA_VOCAB
    from model_core.features import MT5FeatureEngineer
    manifest = json.loads((DATA / "manifest.json").read_text())
    strategy = json.loads(STRATEGY.read_text())
    assert Path(strategy["data_file"]).resolve() == DEV.resolve()
    assert hashlib.sha256(DEV.read_bytes()).hexdigest() == manifest["development_sha256"]
    frame = pd.read_csv(DATA / "market_snapshot.csv", dtype={"date": str})
    dividends = json.loads((DATA / "cash_dividends.json").read_text())
    manager = ParquetDataManager(FULL)
    manager.load()
    torch.set_num_threads(2)
    with torch.no_grad():
        feat = manager.feat_tensor
        factor = StackVM().execute(strategy["formula"], feat)
        if factor is None:
            raise RuntimeError("Saved formula is invalid")
        # A prefix-only recomputation must not change historical features/signals.
        cut = manifest["development_bars"]
        prefix_raw = {k: v[:, :cut] for k, v in manager.raw_dict.items()}
        prefix_feat = MT5FeatureEngineer.compute_features(prefix_raw)
        prefix_factor = StackVM().execute(strategy["formula"], prefix_feat)
        torch.testing.assert_close(prefix_feat, feat[:, :, :cut], rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(prefix_factor, factor[:, :cut], rtol=2e-4, atol=2e-4)
        positions = compute_target_positions_stateless(factor)[0].numpy()
    close = frame.close
    targets = {
        "AlphaMaster": positions,
        "Buy & hold": np.ones(len(frame)),
        "SMA20 > SMA60": (close.rolling(20).mean() > close.rolling(60).mean()).astype(float).to_numpy(),
        "Momentum20 > 0": (close.pct_change(20) > 0).astype(float).to_numpy(),
        "Cash": np.zeros(len(frame)),
    }
    results = {name: simulate(frame, target, cut, len(frame), dividends, buy_once=name == "Buy & hold") for name, target in targets.items()}
    readable = " → ".join(FORMULA_VOCAB.token_names[t] for t in strategy["formula"])
    output = {"manifest": manifest, "formula_tokens": strategy["formula"], "formula": readable,
              "prefix_causality_check": "passed", "results": results,
              "note": "Single frozen 120-session holdout; no tuning on this result."}
    dump(REPORT / "results.json", output)
    pd.DataFrame({"date": frame.date.iloc[cut:].to_numpy(), **{n:r["equity"] for n,r in results.items()}}).to_csv(REPORT / "equity.csv", index=False)
    pd.DataFrame([{ "strategy": n, **fill} for n,r in results.items() for fill in r["fills"]]).to_csv(REPORT / "fills.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    dates = pd.to_datetime(frame.date.iloc[cut:])
    for name, r in results.items():
        ax.plot(dates, np.array(r["equity"]) / 100000, label=name, linewidth=2 if name == "AlphaMaster" else 1.3)
    ax.set(title="Frozen holdout · 000001.SZ · long-only", ylabel="Equity / initial capital")
    ax.legend(loc="best"); ax.grid(alpha=.2); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(REPORT / "equity.png", dpi=150); plt.close(fig)
    rows = "".join(f'<tr><td>{html.escape(n)}</td><td>{r["return_pct"]:+.2f}%</td><td>{r["max_drawdown_pct"]:.2f}%</td><td>{r["sharpe"]:.2f}</td><td>{r["orders"]}</td><td>¥{r["fees"]:.2f}</td></tr>' for n,r in results.items())
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AlphaMaster 日线体验</title>
<style>body{{font:16px/1.75 system-ui,sans-serif;background:#f5f7f6;color:#18352e;margin:0}}main{{max-width:1050px;margin:auto;padding:32px 24px}}a{{color:#086f58}}h1{{font-size:32px;margin-bottom:8px}}.card{{background:white;border:1px solid #dae4df;border-radius:12px;padding:22px;margin:20px 0}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:right;padding:10px;border-bottom:1px solid #e3e9e6}}td:first-child,th:first-child{{text-align:left}}img{{width:100%}}code{{overflow-wrap:anywhere}}.muted{{color:#60746c}}.scroll{{overflow-x:auto}}</style>
<main><a href="/">← AlphaMaster 训练控制台</a><h1>平安银行 · 日线因子挖掘体验</h1><p class="muted">短训练演示 ｜ 只做多或空仓 ｜ 未连接交易账户</p>
<div class="card"><b>数据怎么分</b><p>共 {len(frame)} 个交易日，{manifest['start']}—{manifest['end']}。前 {cut} 日交给 AlphaMaster 搜索与内部验证；最后 120 日（{manifest['holdout_start']}—{manifest['end']}）只在公式选定后评估。</p><p>公式（后缀表达式）：<code>{html.escape(readable)}</code></p><p>历史特征和公式的前缀一致性检查：通过。训练和评估使用同一份冻结数据。</p></div>
<div class="card"><b>留出区间：相同资金、费用与执行规则</b><div class="scroll"><table><tr><th>策略</th><th>收益</th><th>最大回撤</th><th>夏普</th><th>成交笔数</th><th>费用</th></tr>{rows}</table></div><img src="equity.png" alt="留出区间资金曲线"></div>
<div class="card"><b>这次回测模拟了什么</b><p>初始资金10万元；根据上一交易日收盘信号，在下一交易日开盘调仓，每日最多一次；只做多、无杠杆、100股整数手。佣金万三且最低5元、过户费万0.1，印花税按历史日期切换，单边滑点2个基点。触及按昨收估算的10%开盘涨跌停时保守阻止相应买卖。期末持仓按收盘价估值，没有强制平仓。</p><p>所选区间已检查沪深300交易日历，个股没有缺失交易日。现金分红简化为除息日按税前金额入账，未模拟实际到账日和红利税；涨跌停价格是规则估算，未取得历史官方逐日限价。信号使用库中现存前复权价格，不是逐日保存的历史时点数据。</p><p>AlphaMaster训练评分与原生回测页面使用快速近似收益模型；上表另用现金和股票账本复核，两者数值不能混为一谈。80步训练和单股120日检验只用于体验，不能证明策略有效。</p></div>
<p><a href="results.json">完整结果 JSON</a> · <a href="equity.csv">净值 CSV</a> · <a href="fills.csv">成交 CSV</a></p></main></html>'''
    (REPORT / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({n:{k:v for k,v in r.items() if k not in ("equity","fills")} for n,r in results.items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "train", "report", "all"], default="all", nargs="?")
    args = parser.parse_args()
    for name, fn in (("prepare", prepare), ("train", train), ("report", report)):
        if args.stage in (name, "all"):
            fn()
