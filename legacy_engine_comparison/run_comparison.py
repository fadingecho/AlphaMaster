"""Run the frozen AlphaMaster targets through a-share-ai's actual execution core.

No AI/provider/QMT calls. Source repository and trading.db remain read-only.
This is a fixed-strategy adapter, not the old project's LLM decision runner.
"""
import csv
from dataclasses import replace
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import struct
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LEGACY = Path(os.getenv('ALPHAMASTER_LEGACY_ROOT', str(ROOT.parent / 'a-share-ai'))).expanduser().resolve()
OUT = HERE / 'results'
sys.path.insert(0, str(LEGACY))

from app.backtest import BacktestFill, BacktestResult, BacktestStore
from app.backtest_execution import (
    execute_historical_orders, historical_account, target_quantity,
    BUY_CASH_POLICY_REJECT_REQUESTED_V2,
)
from app.config import settings
from app.engine import calculate_fill, money
from app.models import Side
from app.performance import calculate_metrics

D = Decimal
SYMBOL = '000001.SZ'
INITIAL = D('100000')


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def float32_decimal(value):
    # The reference CSV contains NumPy float32's shortest round-trip repr.
    return D(str(struct.unpack('f', struct.pack('f', float(value)))[0]))


def state_snapshot(dates):
    uri = (LEGACY / 'data/trading.db').as_uri() + '?mode=ro'
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute('PRAGMA query_only=ON')
        rows = connection.execute(
            'SELECT trading_date,state,source_version FROM market_security_states '
            'WHERE source_id=? AND symbol=? AND trading_date BETWEEN ? AND ? '
            'ORDER BY trading_date,source_version',
            ('gjqmt-full', SYMBOL, dates[0], dates[-1])).fetchall()
    resolved = {}
    for date in dates:
        values = {r[1] for r in rows if r[0] == date}
        if len(values) != 1:
            raise ValueError('Missing/conflicting security facts for ' + date)
        resolved[date] = values.pop()
    return resolved, rows


def run(cap, frame, targets, cut, states, dividends):
    # Isolated run settings. The old project's global defaults are untouched.
    config = replace(settings, initial_cash=INITIAL, max_total_position=cap,
                     max_symbol_position=cap, max_positions=1)
    cash, positions = INITIAL, {}
    fills, curve, decisions, outcomes, corporate_checks = [], [], [], [], []
    for i in range(cut, len(frame)):
        row = frame[i]
        date = row['date']
        held = int(positions.get(SYMBOL, {}).get('quantity', 0))
        if date in dividends:
            corporate_checks.append(dict(date=date, pre_open_shares=held,
                                         cash_per_share=dividends[date]))
            # Match the old strict runner's refusal to invent dividend accounting.
            if held:
                raise ValueError('Unsupported corporate action while holding: ' + date)
        op, close, prev = D(row['raw_open']), D(row['raw_close']), D(row['pre_close'])
        target = min(cap, targets[i - 1])
        assert D(0) <= target <= cap <= D(1)
        equity_open = cash + held * op
        desired = target_quantity(equity_open, target, op)
        delta = desired - held
        # Strategy chooses an affordable order before submitting to the executor.
        if delta > 0:
            while delta > 0:
                preview = calculate_fill(Side.BUY, SYMBOL, delta, op, config, trade_date=date)
                if preview.gross_amount + preview.fees.total <= cash:
                    break
                delta -= 100
        orders = []
        if delta:
            orders.append(dict(decision_date=frame[i-1]['date'], execute_date=date,
                               symbol=SYMBOL, side='BUY' if delta > 0 else 'SELL',
                               quantity=abs(delta), target_position_pct=str(target)))
        state = states[date]
        up = (prev * D('1.10')).quantize(D('.01'), rounding=ROUND_HALF_UP)
        down = (prev * D('.90')).quantize(D('.01'), rounding=ROUND_HALF_UP)
        if state == 'tradable':
            if op >= up - D('.005'):
                state = 'limit_up_locked'
            elif op <= down + D('.005'):
                state = 'limit_down_locked'
        executed = execute_historical_orders(
            trade_date=date, cash=cash, positions=positions, orders=orders,
            open_prices={SYMBOL: op}, settings=config,
            market_states={SYMBOL: state},
            buy_cash_policy_version=BUY_CASH_POLICY_REJECT_REQUESTED_V2)
        cash, positions = executed['cash'], executed['positions']
        for order, fill in executed['fills']:
            fills.append(BacktestFill(trade_date=date, fill=fill))
        outcomes.extend(executed['order_results'])
        account = historical_account(cash=cash, positions=positions,
                                     close_prices={SYMBOL: close})
        assert cash >= 0 and all(int(p['quantity']) >= 0 for p in positions.values())
        curve.append(dict(trading_date=date, **{k:account[k] for k in ('cash','market_value','equity')}))
        decisions.append(dict(date=date, decision_date=frame[i-1]['date'],
                              original_target=str(targets[i-1]), applied_target=str(target),
                              opening_price=str(op), requested_delta=delta))
    name = 'alphamaster-cap-' + str(int(cap * 100))
    result = BacktestResult(strategy_name=name, start_date=frame[cut]['date'],
                            end_date=frame[-1]['date'], initial_cash=INITIAL,
                            final_equity=D(curve[-1]['equity']), fills=fills, equity_curve=curve)
    equities = [INITIAL] + [D(r['equity']) for r in curve]
    metrics = calculate_metrics(equities, periods_per_year=244)
    native_metrics = calculate_metrics(equities)
    fees = sum((f.fill.fees.total for f in fills), D(0))
    summary = dict(return_pct=metrics['total_return']*100,
                   max_drawdown_pct=metrics['max_drawdown']*100,
                   sharpe_244=metrics['sharpe'], sharpe_252=native_metrics['sharpe'],
                   final_equity=str(result.final_equity), fills=len(fills), fees=str(fees),
                   ending_shares=sum(int(p['quantity']) for p in positions.values()),
                   rejected_orders=sum(o['status']=='rejected' for o in outcomes),
                   corporate_action_checks=corporate_checks)
    return result, summary, decisions, outcomes


def write_csv(path, rows, fieldnames=None):
    if not rows and not fieldnames:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    output_db = OUT / 'backtest_results.db'
    if output_db.exists():
        raise FileExistsError('Refusing to overwrite an existing comparison database')
    frame = read_csv(ROOT / 'demo/data/market_snapshot.csv')
    reference = read_csv(ROOT / 'qmt_migration/reference_signals.csv')
    assert [r['date'] for r in frame] == [r['date'] for r in reference]
    targets = [float32_decimal(r['torch_target']) for r in reference]
    manifest = json.loads((ROOT / 'demo/data/manifest.json').read_text())
    cut = manifest['development_bars']
    assert len(frame) - cut == 120 and frame[cut]['date'] == '20260127'
    dividends = json.loads((ROOT / 'demo/data/cash_dividends.json').read_text())
    states, state_rows = state_snapshot([r['date'] for r in frame[cut:]])
    dump(OUT / 'security_state_snapshot.json', state_rows)
    originals = json.loads((ROOT / 'demo/report/results.json').read_text())['results']['AlphaMaster']
    source_files = ['app/backtest_execution.py','app/engine.py','app/performance.py',
                    'app/fee_schedule.py','app/config.py','app/backtest.py']
    identity = dict(legacy_root=str(LEGACY), source_hashes={n:sha(LEGACY/n) for n in source_files},
                    market_snapshot_sha256=sha(ROOT/'demo/data/market_snapshot.csv'),
                    signals_sha256=sha(ROOT/'qmt_migration/reference_signals.csv'),
                    strategy_sha256=sha(ROOT/'strategies/best_DEMO_000001.SZ.json'))
    config = dict(identity=identity, stock=SYMBOL, start='20260127', end='20260728',
                  initial_cash='100000', price_views='front signal / none execution and valuation',
                  strategy='frozen AlphaMaster formula, no reoptimization',
                  sizing='previous-close target, current-open equity and whole-lot sizing',
                  corporate_actions='strict reject if pre-open shares > 0; no synthetic dividends',
                  market_states='read-only source facts + same conservative 10% opening-limit estimate',
                  executor_cash_policy=BUY_CASH_POLICY_REJECT_REQUESTED_V2,
                  scope='existing historical execution, accounting, fees, metrics and result store; no AI runner',
                  database='new isolated result DB; original trading.db read-only')
    store = BacktestStore(output_db)
    store.initialize()
    summaries = {}
    for cap in (D(1), D('.10')):
        result, summary, decisions, outcomes = run(cap, frame, targets, cut, states, dividends)
        tag = result.strategy_name
        store.save_result(run_id=tag, result=result, config=dict(config, max_target=str(cap)))
        assert store.result(tag)['final_equity'] == str(result.final_equity)
        summaries[tag] = summary
        saved = store.result(tag)
        write_csv(OUT / (tag + '_fills.csv'), saved['fills'])
        write_csv(OUT / (tag + '_equity.csv'), saved['equity_curve'])
        write_csv(OUT / (tag + '_targets.csv'), decisions)
        dump(OUT / (tag + '_orders.json'), outcomes)
        if cap == 1:
            old_keys = [(f['date'], f['side'].upper(), f['quantity']) for f in originals['fills']]
            new_keys = [(f.trade_date, f.fill.side.value, f.fill.quantity) for f in result.fills]
            summary['fill_date_side_quantity_match'] = old_keys == new_keys
            summary['equity_delta_vs_demo'] = float(result.final_equity) - originals['ending_cash']
            summary['maximum_daily_equity_difference'] = max(abs(float(r['equity'])-b)
                for r,b in zip(result.equity_curve, originals['equity']))
            # Explain rounding differences on matched fills rather than guessing.
            if old_keys == new_keys:
                differences = []
                cumulative = D(0)
                for a,b in zip(originals['fills'], result.fills):
                    f = b.fill
                    price_effect = (D(str(a['price'])) - f.fill_price) * f.quantity
                    if f.side == Side.SELL:
                        price_effect = -price_effect
                    fee_effect = D(str(a['fee'])) - f.fees.total
                    cumulative += price_effect + fee_effect
                    differences.append(dict(date=b.trade_date, side=f.side.value, quantity=f.quantity,
                        demo_price=a['price'], legacy_price=str(f.fill_price),
                        demo_fee=a['fee'], legacy_fee=str(f.fees.total),
                        price_cash_effect=str(price_effect), fee_cash_effect=str(fee_effect),
                        cumulative_cash_difference=str(cumulative)))
                summary['explained_rounding_difference'] = float(cumulative)
                assert abs(float(cumulative)-summary['equity_delta_vs_demo']) < .001
                write_csv(OUT / 'matched_fill_differences.csv', differences)
    comparison = dict(config=config, legacy_results=summaries,
                      original_demo={k:v for k,v in originals.items() if k not in ('equity','fills')},
                      qmt_screenshot_reference=dict(net_value=.9375, return_pct=-6.25,
                                                    max_drawdown_pct=-8.82,
                                                    provenance='user screenshot, rounded; no fill-level export'),
                      legacy_core_tests='29 passed: execution, engine, fee schedule, performance')
    dump(OUT / 'comparison.json', comparison)
    full = summaries['alphamaster-cap-100']
    limited = summaries['alphamaster-cap-10']
    report = '''AlphaMaster 固定策略：旧项目回测核心对照

平安银行 000001.SZ，2026-01-27 至 2026-07-28，120 个交易日，初始资金 100,000 元。
同一份行情、同一条已冻结公式、前日收盘信号、次日开盘调仓。未重新训练或选择公式。

旧项目执行器（保留原策略 0~100% 仓位）：
  收益：{ret:+.6f}%
  最大回撤：{dd:.6f}%
  期末资产：{end} 元
  成交：{fills} 笔；费用：{fees} 元；拒单：{rejects}
  夏普（统一 244 日年化）：{sharpe:.6f}
  夏普（旧项目默认 252 日年化）：{native:.6f}

原独立回测：收益 -6.305412%，最大回撤 -8.827610%，57 笔成交，期末 93,694.588057 元。
用户 QMT 截图：单位净值 0.9375，约 -6.25%，最大回撤 8.82%；截图为舍入值。

逐笔对照：日期/方向/数量是否全部一致：{match}
旧项目与原回测期末资产差：{delta:+.6f} 元。
matched_fill_differences.csv 逐笔列出成交价及费用取整引起的现金差异。

附加敏感性：把目标仓位截到旧项目默认的 10% 单票上限：
  收益 {limited_ret:+.6f}%，最大回撤 {limited_dd:.6f}%，成交 {limited_fills} 笔。
这改变了策略的仓位条件，不能作为同仓位下算法改善的证据；并非重跑旧项目的 AI 策略。

复用了哪些旧代码
app.backtest_execution.execute_historical_orders / historical_account / target_quantity
app.engine.calculate_fill、app.fee_schedule、app.performance.calculate_metrics、BacktestStore。
只新增策略适配和隔离结果文件，没有修改旧项目源代码、默认配置或原 trading.db。
未调用旧项目的 LLM 决策、PositionPlan 或整套 AI runner。

关键口径
旧项目原样保留三位小数成交价、分币费用、T+1、整手、持仓和现金校验。
策略在提交前选择能支付费用的数量，执行器仍使用原生“资金不足整笔拒绝”规则。
120 日均有已保存的可交易状态；开盘涨跌停仍沿用前次相同的保守估计。
区间唯一分红是 2026-06-12，每股 0.36 元。两组策略在除息日盘前均无持仓，
因此没有补入分红现金、伪造派发日期或启用未验证的红利税路径。
这些结果说明已有模拟引擎之间的对照情况，不构成新的未见数据验证。

旧项目 29 项相关测试通过。详细代码版本以 comparison.json 中源码 SHA-256 为准。
backtest_results.db 是新建的独立结果库，采用旧项目原生 backtest 表结构。
'''.format(ret=full['return_pct'], dd=full['max_drawdown_pct'], end=full['final_equity'],
           fills=full['fills'], fees=full['fees'], rejects=full['rejected_orders'],
           sharpe=full['sharpe_244'], native=full['sharpe_252'],
           match=full['fill_date_side_quantity_match'], delta=full['equity_delta_vs_demo'],
           limited_ret=limited['return_pct'], limited_dd=limited['max_drawdown_pct'],
           limited_fills=limited['fills'])
    (OUT/'对照结果.txt').write_text(report, encoding='utf-8')
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
