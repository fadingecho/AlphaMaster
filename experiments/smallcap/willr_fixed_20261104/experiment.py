"""Matched-budget AlphaMaster architecture comparison on a frozen stock basket.

This is a fractional-weight research evaluator, NOT a QMT cash/share ledger.
The current-cap selection has survivorship/look-ahead bias; see universe.json.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / 'data'
sys.path.insert(0, str(ROOT / 'upstream'))
os.environ['ALPHAMASTER_DEMO'] = '1'
os.environ['ALPHAMASTER_CPU_THREADS'] = '2'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'results/mpl-cache'))

import numpy as np
import pandas as pd
import torch
from config import Config
from model_core.config import ModelConfig
from model_core.alphagpt import AlphaGPT
from model_core.engine import AlphaEngine, ConstrainedSampler
from model_core.features import MT5FeatureEngineer
from model_core.vm import StackVM
from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION
from strategy_manager.signal import compute_target_positions_stateless

ARCHITECTURES = {
    'small': dict(D_MODEL=96, N_HEADS=4, N_LAYERS=3, D_FF=192),
    'large': dict(D_MODEL=256, N_HEADS=8, N_LAYERS=6, D_FF=512),
}


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


class CachedSampler(ConstrainedSampler):
    """Same masks as upstream, one CPU->GPU transfer per batch, no tiny kernels."""
    def apply_mask_to_logits(self, logits, stack_depths, step_idx, total_steps,
                             prev_tokens=None, infected_chain_lens=None):
        if not hasattr(self, '_mask_cache'):
            self._mask_cache = {}
        masks = []
        for b, depth in enumerate(stack_depths):
            infection = infected_chain_lens[b] if infected_chain_lens else 0
            previous = prev_tokens[b] if prev_tokens else None
            key = (depth, step_idx, total_steps, previous, infection)
            if key not in self._mask_cache:
                self._mask_cache[key] = self.valid_mask(depth, step_idx, total_steps,
                    torch.device('cpu'), prev_token=previous, infected_chain_len=infection)
            masks.append(self._mask_cache[key])
        mask = torch.stack(masks).to(logits.device)
        return logits.masked_fill(~mask, -1e9)


def configure(architecture, device, steps, batch):
    for key, value in ARCHITECTURES[architecture].items():
        setattr(ModelConfig, key, value)
    ModelConfig.DEVICE = torch.device(device)
    ModelConfig.EVAL_DEVICE = torch.device('cpu')
    ModelConfig.TRAIN_STEPS = steps
    ModelConfig.BATCH_SIZE = batch
    ModelConfig.REWARD_MODE = 'standard'
    Config.A_SHARE_LONG_ONLY = True
    # Same approximate one-way proportional friction for both training arms.
    Config.COST_RATE = 0.001


def load_data():
    universe = json.loads((DATA_ROOT/'universe.json').read_text())
    symbols = [r['symbol'] for r in universe['selected']]
    stocks = {s: json.loads((DATA_ROOT/f'{s}.json').read_text()) for s in symbols}
    dates = sorted(set.union(*[set(r[0] for r in b['raw']) for b in stocks.values()]))
    fields = {k: [] for k in ['open','close','high','low','volume','raw_open','raw_close','available']}
    missing = {}
    for symbol in symbols:
        b = stocks[symbol]
        adjusted = pd.DataFrame(b['hfq'], columns=['date','open','close','high','low','volume']).set_index('date').astype(float)
        raw = pd.DataFrame(b['raw'], columns=['date','open','close','high','low','volume']).set_index('date').astype(float)
        available = pd.Series(True, index=raw.index).reindex(dates).fillna(False).to_numpy(dtype=bool)
        missing[symbol] = int((~available).sum())
        adjusted, raw = adjusted.reindex(dates), raw.reindex(dates)
        for frame in [adjusted, raw]:
            last = frame.close.ffill()
            for col in ['open','close','high','low']:
                frame[col] = frame[col].fillna(last)
            frame['volume'] = frame.volume.fillna(0)
        if adjusted.isna().any().any() or raw.isna().any().any():
            raise ValueError(f'{symbol}: cannot backfill before listing')
        for col in ['open','close','high','low','volume']:
            fields[col].append(adjusted[col].to_numpy())
        fields['raw_open'].append(raw.open.to_numpy())
        fields['raw_close'].append(raw.close.to_numpy())
        fields['available'].append(available & (raw.volume.to_numpy() > 0))
        # Validate that cap and daily providers refer to the same latest price.
        if abs(raw.close.iloc[-1] - float(b['metadata']['trade'])) > .011:
            raise ValueError(f'{symbol}: cap snapshot / daily quote date-price mismatch')
    arrays = {k: np.asarray(v) for k, v in fields.items()}
    tensors = {k: torch.tensor(arrays[k], dtype=torch.float32) for k in ['open','close','high','low','volume']}
    stamps = np.array([pd.Timestamp(d).timestamp() for d in dates])
    tensors['time'] = torch.tensor(np.tile(stamps, (len(symbols),1)), dtype=torch.float64)
    return universe, symbols, np.asarray(dates), arrays, tensors, missing


def features(raw):
    with torch.no_grad():
        return MT5FeatureEngineer.compute_features(raw)


def train(args):
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing a silent CPU fallback')
    configure(args.arch, args.device, args.steps, args.batch)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    universe, symbols, dates, arrays, raw, missing = load_data()
    # Training process never receives 2025 validation or 2026 test bars/features.
    end = int(np.searchsorted(dates, '2025-01-01'))
    train_raw = {k: v[:,:end] for k,v in raw.items()}
    feat = features(train_raw)
    ret = torch.zeros_like(train_raw['open'])
    ret[:,:-2] = torch.log(train_raw['open'][:,2:] / train_raw['open'][:,1:-1])
    # Preserve full causal indicator history, remove the initial feature warmup.
    warmup = 220
    manager = SimpleNamespace(feat_tensor=feat[:,:,warmup:], target_ret=ret[:,warmup:],
        raw_dict={k:v[:,warmup:] for k,v in train_raw.items()})
    out = ROOT/'results'/f'{args.arch}_s{args.seed}_{args.steps}steps'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'completed.json').exists():
        raise RuntimeError('Completed run already exists; use a new run identity')
    os.chdir(out)
    engine = AlphaEngine(manager, target_symbol='SMALLCAP20', n_folds=3)
    engine.sampler = CachedSampler(FORMULA_VOCAB.size, FORMULA_VOCAB.operator_offset,
        engine.vm.arity_map, engine.vm.positive_only_ids)
    parameters = sum(p.numel() for p in engine.model.parameters())
    metadata = dict(architecture=ARCHITECTURES[args.arch], architecture_name=args.arch,
        parameters=parameters, device=str(next(engine.model.parameters()).device),
        eval_device='cpu', gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        torch=torch.__version__, seed=args.seed, steps=args.steps, batch=args.batch,
        new_formula_draws=args.steps*(args.batch-max(1,int(args.batch*.25))),
        train_start=str(dates[warmup]), train_end=str(dates[end-1]),
        train_bars=end-warmup, validation_year=2025, test_year=2026,
        vocab_version=VOCAB_VERSION, missing_bars=missing,
        universe_sha256=hashlib.sha256((DATA_ROOT/'universe.json').read_bytes()).hexdigest())
    dump(out/'config.json', metadata)
    print('RUN', json.dumps(metadata, ensure_ascii=False), flush=True)
    start = time.monotonic()
    engine.train()
    if args.device == 'cuda': torch.cuda.synchronize()
    # Freeze a bounded candidate set selected only with training data.
    ranked = sorted(engine._elite_pool, key=lambda x: x[0], reverse=True)
    unique = []
    for f in [engine.best_formula] + [x[2] for x in ranked]:
        if f and list(f) not in unique:
            unique.append(list(f))
        if len(unique) == 20:
            break
    metadata.update(seconds=time.monotonic()-start, candidate_formulas=unique,
        best_training_score=engine.best_score,
        gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if args.device=='cuda' else 0,
        gpu_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if args.device=='cuda' else 0,
        restarts=engine._restart_count)
    torch.save({'state_dict':engine.model.state_dict(), 'config':metadata}, out/'policy.pt')
    dump(out/'completed.json', metadata)
    print('COMPLETE', out, metadata['seconds'], flush=True)


def stats(pnl, exposure=None, turnover=None):
    equity = np.cumprod(1+np.asarray(pnl))
    peaks = np.maximum.accumulate(np.r_[1.,equity])[1:]
    sd = np.std(pnl, ddof=1) if len(pnl)>1 else 0
    return dict(total_return=float(equity[-1]-1), max_drawdown=float(np.min(equity/peaks-1)),
        sharpe=float(np.mean(pnl)/sd*np.sqrt(244)) if sd>1e-12 else 0,
        sessions=len(pnl), average_exposure=float(np.mean(exposure)) if exposure is not None else 0,
        total_one_way_turnover=float(np.sum(turnover)) if turnover is not None else 0)


def research_backtest(targets, arrays, dates, start_date, end_date, cost_rate=.001):
    """Signal at close t -> weights at open t+1 -> total-return open t+2.

    One rebalance per day, no shorts/leverage. Fractional shares, flat 10bps per
    one-way notional change. No minimum commission/lot rounding/order book model.
    Gaps with missing raw quotes freeze that sleeve's weight. This remains a
    return-index approximation: it is not an executable corporate-action ledger.
    """
    n,t = targets.shape
    weight = np.zeros(n)
    pnl, exposures, turnovers, period_dates, blocked = [], [], [], [], 0
    for execution in range(1,t-1):
        if not (start_date <= dates[execution] <= end_date and dates[execution+1] <= end_date):
            continue
        desired = targets[:,execution-1] / n
        op = arrays['raw_open'][:,execution]
        previous = arrays['raw_close'][:,execution-1]
        available = arrays['available'][:,execution]
        # Conservative main-board +/-10% diagnostic, not authoritative historic ST/limits.
        locked_up = op >= np.round(previous*1.1,2)-.005
        locked_down = op <= np.round(previous*.9,2)+.005
        frozen = (~available) | ((desired>weight)&locked_up) | ((desired<weight)&locked_down)
        blocked += int(np.count_nonzero(frozen & (np.abs(desired-weight)>1e-9)))
        effective = np.where(frozen,weight,desired)
        # Locked holdings may have drifted above their target. Allocate only
        # remaining cash to buys after sells, rather than implicitly borrowing.
        base = np.minimum(effective,weight)
        buys = np.maximum(effective-weight,0)
        if buys.sum()>0:
            effective = base + buys*min(1.,max(0.,1.-base.sum())/buys.sum())
        assert (effective>=0).all() and effective.sum()<=1.000001
        turnover = np.abs(effective-weight).sum()
        returns = arrays['open'][:,execution+1]/arrays['open'][:,execution]-1
        pnl.append(float(effective@returns - turnover*cost_rate))
        exposures.append(effective.sum());turnovers.append(turnover);period_dates.append(dates[execution+1])
        # Drift current weights with returns before the next open rebalance.
        portfolio_return = float(effective@returns)
        weight = effective*(1+returns)/(1+portfolio_return)
        # Price drift can make an individual sleeve exceed 1/N; the next rebalance
        # can restore it. The sum remains <=1 without leverage.
    result = stats(np.asarray(pnl),exposures,turnovers)
    result['blocked_weight_changes'] = blocked
    return result, pd.DataFrame(dict(date=period_dates,pnl=pnl,exposure=exposures,turnover=turnovers))


def evaluate(args):
    configure('small','cpu',args.steps,args.batch)
    universe,symbols,dates,arrays,raw,missing = load_data()
    # Validation features have no access to test bars. Candidate selection is
    # frozen on disk before test features or test strategy returns are computed.
    val_end = int(np.searchsorted(dates,'2026-01-01'))
    val_feat = features({k:v[:,:val_end] for k,v in raw.items()})
    vm = StackVM()
    chosen = {}
    for arch in ['small','large']:
        run = ROOT/'results'/f'{arch}_s{args.seed}_{args.steps}steps'
        metadata = json.loads((run/'completed.json').read_text())
        candidates = []
        for formula in metadata['candidate_formulas']:
            factor = vm.execute(formula,val_feat)
            if factor is None: continue
            target = compute_target_positions_stateless(factor).numpy()
            metrics,_ = research_backtest(target,{k:v[:,:val_end] for k,v in arrays.items()},dates[:val_end],'2025-01-01','2025-12-31')
            candidates.append(dict(formula=formula,validation=metrics))
        # Predeclared single selection criterion, with return/drawdown retained.
        candidates.sort(key=lambda x:(-x['validation']['sharpe'],x['formula']))
        chosen[arch] = dict(candidates[0], training=metadata,
            tokens=[FORMULA_VOCAB.token_names[t] for t in candidates[0]['formula']])
        dump(run/'validation_candidates.json',candidates)
    dump(ROOT/'results/frozen_selection.json',chosen)
    full_feat = features(raw)
    all_targets = {}
    causality = {}
    for arch,item in chosen.items():
        factor = vm.execute(item['formula'],full_feat)
        val_factor = vm.execute(item['formula'],val_feat)
        error = float((factor[:,:val_end]-val_factor).abs().max())
        if error>1e-4: raise AssertionError(f'Future-prefix mismatch {arch}: {error}')
        causality[arch] = error
        all_targets[arch] = compute_target_positions_stateless(factor).numpy()
    close = pd.DataFrame(arrays['close'].T)
    all_targets['equal_weight'] = np.ones_like(arrays['close'])
    all_targets['sma20_60'] = (close.rolling(20).mean()>close.rolling(60).mean()).to_numpy().T.astype(float)
    all_targets['momentum20'] = (close.pct_change(20)>0).to_numpy().T.astype(float)
    metrics, stressed = {}, {}
    for name,target in all_targets.items():
        result,curve = research_backtest(target,arrays,dates,'2026-01-01','2026-10-23')
        metrics[name] = result
        stressed[name],_ = research_backtest(target,arrays,dates,'2026-01-01','2026-10-23',cost_rate=.002)
        curve.to_csv(ROOT/'results'/f'{name}_test_curve.csv',index=False)
    result = dict(universe_warning=universe['selection_warning'],
        model_comparison='one seed, matched sampling budget; preliminary capacity experiment',
        backtest='hfq total-return proxy, fractional weights, daily next-open, long-only, 10bps one-way; not a QMT cash/share ledger',
        limitations=['current-cap and survival selection bias','no historical ST membership or authoritative daily price limits',
            'no 100-share lots, minimum commission, dividend tax or order-book fills','training reward differs from final evaluator'],
        validation_rule='Highest 2025 Sharpe among at most 20 training-selected formulas; test year 2026 not used to select',
        missing_bars=missing, selected=chosen, prefix_error=causality, test=metrics,
        test_with_double_cost=stressed)
    dump(ROOT/'results/comparison.json',result)
    print(json.dumps(metrics,ensure_ascii=False,indent=2))
    print('Evaluation complete. Render curves with render_report.py using the existing CPU environment.')


def smoke(args):
    configure('large','cuda',args.steps,args.batch)
    model=AlphaGPT().cuda()
    x=torch.zeros((32,8),dtype=torch.long,device='cuda')
    logits,_,_=model(x);loss=logits.square().mean();loss.backward()
    assert torch.isfinite(logits).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    sampler=ConstrainedSampler(FORMULA_VOCAB.size,FORMULA_VOCAB.operator_offset,StackVM().arity_map,StackVM().positive_only_ids)
    cached=CachedSampler(sampler.vocab_size,sampler.feat_offset,sampler.arity_map,sampler.positive_only_ids)
    for step in range(8):
        depth=[0,1,2,3];infection=[0,1,2,3];lg=torch.randn(4,sampler.vocab_size)
        expected=sampler.apply_mask_to_logits(lg,depth,step,8,infected_chain_lens=infection)
        actual=cached.apply_mask_to_logits(lg.cuda(),depth,step,8,infected_chain_lens=infection).cpu()
        assert torch.equal(expected,actual)
    # Timing: a last-day signal must never earn the preceding day's return.
    sample_dates=np.array(['2026-01-01','2026-01-02','2026-01-03','2026-01-04'])
    a={'open':np.array([[10.,10.,20.,20.]]),'raw_open':np.ones((1,4))*10,
       'raw_close':np.ones((1,4))*10,'available':np.ones((1,4),dtype=bool)}
    late=np.array([[0.,1.,0.,0.]])
    m,c=research_backtest(late,a,sample_dates,sample_dates[0],sample_dates[-1])
    assert np.allclose(c.pnl,[0.,-.001]), 'signal accidentally executed before observation'
    early=np.ones((1,4))
    m,c=research_backtest(early,a,sample_dates,sample_dates[0],sample_dates[-1])
    assert abs(c.pnl.iloc[0]-.999)<1e-10 and c.exposure.max()<=1
    report=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                large_parameters=sum(p.numel() for p in model.parameters()),
                cuda_forward_backward='passed',sampler_mask_equivalence='passed',
                evaluator_timing_cost_and_no_leverage='passed')
    dump(ROOT/'results/gpu_smoke.json',report);print(report)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['train','evaluate','smoke'])
    parser.add_argument('--arch',choices=['small','large'],default='small')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--seed',type=int,default=20261025)
    args=parser.parse_args()
    globals()[args.command](args)
