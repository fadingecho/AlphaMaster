"""Validation milestones and retrospective 2026 comparison of all fixed seeds."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import experiment as ex
from runner import atomic_json, run_id

ROOT=Path(__file__).resolve().parent


def main(final=False):
    plan=json.loads((ROOT/'plan.json').read_text())
    ex.configure(plan['jobs'][0]['arch'],'cpu',plan['steps'],plan['batch'])
    u,symbols,dates,arrays,raw,missing=ex.load_data()
    val_end=int(np.searchsorted(dates,'2026-01-01'))
    val_feat=ex.features({k:v[:,:val_end] for k,v in raw.items()})
    val_arrays={k:v[:,:val_end] for k,v in arrays.items()}
    vm=ex.StackVM();selected={};stage_rows=[]
    for job in plan['jobs']:
        rid=run_id(job['arch'],job['seed'],plan['steps'],plan['batch'])
        run=ROOT/'results'/rid
        if not (run/'completed.json').exists():continue
        config=json.loads((run/'completed.json').read_text())
        for stage in sorted(run.glob('candidates_*.json')):
            data=json.loads(stage.read_text());output=run/f"validation_{data['step']:05d}.json"
            if output.exists():scores=json.loads(output.read_text())
            else:
                scores=[]
                for formula in data['formulas']:
                    factor=vm.execute(formula,val_feat)
                    if factor is None:continue
                    target=ex.compute_target_positions_stateless(factor).numpy()
                    metrics,_=ex.research_backtest(target,val_arrays,dates[:val_end],'2025-01-01','2025-12-31')
                    scores.append(dict(formula=formula,tokens=[ex.FORMULA_VOCAB.token_names[t] for t in formula],validation=metrics))
                scores.sort(key=lambda x:(-x['validation']['sharpe'],x['formula']))
                atomic_json(output,scores)
            if not scores:raise RuntimeError('No valid candidates in '+str(stage))
            stage_rows.append(dict(run_id=rid,arch=job['arch'],seed=job['seed'],step=data['step'],**scores[0]))
            if data['step']==plan['steps']:selected[rid]=dict(scores[0],training=config)
    atomic_json(ROOT/'results/validation_stages.json',stage_rows)
    if not final:return
    if len(selected)!=len(plan['jobs']):raise RuntimeError('Final evaluation requires every planned run')
    # Selection is frozen before constructing full features or 2026 returns.
    atomic_json(ROOT/'results/frozen_selection.json',selected)
    full_feat=ex.features(raw);targets={};prefix_errors={}
    for rid,item in selected.items():
        f=vm.execute(item['formula'],full_feat);v=vm.execute(item['formula'],val_feat)
        checks={}
        for cutoff in ['2026-01-01','2026-04-01','2026-08-03','2026-10-22']:
            end=int(np.searchsorted(dates,cutoff))
            prefix=v if end==val_end else vm.execute(item['formula'],ex.features({k:a[:,:end] for k,a in raw.items()}))
            err=float((f[:,:end]-prefix).abs().max())
            checks[cutoff]=err
            if err>1e-4:raise AssertionError(f'Causal prefix mismatch {rid} {cutoff}: {err}')
        prefix_errors[rid]=checks
        targets[rid]=ex.compute_target_positions_stateless(f).numpy()
    close=pd.DataFrame(arrays['close'].T)
    targets['equal_weight']=np.ones_like(arrays['close'])
    targets['sma20_60']=(close.rolling(20).mean()>close.rolling(60).mean()).to_numpy().T.astype(float)
    targets['momentum20']=(close.pct_change(20)>0).to_numpy().T.astype(float)
    metrics={};stress={}
    for rid,target in targets.items():
        metrics[rid],curve=ex.research_backtest(target,arrays,dates,'2026-01-01','2026-10-23')
        stress[rid],_=ex.research_backtest(target,arrays,dates,'2026-01-01','2026-10-23',cost_rate=.002)
        curve.to_csv(ROOT/'results'/f'{rid}_reference_curve.csv',index=False)
    aggregate={}
    for arch in sorted({job['arch'] for job in plan['jobs']}):
        values=[metrics[rid] for rid,item in selected.items() if item['training']['architecture_name']==arch]
        aggregate[arch]={}
        for k in ['total_return','max_drawdown','sharpe','average_exposure']:
            a=np.array([v[k] for v in values]);aggregate[arch][k]=dict(mean=float(a.mean()),std=float(a.std(ddof=1)) if len(a)>1 else None,min=float(a.min()),max=float(a.max()))
    atomic_json(ROOT/'results/comparison.json',dict(plan=plan,selected=selected,aggregate=aggregate,
        reference_2026=metrics,double_cost_2026=stress,validation_stages=stage_rows,prefix_errors=prefix_errors,
        caveats=['2026 already observed: retrospective comparison, not a fresh blind test',
                 u['selection_warning'],'Fractional-weight hfq research return index; not a QMT cash/share ledger',
                 'No lots/minimum commission/dividend tax/order book or historical ST and precise limit data',
                 'One seed per architecture for matched comparison; no cross-seed robustness estimate']))
    print('All fixed-seed comparisons completed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true')
    main(p.parse_args().final)
