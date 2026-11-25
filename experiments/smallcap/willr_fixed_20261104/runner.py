"""Resumable, matched-budget long runs. No account access or trading calls."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import time
from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter

import experiment as ex
from experiment import torch, np, ModelConfig, AlphaEngine, CachedSampler, FORMULA_VOCAB

ROOT = Path(__file__).resolve().parent
NORMALIZATION_VERSION = 'neutral-zero-v1'
FEATURE_VERSION = 'willr-sign-v1'
MILESTONES = (1000, 3000, 6000, 9000)
STOP_REQUESTED = False


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)


def source_hash():
    paths=[ROOT/'runner.py',ROOT/'experiment.py']+sorted((ROOT/'upstream').rglob('*.py'))
    h=hashlib.sha256()
    for p in paths:
        h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def run_id(arch,seed,steps=9000,batch=192):
    return f'{arch}_s{seed}_{steps}steps_b{batch}'


def candidates(engine):
    result=[]
    for formula in [engine.best_formula]+[x[2] for x in sorted(engine._elite_pool,key=lambda x:x[0],reverse=True)]:
        if formula and list(formula) not in result:result.append(list(formula))
        if len(result)==20:break
    return result


class LongEngine(AlphaEngine):
    """Cache duplicate evaluations within a batch; preserve all sample weights."""
    def _eval_formula_task(self,idx,fml,*args,**kwargs):
        if idx==0:self._step_cache={}
        key=tuple(fml)
        if key not in self._step_cache:
            self._step_cache[key]=super()._eval_formula_task(idx,fml,*args,**kwargs)
        return dict(self._step_cache[key],idx=idx)

    def save_full(self,path,step,fingerprint):
        payload={
            'schema':1,'step':step,'fingerprint':fingerprint,
            'model':self.model.state_dict(),'optimizer':self.opt.state_dict(),
            'best_score':self.best_score,'best_formula':self.best_formula,
            'attrs':{key:getattr(self,key,None) for key in [
                '_best_snapshot','factor_pool','_factor_pool_counter','_elite_pool',
                '_elite_counter','_restart_count','_best_update_step','_stagnation_steps',
                '_reward_ema','_reward_ema_step','_low_entropy_streak','_prev_init_dist',
                'wall_seconds','peak_mib']},
            'history':self.training_history,
            'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),
            'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),
        }
        tmp=path.with_suffix('.tmp');path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(payload,tmp)
        if path.exists():os.replace(path,path.with_name('previous.pt'))
        os.replace(tmp,path)

    def load_full(self,path,fingerprint):
        # Only checkpoints created in this local experiment are accepted.
        p=torch.load(path,map_location='cpu',weights_only=False)
        if p['schema']!=1 or p['fingerprint']!=fingerprint:
            raise ValueError('Resume configuration/source/data fingerprint mismatch')
        self.model.load_state_dict(p['model'],strict=True)
        self.opt.load_state_dict(p['optimizer'])
        self.best_score=p['best_score'];self.best_formula=p['best_formula']
        for k,v in p['attrs'].items():setattr(self,k,v)
        self.training_history=p['history']
        # CPU pools stay on CPU; the model/optimizer stay on CUDA.
        random.setstate(p['rng_python']);np.random.set_state(p['rng_numpy'])
        torch.set_rng_state(p['rng_torch']);torch.cuda.set_rng_state_all(p['rng_cuda'])
        return p['step']


def build(args):
    ex.configure(args.arch,'cuda',args.steps,args.batch)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed);torch.cuda.reset_peak_memory_stats()
    u,symbols,dates,arrays,raw,missing=ex.load_data()
    end=int(np.searchsorted(dates,'2025-01-01'));warmup=220
    # Only the training slice reaches feature/target construction and rewards.
    raw={k:v[:,:end] for k,v in raw.items()}
    feat=ex.features(raw);ret=torch.zeros_like(raw['open'])
    ret[:,:-2]=torch.log(raw['open'][:,2:]/raw['open'][:,1:-1])
    manager=SimpleNamespace(feat_tensor=feat[:,:,warmup:],target_ret=ret[:,warmup:],
        raw_dict={k:v[:,warmup:] for k,v in raw.items()})
    engine=LongEngine(manager,target_symbol='SMALLCAP20',n_folds=3)
    engine.sampler=CachedSampler(FORMULA_VOCAB.size,FORMULA_VOCAB.operator_offset,
        engine.vm.arity_map,engine.vm.positive_only_ids)
    config=dict(normalization_version=NORMALIZATION_VERSION,feature_version=FEATURE_VERSION,
        architecture_name=args.arch,architecture=ex.ARCHITECTURES[args.arch],
        parameters=sum(p.numel() for p in engine.model.parameters()),seed=args.seed,
        steps=args.steps,batch=args.batch,device='cuda',eval_device='cpu',
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        optimizer='AdamW lr=0.001 (upstream default, same for both sizes)',
        reward_mode='standard',formula_length=8,folds=3,one_way_cost=.001,
        new_formula_draws=args.steps*(args.batch-args.batch//4),
        train_start=str(dates[warmup]),train_end=str(dates[end-1]),train_bars=end-warmup,
        validation_year=2025,reference_test_year=2026,
        test_warning='2026 previously observed; retrospective reference, not a fresh blind test',
        universe_sha256=hashlib.sha256((ex.DATA_ROOT/'universe.json').read_bytes()).hexdigest(),
        source_sha256=source_hash(),vocab_version=ex.VOCAB_VERSION,missing_bars=missing)
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    return engine,config,fingerprint


class Paused(Exception):pass


def main(args):
    global STOP_REQUESTED
    def request_stop(*_):
        global STOP_REQUESTED
        STOP_REQUESTED=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    out=ROOT/'results'/(args.run_name or run_id(args.arch,args.seed,args.steps,args.batch))
    out.mkdir(parents=True,exist_ok=True);os.chdir(out)
    if (out/'completed.json').exists():
        print('Already complete:',out,flush=True);return
    engine,config,fingerprint=build(args)
    previous_config=out/'config.json'
    if previous_config.exists() and json.loads(previous_config.read_text())!=config:
        raise ValueError('Existing run configuration differs; refuse overwrite')
    atomic_json(previous_config,config)
    start_step=0;latest=out/'checkpoints/latest.pt'
    if args.resume:
        checkpoint=latest if latest.exists() else latest.with_name('previous.pt')
        if checkpoint.exists():start_step=engine.load_full(checkpoint,fingerprint)
    elif latest.exists():raise ValueError('Checkpoint exists; use --resume or a new run name')
    writer=SummaryWriter(log_dir=str(ROOT/'tensorboard'/out.name),
        purge_step=start_step+1 if start_step else None,flush_secs=10)
    writer.add_text('configuration',json.dumps(config,ensure_ascii=False,indent=2),start_step)
    base_seconds=getattr(engine,'wall_seconds',0.) or 0.
    start=time.monotonic();engine.peak_mib=getattr(engine,'peak_mib',0.) or 0.
    stop_at=min(args.steps,args.stop_after) if args.stop_after else args.steps
    if stop_at<start_step:raise ValueError('Requested stopping point precedes checkpoint')
    def progress(e,step,status='running'):
        e.wall_seconds=base_seconds+time.monotonic()-start
        e.peak_mib=max(e.peak_mib,torch.cuda.max_memory_allocated()/2**20)
        payload=dict(status=status,updated_at=time.time(),pid=os.getpid(),run_id=out.name,
            step=step,total_steps=args.steps,batch=args.batch,architecture=args.arch,
            seed=args.seed,seconds=e.wall_seconds,seconds_per_step=e.wall_seconds/max(step,1),
            best_training_score=e.best_score if np.isfinite(e.best_score) else None,
            restarts=e._restart_count,peak_allocated_mib=e.peak_mib,
            last_entropy=e.training_history.get('entropy',[None])[-1] if e.training_history.get('entropy') else None,
            unique_evaluations_last_batch=len(getattr(e,'_step_cache',{})))
        atomic_json(out/'progress.json',payload)
    def hook(e,step):
        progress(e,step)
        tags={
            'avg_reward':'reward/batch_mean', 'val_score':'reward/train_fold_validation_mean',
            'best_score':'reward/best_train_fold_validation', 'entropy':'exploration/token_entropy',
            'ic_mean':'factor/ic_mean', 'ic_stability':'factor/ic_stability',
            'elite_pool_size':'formulas/elite_pool_size', 'init_entropy':'exploration/initial_entropy',
            'kl_uniform':'exploration/kl_uniform', 'kl_prev':'exploration/kl_previous',
            'top1_prob':'exploration/top1_probability', 'eff_vocab':'exploration/effective_vocab',
            'batch_uniq_fmls':'formulas/unique_new_formulas', 'batch_fml_div':'formulas/diversity',
        }
        scalars=dict(getattr(e,'_last_train_metrics',{}))
        for key,tag in tags.items():
            values=e.training_history.get(key,[])
            if values:scalars[tag]=values[-1]
        valid_factors=[r['res'] for r in e._step_cache.values() if 'res' in r]
        if valid_factors:
            scalars['normalization/neutral_day_fraction_valid_unique_formulas']=float(
                torch.stack([(f==0).all(dim=0).float().mean() for f in valid_factors]).mean())
        scalars.update({'runtime/seconds_per_step':e.wall_seconds/step,
            'runtime/peak_allocated_mib':e.peak_mib,'runtime/restarts':e._restart_count,
            'formulas/unique_evaluations':len(e._step_cache)})
        for tag,value in scalars.items():
            if np.isfinite(value):writer.add_scalar(tag,value,step)
        if step==1 or step%10==0:writer.flush()
        if step in MILESTONES or step==stop_at:
            atomic_json(out/f'candidates_{step:05d}.json',dict(step=step,formulas=candidates(e),
                best_training_score=e.best_score,source='training only; validation not fed back'))
        if step%args.checkpoint_every==0 or step in MILESTONES or step==stop_at or STOP_REQUESTED:
            e.save_full(latest,step,fingerprint)
            if step in MILESTONES:shutil.copy2(latest,latest.with_name(f'milestone_{step:05d}.pt'))
        if STOP_REQUESTED:raise Paused()
    progress(engine,start_step)
    print('RUN',json.dumps(config,ensure_ascii=False),'START',start_step,flush=True)
    try:
        engine.train(start_step=start_step,end_step=stop_at,step_hook=hook)
    except Paused:
        step=len(engine.training_history['step']);progress(engine,step,'paused')
        print('Paused with resumable checkpoint',step,flush=True);return
    finally:
        writer.close()
    torch.cuda.synchronize()
    progress(engine,stop_at,'completed' if stop_at==args.steps else 'paused')
    if stop_at==args.steps:
        result=dict(config,seconds=engine.wall_seconds,gpu_peak_allocated_mib=engine.peak_mib,
            candidate_formulas=candidates(engine),best_training_score=engine.best_score,
            restarts=engine._restart_count,final_step=stop_at)
        atomic_json(out/'completed.json',result)
        print('COMPLETE',out,'SECONDS',engine.wall_seconds,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--arch',choices=['small','large'],required=True)
    p.add_argument('--seed',type=int,default=20261025)
    p.add_argument('--steps',type=int,default=9000)
    p.add_argument('--batch',type=int,default=192)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after',type=int)
    p.add_argument('--run-name')
    p.add_argument('--checkpoint-every',type=int,default=200)
    main(p.parse_args())
