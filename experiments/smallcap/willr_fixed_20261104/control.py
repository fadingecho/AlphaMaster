"""Run the corrected small and large models sequentially in this local process."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
PY=sys.executable
CHILD=None
STOP=False


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):return {}


def status(phase,**extra):
    obj=dict(phase=phase,pid=os.getpid(),updated_at=time.time(),**extra)
    tmp=ROOT/'status.json.tmp'
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    os.replace(tmp,ROOT/'status.json')


def active_pid():
    try:
        pid=int((ROOT/'training.pid').read_text())
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        return pid if str(ROOT/'control.py').encode() in cmd and b'worker' in cmd else None
    except (OSError,ValueError):return None


def stop(*_):
    global STOP
    STOP=True
    if CHILD and CHILD.poll() is None:CHILD.terminate()


def verify_sources():
    manifest=read(ROOT/'launch_manifest.json')
    if not manifest.get('files'):raise RuntimeError('Missing launch source manifest')
    for name,expected in manifest['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Launch source changed: '+name)


def worker():
    global CHILD
    lock=(ROOT/'training.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    (ROOT/'training.pid').write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    plan=read(ROOT/'plan.json')
    assert [job['arch'] for job in plan['jobs']]==['small','large']
    env=dict(os.environ,PYTHON_DOTENV_DISABLED='1',PYTHONDONTWRITEBYTECODE='1')
    commands=[]
    for job in plan['jobs']:
        rid=f"{job['arch']}_s{job['seed']}_{plan['steps']}steps_b{plan['batch']}"
        run=ROOT/'results'/rid;run.mkdir(parents=True,exist_ok=True)
        if not (run/'completed.json').exists():
            commands.append(('training',rid,run,[PY,'-B','-u',str(ROOT/'runner.py'),'--arch',job['arch'],
                '--seed',str(job['seed']),'--steps',str(plan['steps']),'--batch',str(plan['batch']),
                '--checkpoint-every',str(plan['checkpoint_interval']),'--resume'],run/'training.log'))
    commands.append(('evaluating','small_and_large',ROOT/'results',
        [PY,'-B','-u',str(ROOT/'evaluate_all.py'),'--final'],ROOT/'evaluation.log'))
    for phase,rid,run,command,log in commands:
        if STOP:status('paused',run_id=rid);return
        try:verify_sources()
        except Exception as exc:status('failed',run_id=rid,error=str(exc));return
        with log.open('a') as stream:
            CHILD=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT)
            status(phase,run_id=rid,child_pid=CHILD.pid,log=str(log))
            code=CHILD.wait();CHILD=None
        if STOP:status('paused',run_id=rid);return
        if code:status('failed',run_id=rid,exit_code=code,log=str(log));return
        if phase=='training' and not (run/'completed.json').exists():
            status('paused',run_id=rid);return
    status('completed',run_id=rid)


def main(action):
    if action=='worker':worker();return
    pid=active_pid()
    if action=='start':
        if pid:print('Already running',pid);return
        verify_sources()
        with (ROOT/'supervisor.log').open('a') as log:
            p=subprocess.Popen([PY,'-B','-u',str(ROOT/'control.py'),'worker'],cwd=ROOT,
                env=dict(os.environ,PYTHON_DOTENV_DISABLED='1',PYTHONDONTWRITEBYTECODE='1'),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print('Started sequential small-then-large training supervisor',p.pid)
    elif action=='pause':
        if pid:os.kill(pid,signal.SIGTERM);print('Pause requested; checkpoint after current step',pid)
        else:print('Not running')
    else:
        print('Supervisor PID:',pid)
        print(json.dumps(read(ROOT/'status.json'),ensure_ascii=False,indent=2))
        for p in (ROOT/'results').glob('*/progress.json'):
            print(json.dumps(read(p),ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['start','status','pause','worker'])
    main(parser.parse_args().action)
