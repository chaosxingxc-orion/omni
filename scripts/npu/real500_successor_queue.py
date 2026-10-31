"""Detached real500 successor queue using shared deployment and scientific gates.

Runs TP4-128 then TP8-256 after each predecessor releases generation custody.
Official CPU scoring may continue in its isolated tree. Conditional384 remains
held for a result-based decision. Failed steps are never replayed automatically.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import time

D=Path('/mnt/cephfs/xingchao/omni')
U=D/'publish/omni';IMAGE='sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026'


def read(p):return json.loads(p.read_bytes())
def write(p,obj):
    with p.open('x') as f:json.dump(obj,f,indent=2);f.flush();os.fsync(f.fileno())
def status(root,state,**extra):
    p=root/'status.tmp';p.write_text(json.dumps({'state':state,'pid':os.getpid(),'time':time.time(),**extra}));p.replace(root/'status.json')


def call(args,log,env=None):
    with log.open('x') as f:subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)


def formal(tree,args,log,profile=None):
    env={**os.environ,'TREE':str(tree),'CPUSET':'32-63','PORT_CONTAINER_USER':'0:0','PORT_IMAGE':IMAGE,'PORT_DOCKER_FLAGS':'--pid host --network host'}
    if profile:env['SAEA_DEPLOYMENT_PROFILE']=profile
    call(['bash',str(D/'work/scripts/port_py.sh'),*args],log,env)


def git(study,*args):return subprocess.check_output(['git','-c','safe.directory='+str(study),'-C',str(study),*args],text=True).rstrip()


def wait_seal(spec,terminal,root):
    directory=D/'runs'/spec['run_id']/spec['attempt_id']
    generation=directory/(spec['run_id']+'.generation-manifest.json')
    while True:
        if terminal.exists() and terminal.read_text().strip()!='0':raise ValueError('predecessor failed; preserve checkpoints')
        if generation.exists() and (directory/'score-jobs.jsonl').exists():
            with (D/'locks/generation.guard').open('r+') as guard:
                try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:pass
                else:
                    if not (D/'locks/flight.lock').exists():
                        doc=read(generation)
                        if doc.get('status')!='COMPLETED' or doc['run_id']!=spec['run_id']:raise ValueError('invalid predecessor seal')
                        write(root/'predecessor-seal.json',{'manifest':str(generation),'sha256':hashlib.sha256(generation.read_bytes()).hexdigest(),'observed_at':time.time(),'cpu_scoring_may_continue':True})
                        return generation
        time.sleep(600)


def capture(study,transition,out):
    out.mkdir(exist_ok=False)
    with (D/'locks/generation.guard').open('r+') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (D/'locks/flight.lock').exists():raise ValueError('flight lease present')
        plan=read(transition/'plan.json');ids=read(transition/'candidate-container-ids.json')
        names=[c['name'] for c in plan['after']]
        live=json.loads(subprocess.check_output(['docker','inspect',*names]));observations={}
        for c,d in zip(plan['after'],live):
            if d['Id']!=ids[c['name']] or d['Image']!=IMAGE or not d['State']['Running'] or d['RestartCount']:raise ValueError('target identity changed')
            if d['Config'].get('Labels',{}).get('saea.deployment.profile')!=plan['target']:raise ValueError('target ownership changed')
            log=subprocess.run(['docker','logs',c['name']],capture_output=True,check=True);raw=log.stdout+log.stderr
            (out/(c['name']+'.startup.log')).write_bytes(raw)
            observations[c['name']]={'pid':d['State']['Pid'],'started_at':d['State']['StartedAt'],'startup_stdout_sha256':hashlib.sha256(raw).hexdigest()}
        write(out/'observations.json',observations);write(out/'docker-identity.json',live)
        archived=json.loads(subprocess.check_output(['docker','inspect',*[plan['old_names'][c['name']] for c in plan['before']]]));write(out/'archived-inspect.json',archived)
        args=['docker','run','--rm','--pid','host','--network','host','--user','0:0','--cpuset-cpus','32-47','-v',f'{D}:{D}:ro','-v',f'{out}:{out}','-w',str(study),'-e',f'PYTHONPATH={study}/src','-e','PYTHONDONTWRITEBYTECODE=1','-e',f'SPEECHRL_DATA_DIR={D}','--entrypoint','/opt/saea-venv/bin/python',IMAGE,'scripts/write_vllm_runtime_receipt.py']
        for name,o in observations.items():args+=['--instance',f"{name}={o['pid']}"]
        args+=['--model-lock-key','qwen3-omni-30b-a3b-thinking-bf16','--materialization-root',str(D),'--image-ref',IMAGE,'--image-id',IMAGE,'--namespace-policy','shared-pid-host','--output',str(out/'runtime-receipt.json')]
        call(args,out/'receipt.log')
        write(out/'complete.json',{'runtime_receipt_sha256':hashlib.sha256((out/'runtime-receipt.json').read_bytes()).hexdigest(),'model_calls':0})


def run(tree,root):
    study=tree/'studies/speech-aware-evidence-acquisition'
    previous='slurp-real500-tp4-s96-v1';terminal=D/'work/codex/host-recovery-client-v1/run-real500-tp4-s96.exit'
    previous_started=D/'work/codex/real500-tp4-s96-run-v1/started.json'
    if git(study,'status','--porcelain'):raise ValueError('dirty successor tree')
    for name in ('slurp-real500-tp4-s128-v1','slurp-real500-tp8-s256-v1'):
        item=root/name;item.mkdir(exist_ok=False)
        prev=read(study/'docs/readiness'/previous/'candidate.json');spec=read(study/'docs/readiness'/name/'candidate.json')
        if prev['split_identity_hash']!=spec['split_identity_hash'] or spec['rows']!=500:raise ValueError('paired roster mismatch')
        status(root,'waiting_predecessor_generation_seal',candidate=name)
        seal=wait_seal(prev,terminal,item)
        # Verify complete checkpoint independently before changing any serving process.
        elapsed=seal.stat().st_mtime-read(previous_started)['time']
        if elapsed<=0:raise ValueError('invalid predecessor run boundary')
        formal(tree,['scripts/analysis/real_engine_progress.py','--checkpoint',str(seal.parent/(prev['run_id']+'.rows.ckpt.jsonl')),'--roster',str(D/'work/codex/real-engine-500-v1/roster.json'),'--state','complete','--elapsed-seconds',str(elapsed),'--cards','8','--instances','2' if 'tp4' in previous else '1','--output',str(item/'predecessor-complete-check.json')],item/'predecessor-check.log')
        metric_args=['python3',str(U/'scripts/npu/real_engine_metrics.py'),'--port','8200']
        if 'tp4' in previous:metric_args+=['--port','8201']
        metric_args+=['--output',str(item/'predecessor-metrics-at-release.json')]
        call(metric_args,item/'predecessor-metrics.log')
        status(root,'switching_topology',candidate=name)
        transition=D/'work/codex/frozen-engine-deployment'/('queue-'+name)
        switch=U/'scripts/npu/freeze_engine/topology_switch.py'
        call(['python3',str(switch),'plan','--source',prev['profile'],'--target',spec['profile'],'--record',str(transition)],item/'plan.log')
        call(['python3',str(switch),'apply','--record',str(transition)],item/'apply.log')
        runtime=item/'runtime';status(root,'capturing_runtime',candidate=name);capture(study,transition,runtime)
        formal(tree,['scripts/analysis/register_real_engine_resident.py','--repo',str(study),'--data',str(D),'--candidate',name,'--previous-candidate',previous,'--runtime',str(runtime),'--transition',str(transition),'--archived-inspect',str(runtime/'archived-inspect.json')],item/'register.log')
        git(study,'add','--','docs/'+spec['runtime_receipt']);git(study,'commit','-m','chore: bind observed '+name+' runtime')
        status(root,'admitting',candidate=name)
        formal(tree,['scripts/analysis/admit_real_engine_candidate.py','--repo',str(study),'--data',str(D),'--candidate',name],item/'admit.log')
        git(study,'diff','--check');git(study,'add','--','configs','docs/readiness','docs/exposure-ledger.md');git(study,'commit','-m','chore: admit prepared '+name)
        formal(tree,['scripts/checks/flight_registration_check.py','--repo-root',str(study),'--data-root',str(D),'--cell',name,'--json'],item/'registration-check.log')
        formal(tree,['-m','speech_aware_evidence_acquisition.core.flight.cli','check','--repo-root',str(study),'--data-root',str(D),'--runtime-root',str(D/'work/scratch/runtime-root'),'--lock-path',str(tree/'docs/datasets.lock.json'),'--experiment',name,'--cell',name,'--read-only'],item/'preflight.log',spec['profile'])
        # Run separately so this coordinator can prepare the next topology while CPU scoring finishes.
        script=item/'run.sh';ports=['8200','8201'] if 'tp4' in name else ['8200']
        ports_args=' '.join('--port '+p for p in ports)
        script.write_text(f'''#!/usr/bin/env bash
set -uo pipefail
export TREE={tree} CPUSET=32-63 PORT_CONTAINER_USER=0:0 PORT_IMAGE={IMAGE}
export SAEA_DEPLOYMENT_PROFILE={spec['profile']}
export PORT_DOCKER_FLAGS='--pid host --network host --name saea-{name}-generation -e SAEA_STAGE_TIMING_PATH={item}/stages.jsonl'
python3 {U}/scripts/npu/real_engine_metrics.py {ports_args} --output {item}/metrics-before.json || exit "$?"
bash {D}/work/scripts/port_py.sh scripts/analysis/run_real_engine_candidate.py --repo {study} --data {D} --candidate {name} --out {item}/run > {item}/run.log 2>&1
code=$?
python3 {U}/scripts/npu/real_engine_metrics.py {ports_args} --output {item}/metrics-after.json
printf '%s\\n' "$code" > {item}/run.exit
exit "$code"
''')
        with (item/'launch.log').open('x') as log:proc=subprocess.Popen(['bash',str(script)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        write(item/'launch.json',{'pid':proc.pid,'time':time.time(),'candidate':name})
        previous=name;terminal=item/'run.exit';previous_started=item/'run/started.json'
        if 'tp4' in name:
            # Preserve this candidate's source tree while its CPU finalizer runs.
            next_tree=tree.with_name(tree.name+'-tp8');next_study=next_tree/'studies/speech-aware-evidence-acquisition'
            if next_tree.exists():raise ValueError('next execution tree already exists')
            next_study.parent.mkdir(parents=True)
            call(['git','-c','safe.directory='+str(study),'clone','--shared',str(study),str(next_study)],item/'clone-next.log')
            for key in ('user.name','user.email'):git(next_study,'config',key,git(study,'config',key))
            (next_tree/'docs').mkdir();shutil.copyfile(tree/'docs/datasets.lock.json',next_tree/'docs/datasets.lock.json')
            tree=next_tree;study=next_study
            if git(study,'status','--porcelain'):raise ValueError('cloned execution tree dirty')
    status(root,'tp8_256_dispatched_384_held',candidate=previous)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tree',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if not a.out.resolve().is_relative_to(D/'work/codex'):raise ValueError('owned output required')
    a.out.mkdir(exist_ok=False)
    try:run(a.tree,a.out)
    except BaseException as e:status(a.out,'failed',error=str(e),automatic_retry=False);raise
