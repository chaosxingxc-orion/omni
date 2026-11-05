#!/usr/bin/env python3
"""Explicitly retire a stopped native attempt with unknown responses.

This is a separate, opt-in custody transition. It never reconciles unknown
responses as observed and never authorizes reuse of the retired run.
"""
import argparse
import base64
import fcntl
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/environment')]
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.core.attempts import AttemptStore
from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal, canonical, digest
from speech_aware_evidence_acquisition.contracts import ExecutionPlan
from transition_serving_budget import quiescence
from speech_aware_evidence_acquisition.core.session.receipt import SessionReceipt
from speech_aware_evidence_acquisition.core.session.vllm_identity import verify_vllm_session, verify_vllm_process_identity

SCHEMA='saea-unknown-attempt-retirement-v1'

def require(condition, message):
    if not condition: raise ValueError(message)

def read(path): return json.loads(Path(path).read_bytes())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sync(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
def docker_inspect(container): return json.loads(subprocess.check_output(['docker','inspect',container]))[0]

def inventory(journal):
    files=sorted(journal.root.iterdir())
    require(all(p.is_file() and not p.is_symlink() for p in files),'unexpected journal entry')
    records={}; names=set()
    for p in files:
        if p.name=='retirement.json': continue
        require(p.name.endswith(('.intent.json','.response.json','.fit.json')), 'unknown journal file')
        names.add(p.name);records[p.name]=sha(p)
    intents=sorted(p for p in files if p.name.endswith('.intent.json'))
    rows=[];audio=[];expected_responses=set()
    for p in intents:
        doc=read(p);rid=doc['request_id'];require(journal._path(rid,'intent')==p,'intent filename differs')
        x=journal.inspect(rid);require(x['state'] in ('observed','unknown'),'contact state differs')
        audio.append(float(x['intent']['payload']['audio_seconds']))
        response=journal._path(rid,'response')
        if x['state']=='observed':expected_responses.add(response.name)
        rows.append({'request_id':rid,'state':x['state'],'intent_sha256':records[p.name],
                     'request_sha256':x['intent']['request_sha256'],
                     'response_sha256':records.get(response.name)})
    require({n for n in names if n.endswith('.response.json')}==expected_responses,'orphan response')
    return {'files':records,'contacts':rows,'calls':len(rows),'audio_seconds':math.fsum(audio),
            'unknown_count':sum(r['state']=='unknown' for r in rows)}

def verify_service(data, cfg, instances, runtime):
    coordinator = read(data/'serving'/cfg['session_id']/'session.json')
    for instance in instances:
        name=instance['instance_id']
        document=read(data/'serving'/name/'session.json')
        receipt=SessionReceipt.from_json(read(data/'serving'/name/'session-receipt.json'))
        verify_vllm_session(receipt, runtime)
        require(receipt.pid == document['pid'] and receipt.started_at == document['started_at'], 'live service identity differs')
        require(list(verify_vllm_process_identity(receipt.pid,runtime,instance,host=receipt.host,port=receipt.port))==list(receipt.argv), 'live service argv differs')
    require(coordinator['deployment_instance_ids']==[i['instance_id'] for i in instances], 'coordinator membership differs')


def execute(cfg):
    require(cfg.get('schema')==SCHEMA and cfg.get('mode')=='retire-unknown-no-retry', 'explicit retirement mode required')
    data=Path(cfg['data']);out=Path(cfg['out']);plan_path=Path(cfg['plan']);run=cfg['run_id'];attempt_id=cfg['attempt_id']
    require(out.is_absolute() and out.resolve().is_relative_to((data/'work/codex').resolve()),'owned output required')
    require(cfg.get('new_baseline_run_id') and cfg['new_baseline_run_id']!=run,'distinct successor run required')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    require(boot==cfg['boot_id'],'boot identity differs')
    require(not Path('/proc',str(cfg['pid'])).exists(),'producer PID alive or reused')
    container=docker_inspect(cfg['container'])
    require(container['Id']==cfg['container'] and container['State']['Status']=='exited' and
            not container['State']['Running'] and container['State']['Pid']==0 and
            container['State']['ExitCode']==cfg['exit_code'] and cfg['exit_code']!=0,'producer not exact failed terminal')
    require(container['Config']['Labels'].get('saea.launch.config')==cfg['launch_config_identity'], 'launch identity differs')
    start=datetime.fromisoformat(container['State']['StartedAt'].replace('Z','+00:00')).timestamp()
    btime=int(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime ')))
    require(start>=btime, 'container predates current boot')
    require(sha(plan_path)==cfg['plan_sha256'], 'registered plan changed')
    pd=read(plan_path);kwargs={k:v for k,v in pd['execution_plan'].items() if k not in ('schema','study')}
    if 'owner_allowance' in pd:kwargs['owner_allowance']=pd['owner_allowance']
    plan=ExecutionPlan(**kwargs).validate(source_root=ROOT)
    require((plan.run_id,plan.attempt_id)==(run,attempt_id),'plan identity differs')
    require(pd.get('expected_session_id')==cfg['session_id'],'plan session differs')
    require(sha(cfg['started'])==cfg['started_sha256'],'started receipt changed')
    started=read(cfg['started']);require(started['pid']==cfg['pid'] and started['study_commit']==cfg['study_commit'], 'startup identity differs')
    lease=data/'locks/flight.lock';guard=data/'locks/generation.guard'
    require(guard.is_file() and not guard.is_symlink(),'existing guard required')
    with guard.open('r+b') as g:
        fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB);gstat=os.fstat(g.fileno())
        def guard_check():
            stat=guard.stat();require((stat.st_dev,stat.st_ino)==(gstat.st_dev,gstat.st_ino) and not guard.is_symlink(),'guard changed')
        guard_check()
        if (out/'complete.json').exists():
            d=read(out/'complete.json')
            require(d['config_sha256']==cfg['config_sha256'] and
                sha(out/'manifest.json')==d['manifest_sha256'] and sha(out/'lease.json')==cfg['lease_sha256'], 'completed retirement changed')
            # A new tenant may now hold the lane; never inspect or remove its lease.
            return d
        lease_blob=None
        if lease.exists():
            lease_blob=lease.read_bytes();l=json.loads(lease_blob)
            require(digest(lease_blob)==cfg['lease_sha256'] and all(l.get(k)==v for k,v in {'run_id':run,'attempt_id':attempt_id,
                'pid':cfg['pid'],'session_id':cfg['session_id'],'study_head':cfg['study_commit'],'run_ids':[run]}.items()),'lease identity differs')
        attempt_path=data/'attempts'/run/(attempt_id+'.json')
        current=attempt_path.read_bytes();state=json.loads(current)
        require(state['actual']==cfg['actual'] and state['state'] in ('OPENED','ABORTED'), 'attempt or cost changed')
        if state['state']=='OPENED':require(digest(current)==cfg['attempt_sha256'],'attempt changed')
        require(sha(cfg['runtime_receipt'])==cfg['runtime_receipt_sha256'],'runtime receipt differs')
        session=read(data/'serving'/cfg['session_id']/'session.json');runtime=read(cfg['runtime_receipt'])
        instances=runtime['deployment']['instances']
        require(session['session_id']==cfg['session_id'] and session['identity']['runtime_receipt_sha256']==cfg['runtime_receipt_sha256'] and
                session['deployment_instance_ids']==[i['instance_id'] for i in instances], 'serving session differs')
        journal=ContactJournal(data,plan,cfg['arm']);snapshot=inventory(journal)
        require(snapshot['calls']==state['actual']['calls'] and math.isclose(snapshot['audio_seconds'],state['actual']['audio_seconds'],rel_tol=1e-12,abs_tol=1e-9), 'cost and journal differ')
        require(snapshot['unknown_count']==cfg['unknown_count'] and snapshot['unknown_count']>0, 'unknown inventory differs')
        require(digest(canonical(snapshot))==cfg['inventory_sha256'],'journal inventory changed')
        fit_root=Path(cfg['fit_receipts_root']);fit_files={}
        for p in journal.root.glob('*.fit.json'):
            receipt=read(p)['receipt'];target=fit_root/Path(receipt['path']).name
            require(target.is_file() and not target.is_symlink() and sha(target)==receipt['sha256'], 'FIT receipt differs')
            fit_files[target.name]=sha(target)
        require(len(fit_files)==cfg['fit_count'], 'FIT inventory differs')
        checkpoint=Path(cfg['checkpoint']);require(sha(checkpoint)==cfg['checkpoint_sha256'], 'checkpoint differs')
        verify_service(data,cfg,instances,runtime);quiet=quiescence(instances);guard_check()
        require(inventory(journal)==snapshot and attempt_path.read_bytes()==current, 'custody changed after quiescence')
        marker_path=journal.root/'retirement.json'
        if (out/'manifest.json').exists():
            manifest=read(out/'manifest.json')
            require(manifest['config_sha256']==cfg['config_sha256'] and manifest['snapshot']==snapshot and
                    manifest['fit_files']==fit_files and manifest['cost']==state['actual'], 'archive manifest differs')
            require(sha(out/'lease.json')==cfg['lease_sha256'] and sha(out/'attempt-before.json')==cfg['attempt_sha256'] and
                    sha(out/'plan.json')==cfg['plan_sha256'] and sha(out/'checkpoint.jsonl')==cfg['checkpoint_sha256'], 'archive custody differs')
            for name,expected in snapshot['files'].items():require(sha(out/name)==expected,'archive journal copy differs')
            for name,expected in fit_files.items():require(sha(out/'fit'/name)==expected,'archive FIT copy differs')
        else:
            require(lease_blob is not None and state['state']=='OPENED' and not marker_path.exists(), 'fresh source custody required')
            out.mkdir(parents=True,exist_ok=True);sync(out.parent)
            def stage(path,content):
                if path.exists():require(path.is_file() and not path.is_symlink() and path.read_bytes()==content,'partial archive differs')
                else:publish_new(path,content)
            for name in snapshot['files']:stage(out/name,(journal.root/name).read_bytes())
            stage(out/'lease.json',lease_blob);stage(out/'attempt-before.json',current)
            stage(out/'plan.json',plan_path.read_bytes());stage(out/'checkpoint.jsonl',checkpoint.read_bytes())
            for name,expected in fit_files.items():stage(out/'fit'/name,(fit_root/name).read_bytes())
            for name,expected in snapshot['files'].items():require(sha(out/name)==expected,'archive journal copy differs')
            for name,expected in fit_files.items():require(sha(out/'fit'/name)==expected,'archive FIT copy differs')
            require(sha(out/'lease.json')==cfg['lease_sha256'] and sha(out/'attempt-before.json')==cfg['attempt_sha256'] and
                    sha(out/'plan.json')==cfg['plan_sha256'] and sha(out/'checkpoint.jsonl')==cfg['checkpoint_sha256'], 'archive custody differs')
            manifest={'schema':SCHEMA,'status':'retirement-intent','config_sha256':cfg['config_sha256'],
                      'run_id':run,'attempt_id':attempt_id,'successor_run_id':cfg['new_baseline_run_id'],
                      'unknown_responses_remain_unknown':True,'automatic_retry':False,'snapshot':snapshot,
                      'cost':state['actual'],'fit_files':fit_files,'checkpoint_sha256':cfg['checkpoint_sha256'],
                      'producer_container':cfg['container'],'boot_id':boot,'quiescence':quiet,
                      'lease_sha256':cfg['lease_sha256'],'created_at':time.time()}
            publish_new(out/'manifest.json',canonical(manifest))
        marker={'schema':SCHEMA,'run_id':run,'attempt_id':attempt_id,'automatic_retry':False,
                'unknown_count':snapshot['unknown_count'],'manifest_sha256':sha(out/'manifest.json')}
        if marker_path.exists():require(read(marker_path)==marker,'retirement marker differs')
        else:
            require(lease_blob is not None and state['state']=='OPENED','marker missing after abort or release')
            publish_new(marker_path,canonical(marker))
        if state['state']=='OPENED':
            require(lease_blob is not None,'lease missing before abort')
            require(AttemptStore(data/'attempts').abort_if_opened(run,attempt_id,reason='user interrupted; unknown requests retired, no automatic retry'),'attempt abort failed')
        after=read(attempt_path);require(after['state']=='ABORTED' and after['actual']==state['actual'],'abort changed cost')
        if not (out/'attempt-aborted.json').exists():publish_new(out/'attempt-aborted.json',attempt_path.read_bytes())
        require((out/'attempt-aborted.json').read_bytes()==attempt_path.read_bytes(),'aborted attempt copy differs')
        require(inventory(journal)==snapshot,'custody changed before release')
        for name,expected in snapshot['files'].items():require(sha(out/name)==expected,'archive journal copy differs')
        for name,expected in fit_files.items():require(sha(out/'fit'/name)==expected,'archive FIT copy differs')
        require(sha(out/'lease.json')==cfg['lease_sha256'] and sha(out/'plan.json')==cfg['plan_sha256'] and
                sha(out/'checkpoint.jsonl')==cfg['checkpoint_sha256'], 'final archive custody differs')
        verify_service(data,cfg,instances,runtime);quiescence(instances);guard_check();require(not Path('/proc',str(cfg['pid'])).exists(),'producer PID reappeared')
        if lease_blob is not None:
            require(lease.read_bytes()==lease_blob,'lease changed before release');os.unlink(lease);sync(lease.parent)
        done={'schema':SCHEMA,'status':'RETIRED_UNKNOWN_RESOURCE_RELEASED','config_sha256':cfg['config_sha256'],
              'manifest_sha256':sha(out/'manifest.json'),'unknown_count':snapshot['unknown_count'],
              'observed_count':snapshot['calls']-snapshot['unknown_count'],'actual':state['actual'],
              'new_model_calls':0,'automatic_retry':False,'scientific_success':False}
        publish_new(out/'complete.json',canonical(done));return done

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    blob=args.config.read_bytes();cfg=json.loads(blob);cfg['config_sha256']=digest(blob)
    print(json.dumps(execute(cfg),sort_keys=True))
if __name__=='__main__':main()
