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
            d=read(out/'complete.json');require(d['config_sha256']==cfg['config_sha256'] and not lease.exists() and
                sha(out/'manifest.json')==d['manifest_sha256'] and sha(out/'lease.json')==cfg['lease_sha256'], 'completed retirement changed')
            return d
        require(not out.exists(),'fresh retirement directory required')
        raw=lease.read_bytes();l=json.loads(raw)
        require(digest(raw)==cfg['lease_sha256'] and all(l.get(k)==v for k,v in {'run_id':run,'attempt_id':attempt_id,
            'pid':cfg['pid'],'session_id':cfg['session_id'],'study_head':cfg['study_commit'],'run_ids':[run]}.items()),'lease identity differs')
        attempt_path=data/'attempts'/run/(attempt_id+'.json');before=attempt_path.read_bytes();state=json.loads(before)
        require(digest(before)==cfg['attempt_sha256'] and state['state']=='OPENED' and state['actual']==cfg['actual'], 'attempt or cost changed')
        require(sha(cfg['runtime_receipt'])==cfg['runtime_receipt_sha256'],'runtime receipt differs')
        session=read(data/'serving'/cfg['session_id']/'session.json');runtime=read(cfg['runtime_receipt'])
        instances=runtime['deployment']['instances']
        require(session['session_id']==cfg['session_id'] and session['identity']['runtime_receipt_sha256']==cfg['runtime_receipt_sha256'] and
                session['deployment_instance_ids']==[i['instance_id'] for i in instances], 'serving session differs')
        journal=ContactJournal(data,plan,cfg['arm']);snapshot=inventory(journal)
        require(snapshot['calls']==state['actual']['calls'] and math.isclose(snapshot['audio_seconds'],state['actual']['audio_seconds'],rel_tol=1e-12,abs_tol=1e-9), 'cost and journal differ')
        require(snapshot['unknown_count']==cfg['unknown_count'] and snapshot['unknown_count']>0, 'unknown inventory differs')
        require(digest(canonical(snapshot))==cfg['inventory_sha256'],'journal inventory changed')
        fit_root=Path(cfg['fit_receipts_root'])
        fit_files={}
        for p in journal.root.glob('*.fit.json'):
            receipt=read(p)['receipt'];target=fit_root/Path(receipt['path']).name
            require(target.is_file() and not target.is_symlink() and sha(target)==receipt['sha256'], 'FIT receipt differs')
            fit_files[target.name]=sha(target)
        require(len(fit_files)==cfg['fit_count'], 'FIT inventory differs')
        checkpoint=Path(cfg['checkpoint']);require(sha(checkpoint)==cfg['checkpoint_sha256'], 'checkpoint differs')
        quiet=quiescence(instances)
        require(inventory(journal)==snapshot and lease.read_bytes()==raw and attempt_path.read_bytes()==before, 'custody changed after quiescence')
        guard_check()
        out.mkdir(parents=True);sync(out.parent)
        for name in snapshot['files']:
            publish_new(out/name,(journal.root/name).read_bytes())
        publish_new(out/'lease.json',raw);publish_new(out/'attempt-before.json',before)
        publish_new(out/'plan.json',plan_path.read_bytes())
        publish_new(out/'checkpoint.jsonl',checkpoint.read_bytes())
        for name,expected in fit_files.items():
            content=(fit_root/name).read_bytes();require(digest(content)==expected,'FIT changed during archive')
            publish_new(out/'fit'/name,content)
        manifest={'schema':SCHEMA,'status':'retirement-intent','config_sha256':cfg['config_sha256'],
                  'run_id':run,'attempt_id':attempt_id,'successor_run_id':cfg['new_baseline_run_id'],
                  'unknown_responses_remain_unknown':True,'automatic_retry':False,'snapshot':snapshot,
                  'cost':state['actual'],'fit_files':fit_files,'checkpoint_sha256':cfg['checkpoint_sha256'],
                  'producer_container':cfg['container'],'boot_id':boot,
                  'quiescence':quiet,'lease_sha256':digest(raw),'created_at':time.time()}
        publish_new(out/'manifest.json',canonical(manifest))
        marker={'schema':SCHEMA,'run_id':run,'attempt_id':attempt_id,'automatic_retry':False,
                'unknown_count':snapshot['unknown_count'],'manifest_sha256':sha(out/'manifest.json')}
        publish_new(journal.root/'retirement.json',canonical(marker))
        require(inventory(journal)==snapshot and lease.read_bytes()==raw and attempt_path.read_bytes()==before,'custody changed during archive')
        require(AttemptStore(data/'attempts').abort_if_opened(run,attempt_id,reason='user interrupted; unknown requests retired, no automatic retry'),'attempt abort failed')
        after=read(attempt_path);require(after['state']=='ABORTED' and after['actual']==state['actual'],'abort changed cost')
        publish_new(out/'attempt-aborted.json',attempt_path.read_bytes())
        require(inventory(journal)==snapshot and lease.read_bytes()==raw,'custody changed before release')
        quiescence(instances);guard_check();require(not Path('/proc',str(cfg['pid'])).exists(),'producer PID reappeared')
        os.unlink(lease);sync(lease.parent)
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
