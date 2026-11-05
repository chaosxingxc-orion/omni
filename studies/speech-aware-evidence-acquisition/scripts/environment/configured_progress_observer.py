#!/usr/bin/env python3
"""Read configured lane progress every 600 seconds; never run or replay a job."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis'),str(Path(__file__).parent)]
import configured_terminal_settlement as settlement
import real_engine_progress
from speech_aware_evidence_acquisition.contracts import ExecutionPlan
from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal, canonical, comparable_request, digest
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha
queue=settlement.queue
require=queue.require
SCHEMA='saea-configured-progress-observer-v1'
# Legacy field describes monitoring side effects, never experiment throughput.
OBSERVER_CALL_ACCOUNTING={'new_model_calls':0,'observer_model_calls':0,
    'new_model_calls_scope':'observer invocation only; workload calls are reported separately'}


def load(path):
    d=settlement.read(path)
    require(set(d)=={'schema','state_dir','interval_seconds','lanes'} and d['schema']==SCHEMA
        and d['interval_seconds']==600 and isinstance(d['lanes'],list) and d['lanes'], 'exact progress config required')
    ids=set()
    for lane in d['lanes']:
        require(set(lane)=={'id','cards','instances','jobs'} and lane['id'] not in ids
            and type(lane['cards']) is int and lane['cards']>0
            and type(lane['instances']) is int and 0<lane['instances']<=lane['cards'], 'lane identity/resources differ')
        ids.add(lane['id'])
        for j in lane['jobs']:
            require(set(j)=={'id','kind','launch','launcher','source_plan','logical_data','arm','total_rows','roster','learner_job'}
                and j['kind'] in ('row-checkpoint','learning-updates') and type(j['total_rows']) is int
                and j['total_rows']>0, 'exact progress job config required')
            for key in ('launch','launcher','source_plan'):queue.pin_path(j[key])
            require((j['kind']=='row-checkpoint')==(j['roster'] is not None),'row progress roster required')
            if j['roster'] is not None:queue.pin_path(j['roster'])
            require((j['kind']=='learning-updates')==(j['learner_job'] is not None),'learning manifest required')
            if j['learner_job'] is not None:queue.pin_path(j['learner_job'])
    out=queue.launcher.absolute(d['state_dir'])
    for lane in d['lanes']:
        for j in lane['jobs']:
            producer=queue.launcher.load_config(queue.pin_path(j['launch']))
            for mount in producer['mounts']:
                source=Path(mount['source']).resolve()
                require(not out.is_relative_to(source) and not source.is_relative_to(out),'observer output overlaps source mount')
    return d,out


def timestamp(value):
    return datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()


def source(j):
    producer=queue.launcher.load_config(queue.pin_path(j['launch']))
    queue.pin_path(j['launcher'])
    observed=queue.launcher.status(producer,queue.canonical_hash(producer),j['launcher']['sha256'])
    if observed['state']!='not-launched':
        root=Path(producer['receipt_dir']);intent=settlement.read(root/'launch-intent.json')
        result=settlement.read(root/'launch-result.json')
        require(intent=={'schema':queue.launcher.SCHEMA,'version':queue.launcher.VERSION,
            'config_sha256':queue.canonical_hash(producer),'runner_sha256':j['launcher']['sha256'],
            'expected_commit':producer['expected_commit'],'automatic_retry':False},'observer launch intent differs')
        require(set(result)=={'returncode','stdout','stderr','automatic_retry'} and type(result['returncode']) is int
            and result['returncode']==0 and result['automatic_retry'] is False
            and re.fullmatch('[a-f0-9]{64}',result['stdout'].strip())
            and result['stdout'].strip()==observed.get('container_id'),'observer exact launch result differs')
    plan_doc=settlement.read(queue.pin_path(j['source_plan']));values=dict(plan_doc['execution_plan'])
    if 'owner_allowance' in plan_doc:values['owner_allowance']=plan_doc['owner_allowance']
    plan=ExecutionPlan(**values).validate(source_root=Path(producer['repo_path']))
    return producer,observed,plan


def native_contacts(producer,logical,plan,arm):
    journal=ContactJournal(logical,plan,arm,source_root=producer.get('repo_path'))
    journal.root=settlement.mapped(producer,logical/'contact-journals'/plan.run_id/plan.attempt_id)
    files=[];transient=[];vanished=[]
    for path in sorted(journal.root.glob('*')):
        try:
            stat=path.lstat()
        except FileNotFoundError:
            vanished.append(path.name);continue
        require(not path.is_symlink() and path.is_file(),'unexpected contact journal file')
        if re.fullmatch(r'\.[a-f0-9]{64}\.(intent|response|fit)\.json\.[A-Za-z0-9_-]+',path.name):
            transient.append(path.name);continue
        require(re.fullmatch(r'[a-f0-9]{64}\.(intent|response|fit)\.json',path.name),'unexpected contact journal file')
        files.append(path)
    fit_files=[p for p in files if p.name.endswith('.fit.json')]
    for path in fit_files:
        fit=settlement.read(path)
        require(fit.get('schema')=='saea-native-fit-custody-v1' and fit.get('identity')==journal.identity
            and path.name==digest(canonical(comparable_request(fit['payload'])))+'.fit.json', 'native FIT custody differs')
    intents=[p for p in files if p.name.endswith('.intent.json')]
    response_names={p.name for p in files if p.name.endswith('.response.json')}
    require(response_names<={p.name.replace('.intent.','.response.') for p in intents},'orphan native response')
    known=defaultdict(float,{'calls':0,'prompt_tokens':0,'completion_tokens':0,'latency_seconds':0})
    unknown=[]; records={}
    for ip in intents:
        rid=settlement.read(ip)['request_id'];require(ip==journal._path(rid,'intent'),'native filename differs')
        record=journal.inspect(rid)
        if record['state']=='unknown':unknown.append(rid);continue
        require(record['state']=='observed','native state differs')
        response=json.loads(record['response_bytes']);usage=response.get('usage')
        require(isinstance(usage,dict),'observed response usage absent')
        for key in ('prompt_tokens','completion_tokens'):
            value=usage.get(key);require(type(value) is int and value>=0,'invalid native metered tokens');known[key]+=value
        known['calls']+=1;known['latency_seconds']+=record['response']['latency_seconds']
        # Payload audio_seconds describes the source sample even for text-only
        # learning. It is not transmitted/billed audio and must not be charged.
        records[rid]=record
    return {'attempted_contacts':len(intents),'observed_contacts':len(records),
        'inflight_or_unknown_contacts':len(unknown),'known_usage':dict(known),
        'unknown_cost':None,'unknown_cost_reason':'Unreturned requests have unknown spend, never assumed zero.',
        'fit_custody_records':len(fit_files),'fit_cost':'accounted separately; not included in native generation contacts',
        'uncommitted_publish_files':transient,'vanished_during_enumeration':vanished,
        'snapshot_boundary':'native intents present at directory enumeration; responses may finish during this read',
        **OBSERVER_CALL_ACCOUNTING,
        'workload_model_calls':{'attempted':len(intents),'observed':len(records),
            'inflight_or_unknown':len(unknown),'run_id':plan.run_id,'attempt_id':plan.attempt_id,
            'scope':'cumulative current attempt native journal; excludes FIT and prior attempts; not a polling-interval delta'}},records


def learning_progress(j,producer,logical,plan,native):
    job=settlement.read(queue.pin_path(j['learner_job']))
    require(job['run_id']==plan.run_id and job['runtime_rows']==j['total_rows'],'learning manifest differs')
    base=logical/'runs'/plan.run_id/plan.attempt_id/'tfgrpo/learner'
    root=settlement.mapped(producer,base)
    if not (root/'batch.json').exists():return {'completed_rows':0,'total_rows':j['total_rows'],'completion_basis':'no durable batch yet'}
    batch=settlement.read(root/'batch.json');require(sha(batch)==job['batch_sha256'],'learning batch differs')
    groups=batch['groups'];require(len(groups)==j['total_rows'] and len({g['query_id'] for g in groups})==len(groups),
        'learning groups not distinct roster')
    updates=sorted((root/'updates').glob('*.json'))
    require([p.name for p in updates]==[f'{i:04d}.json' for i in range(len(updates))], 'update prefix differs')
    require(len(updates)<=len(groups),'more updates than groups')
    strategy=batch['strategy'];items=batch['parent']['items'];memory=None
    if strategy=='reflexion_offline_transfer_v1':
        from speech_aware_evidence_acquisition.rails import reflexion_offline as algorithm
        memory=algorithm.empty_memory();stages=[algorithm.REFLECT]
    elif strategy=='ace_incremental_offline_v1':
        from speech_aware_evidence_acquisition.rails import ace_incremental as algorithm
        stages=[algorithm.REFLECT,algorithm.CURATE]
    else:raise ValueError('unsupported learning updates strategy')
    for index,path in enumerate(updates):
        group=groups[index];event=settlement.read(path);require(event['group_index']==index and event['query_id']==group['query_id'],
            'update group identity differs')
        responses=[]
        for stage in stages:
            identity=f"{batch['input_sha256']}:group-{index}";key=sha({'stage':stage,'identity':identity})
            intent=settlement.read(root/'contacts'/(key+'.intent.json'))
            envelope=settlement.read(root/'contacts'/(key+'.response.json'));r=envelope['response']
            require(intent['stage']==stage and intent['identity']==identity and intent['batch_sha256']==sha(batch)
                and envelope['request_sha256']==sha(intent) and envelope['response_sha256']==sha(r),'learning envelope differs')
            prefix=f'{plan.run_id}/{plan.attempt_id}/'
            require(r['receipt_id'].startswith(prefix),'learning receipt run/attempt differs')
            rid=r['receipt_id'][len(prefix):];require(rid in native,'update contact not durably observed')
            n=native[rid];normalized=json.loads(n['response_bytes'])
            require(r['request_sha256']==n['intent']['request_sha256'] and r['response_sha256']==n['response']['response_sha256']
                and r['text']==normalized['text'],'learning/native bridge differs')
            responses.append(r['text'])
        bounds=batch['bounds']
        if memory is None:
            reflection=algorithm.validate_reflection(responses[0],group,items,bounds['max_findings'],bounds.get('reflector_contract'),bounds=bounds)
            merged=algorithm.merge_delta(responses[1],items,reflection,bounds)
            expected={'group_index':index,'query_id':group['query_id'],'reflection':reflection,'merge':merged,
                'before_items_sha256':sha(items),'after_items_sha256':sha(merged['items'])}
            require(event==expected,'ACE update replay differs');items=merged['items']
        else:
            reflection=algorithm.validate_reflection(responses[0],group,bounds['max_entry_characters'],bounds.get('reflector_contract'))
            memory,expected=algorithm.advance_memory(memory,reflection,group,index,bounds)
            require(event==expected,'Reflexion update replay differs')
    return {'completed_rows':len(updates),'total_rows':len(groups),
        'completion_basis':'durable sequential group updates, rederived from observed responses',
        'batch_sha256':sha(batch),'candidate_accepted':False}


def row_progress(j,producer,logical,plan,elapsed,state,scratch,lane):
    source=settlement.mapped(producer,logical/'runs'/plan.run_id/plan.attempt_id/(plan.run_id+'.rows.ckpt.jsonl'))
    if not source.exists():return {'completed_rows':0,'total_rows':j['total_rows'],'completion_basis':'no durable checkpoint yet'}
    # Bound the copy to its original byte length; append-only writers never block.
    with source.open('rb') as src,tempfile.NamedTemporaryFile(dir=scratch,suffix='.jsonl') as target:
        remaining=os.fstat(src.fileno()).st_size
        while remaining:
            block=src.read(min(remaining,4*1024*1024));require(block,'checkpoint shrank');target.write(block);remaining-=len(block)
        target.flush()
        result=real_engine_progress.report(Path(target.name),queue.pin_path(j['roster']),state=state,
            elapsed_seconds=elapsed,cards=lane['cards'],instances=lane['instances'])
    require(result['total_rows']==j['total_rows'],'checkpoint total differs');return result


def recovery_context(producer,logical,plan):
    """Identify recovery without treating replayed rows as new inference work."""
    path=settlement.mapped(producer,logical/'runs'/plan.run_id/plan.attempt_id/'resume-declaration.json')
    if not path.exists():return None
    declaration=settlement.read(path)
    require(declaration.get('schema')=='saea-resume-declaration-v1'
        and declaration.get('run_id')==plan.run_id and declaration.get('attempt_id')==plan.attempt_id,
        'observer recovery declaration identity differs')
    pin=declaration.get('native_recovery_receipt')
    require(isinstance(pin,dict) and set(pin)=={'path','sha256'},'observer recovery receipt pin required')
    receipt_path=settlement.mapped(producer,Path(pin['path']))
    require(queue.launcher.digest(receipt_path)==pin['sha256'],'observer recovery receipt hash differs')
    receipt=settlement.read(receipt_path);target=receipt.get('target_plan',{})
    require(receipt.get('schema')=='saea-native-recovery-v1'
        and all(target.get(k)==getattr(plan,k) for k in ('run_id','attempt_id','protocol_hash','split_identity_hash')),
        'observer recovery target differs')
    return {'source_attempt_id':receipt['source']['plan']['attempt_id'],
        'receipt_sha256':pin['sha256'],
        'checkpoint_basis':'reconstructed historical rows and newly completed rows; not additive across attempts',
        'cost_basis':'current native journal counts new contacts; checkpoint costs may include historical replay',
        'fresh_inference_rate_verified':False}


def observe_job(j,lane,scratch,now):
    producer,observed,plan=source(j);logical=Path(j['logical_data'])
    result={'id':j['id'],'kind':j['kind'],'observed_at_utc':datetime.fromtimestamp(now,timezone.utc).isoformat(),
        'container':observed,'allocated_cards':lane['cards'],'allocated_instances':lane['instances'],
        'completed_rows':0,'total_rows':j['total_rows'],**OBSERVER_CALL_ACCOUNTING,
        'workload_model_calls':None,'measured_npu_utilization_percent':None}
    if observed['state']=='not-launched':
        return {**result,'state':'not-started','wall_seconds_per_complete_row':None,'estimated_remaining_seconds':None}
    state=queue.outcome(observed);require(state in ('running','complete','failed'),'unknown producer state')
    end=now if state=='running' else timestamp(observed['finished_at'])
    elapsed=max(end-timestamp(observed['started_at']),.001)
    contact,records=native_contacts(producer,logical,plan,j['arm'])
    # EXIT0 is a process fact; complete rows and final scientific seal remain independent.
    status='running' if state=='running' else 'failed' if state=='failed' else 'stopped'
    progress=(row_progress(j,producer,logical,plan,elapsed,status,scratch,lane) if j['kind']=='row-checkpoint'
        else learning_progress(j,producer,logical,plan,records))
    n=progress['completed_rows'];total=j['total_rows'];sp=elapsed/n if n else None
    recovery=recovery_context(producer,logical,plan) if j['kind']=='row-checkpoint' else None
    result={**result,**progress,'state':status,'process_state':state,'elapsed_seconds':elapsed,'cards':lane['cards'],'instances':lane['instances'],
        'remaining_rows':total-n,'wall_seconds_per_complete_row':sp,
        'estimated_remaining_seconds':(total-n)*sp if state=='running' and n else (0 if n==total else None),
        'eta_basis':'linear cumulative valid completion rate; startup, serial learning and tail can change it',
        'contacts':contact,'workload_model_calls':contact['workload_model_calls'],'failed_attempt_known_usage':contact['known_usage'] if state=='failed' else None,
        'sample_count_is_not_contact_count':True,'scientific_gain_claimed':False}
    if recovery is not None:
        result.update(recovery=recovery,wall_seconds_per_complete_row=None,rows_per_second=None,
            estimated_remaining_seconds=0 if n==total else None,
            eta_basis='recovery mixes historical replay and new work; fresh inference ETA not established',
            reconstruction_wall_seconds_per_row=sp)
    return result


def observe(path):
    d,out=load(path);out.mkdir(parents=True,exist_ok=True);now=time.time();lanes=[]
    for lane in d['lanes']:
        jobs=[]
        for j in lane['jobs']:
            try:jobs.append(observe_job(j,lane,out,now))
            except Exception as error:jobs.append({'id':j['id'],'state':'observation-error','error_type':type(error).__name__,
                'error':str(error),'completed_rows':None,'total_rows':j['total_rows'],'estimated_remaining_seconds':None})
        lanes.append({'id':lane['id'],'allocated_cards':lane['cards'],'allocated_instances':lane['instances'],
            'jobs':jobs,'resource_note':'Jobs share this lane; do not sum their cards or unique dataset rows.'})
    return {'schema':SCHEMA,'observed_at_utc':datetime.fromtimestamp(now,timezone.utc).isoformat(),
        'configuration_sha256':queue.launcher.digest(path),'interval_seconds':600,'lanes':lanes,**OBSERVER_CALL_ACCOUNTING}


def execute(path,mode):
    d,out=load(path);out.mkdir(parents=True,exist_ok=True)
    with (out/'observer.guard').open('a+b') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            value=observe(path);queue.atomic(out/'status.json',value)
            queue.atomic(out/'last-observation.json',{'time':time.time(),'pid':os.getpid(),
                'configuration_sha256':value['configuration_sha256'],**OBSERVER_CALL_ACCOUNTING})
            if mode=='once':return value
            time.sleep(600)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('once','watch'))
    p.add_argument('--config',type=Path,required=True);a=p.parse_args();print(json.dumps(execute(a.config,a.mode),sort_keys=True))
