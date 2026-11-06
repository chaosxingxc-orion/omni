#!/usr/bin/env python3
"""Wait for a configured lane queue to finish, then prove safe resource handoff."""
# Host tools import pinned release modules; keep its signed file set immutable.
import sys
sys.dont_write_bytecode = True

import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import configured_terminal_settlement as settlement
import configured_engine_transition as transition
queue=settlement.queue
require=queue.require
SCHEMA='saea-configured-queue-handoff-v1'


def load(path):
    d=settlement.read(path)
    require(set(d)=={'schema','queue_config','queue_tool','watch_launch','watch_launcher','settlements','state_dir',
        'monitor_interval_seconds'} and d['schema']==SCHEMA and d['monitor_interval_seconds']==600,
        'exact configured queue handoff required')
    script=queue.pin_path(d['queue_tool'])
    require(script.name=='configured_job_queue.py','original queue implementation required')
    spec=importlib.util.spec_from_file_location('original_handoff_queue',script)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    config=queue.pin_path(d['queue_config']); source=original.load(config)
    watch=queue.launcher.load_config(queue.pin_path(d['watch_launch']))
    require(watch['argv'][1:]==[str(script),'watch','--config',str(config)],'watcher exact command differs')
    require(queue.pin_path(d['watch_launcher']).name=='configured_job_launch.py','watcher launcher required')
    generation={j['id']:j for j in source['jobs'] if 'generation_release' in j}
    require(generation and set(d['settlements'])==set(generation),'every producer settlement source required')
    bindings={}
    for name,pin in d['settlements'].items():
        sp=queue.pin_path(pin);sd,producer,release,logical,expected,out=settlement.load(sp)
        require(sd['producer_launch']==generation[name]['launch']
            and sd['generation_release']==generation[name]['generation_release'], 'handoff source mismatch')
        bindings[name]={'doc':{'source_settlement':pin},'source':sd,'producer':producer,'release':release,
            'logical':logical,'expected':expected,'locks':settlement.mapped(producer,logical/'locks',writable=True)}
    require(len({str(v['locks']) for v in bindings.values()})==1,'single shared lane guard required')
    out=queue.launcher.absolute(d['state_dir']); locks=next(iter(bindings.values()))['locks']
    require(not out.is_relative_to(locks) and not locks.is_relative_to(out),'handoff output overlaps shared guard')
    return d,original,source,watch,bindings,out


def inspect(path):
    d,original,source,watch,bindings,out=load(path)
    state_path=Path(source['state_dir'])/'state.json'
    require(state_path.is_file(),'source queue state absent')
    state=settlement.read(state_path)
    require(state.get('identity')==original.identities(queue.pin_path(d['queue_config']))
        and state.get('queue_id')==source['queue_id']
        and [r['id'] for r in state['jobs']]==[j['id'] for j in source['jobs']], 'source queue custody changed')
    require(state['status'] in ('ready','running','waiting','dispatching','complete','failed'),'unknown queue blocks handoff')
    if state['status'] in ('ready','running','waiting','dispatching'):
        return {'ready':False,'reason':'queue-running','state':state['status']}
    # Old queue CLIs returned 0 even for failed state. Verify state and every source, not exit==0.
    terminal=settlement.producer_terminal({'producer_launcher':d['watch_launcher']},watch)
    records={r['id']:r for r in state['jobs']}
    require(all(r['status'] in ('complete','failed','skipped') for r in records.values()),
        'pending blocked unknown queue jobs prevent handoff')
    locks=next(iter(bindings.values()))['locks']
    with settlement.guard(locks/'generation.guard') as (check,guard):
        require(not (locks/'flight.lock').exists() and not (locks/'engine-transition.pending').exists(),
                'active lease or engine transition blocks handoff')
        proofs={}
        for name,c in bindings.items():
            producer_terminal=settlement.producer_terminal(c['source'],c['producer'])
            require(records[name].get('container_id')==producer_terminal['container_id']
                and records[name]['status'] in ('complete','failed'), 'producer exact terminal state differs')
            custody=settlement.source_custody(c['source'],c['producer'],c['logical'],c['expected'],c['release'])
            proof=transition.source_barrier(c)
            require(proof['contacts']==custody['contacts'],'terminal native custody differs')
            proofs[name]={'terminal':producer_terminal,'source':custody,'barrier':proof}
        last=next(reversed(bindings.values()))
        instances=settlement.deployment(last['source'],last['producer'],last['logical'],last['expected'])
        quiet=settlement.transition.quiescence(instances)
        require(settlement.read(state_path)==state,'source queue changed while reconciling')
        for name,c in bindings.items():
            require(transition.source_barrier(c)==proofs[name]['barrier'],'source barrier changed during quiescence')
        require(not (locks/'flight.lock').exists(),'new lane lease blocks handoff');check()
        return {'ready':True,'schema':SCHEMA,'queue_state':state['status'],'state_sha256':queue.launcher.digest(state_path),
            'watcher_terminal':terminal,'guard':guard,'proofs':proofs,'quiescence':quiet,
            'new_model_calls':0,'scientific_success':False,'successor_requires_own_lease_and_readiness':True}


def execute(path,mode):
    d,original,source,watch,bindings,out=load(path)
    if mode=='plan':return {'schema':SCHEMA,'producers':list(bindings),'new_model_calls':0}
    require(mode in ('once','watch'),'handoff mode required')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'handoff.guard').open('a+b') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity={'config_sha256':queue.launcher.digest(path),'tool_sha256':queue.launcher.digest(__file__)}
        complete=out/'complete.json'
        if complete.exists():
            result=settlement.read(complete);require(result['identity']==identity,'handoff completed identity differs');return result
        helper=original.release_helper(source)
        events=helper.DirectoryEvents(Path(source['state_dir']))
        try:
            while True:
                result={**inspect(path),'identity':identity,'observed_at_unix':time.time()}
                queue.atomic(out/'status.json',result)
                if result['ready']:
                    queue.atomic(complete,result);return result
                if mode=='once':return result
                events.wait(600)
        finally:events.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('plan','once','watch'))
    p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    r=execute(a.config,a.mode);print(json.dumps(r,sort_keys=True));sys.exit(0 if a.mode=='plan' or r.get('ready') else 1)
