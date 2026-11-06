#!/usr/bin/env python3
"""Preserve and settle a verified pre-attempt candidate lease; never dispatch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

import configured_terminal_settlement as s

SCHEMA = 'saea-pre-attempt-settlement-v1'


def absent(producer, logical, expected):
    paths = []
    for run, attempt in ((expected['run_id'], expected['attempt_id']),
                         (expected['fit_run_id'], expected['fit_attempt_id'])):
        for rel in (Path('attempts')/run/(attempt+'.json'),
                    Path('runs')/run/attempt, Path('contact-journals')/run/attempt):
            path = s.mapped(producer, logical/rel, writable=True)
            s.require(not os.path.lexists(path), 'attempt, run or contact custody exists; settlement refused')
            paths.append(str(path))
    return paths


def execute(path):
    path = Path(path).resolve(); doc = s.read(path)
    s.require(set(doc) == {'schema','producer_launch','producer_launcher','source_plan','runtime_receipt','expected','boot_id','state_dir'}
              and doc['schema'] == SCHEMA, 'exact pre-attempt settlement config required')
    producer = s.launcher.load_config(s.pin(doc['producer_launch']))
    helper = s.queue.pin_path(doc['producer_launcher'])
    s.require(helper.name == 'configured_job_launch.py' and helper.resolve() == helper.absolute(), 'launcher pin required')
    exp = doc['expected']
    keys = {'candidate','run_id','attempt_id','fit_run_id','fit_attempt_id','service_session','study_commit'}
    s.require(isinstance(exp, dict) and set(exp) == keys and all(isinstance(v,str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',v) for v in exp.values()), 'exact confined producer identity required')
    s.require(exp['study_commit'] == producer['expected_commit'] and producer['pid_host'] is True, 'producer commit or PID namespace differs')
    args = producer['argv']
    s.require(len(args) in (10,12) and args[1] == producer['repo_path']+'/scripts/analysis/run_real_engine_candidate.py', 'exact real candidate producer required')
    flags = args[2::2]; s.require(len(set(flags)) == len(flags) and set(flags) in ({'--repo','--data','--candidate','--out'}, {'--repo','--data','--candidate','--out','--runtime-config'}), 'producer arguments differ')
    values = dict(zip(flags,args[3::2],strict=True)); logical = s.launcher.absolute(values['--data'])
    s.require(values['--repo'] == producer['repo_path'] and values['--candidate'] == exp['candidate'] and values['--out'].startswith(str(logical/'work/codex')+'/'), 'producer binding differs')
    plan = s.read(s.pin(doc['source_plan']))
    from speech_aware_evidence_acquisition.contracts import ExecutionPlan
    source_root = s.mapped(producer,Path(producer['repo_path']))
    s.require(source_root.is_dir(), 'mapped producer source root absent')
    ep = ExecutionPlan(**plan['execution_plan']).validate(source_root=source_root)
    s.require((ep.run_id,ep.attempt_id) == (exp['run_id'],exp['attempt_id']) and plan.get('cell') == exp['candidate']
              and plan.get('expected_session_id') == exp['service_session'], 'registered plan differs')
    admission = s.launcher.validate_external_admission(producer['external_admission'],producer)
    manifest = s.read(Path(producer['external_admission']['workspace_manifest']))
    bound_plan = s.mapped(producer,Path(producer['environment']['SAEA_EXPERIMENT_WORKSPACE'])/admission['binding']['generation'],writable=False)
    candidate = s.read(s.mapped(producer,Path(producer['environment']['SAEA_EXPERIMENT_WORKSPACE'])/admission['binding']['candidate'],writable=False))
    fit_plan = s.read(s.mapped(producer,Path(producer['environment']['SAEA_EXPERIMENT_WORKSPACE'])/admission['binding']['fit'],writable=False))
    fit_ep = ExecutionPlan(**fit_plan['execution_plan']).validate(source_root=source_root)
    s.require(admission['candidate']==exp['candidate'] and candidate.get('cell')==exp['candidate']
              and (fit_ep.run_id,fit_ep.attempt_id)==(exp['fit_run_id'],exp['fit_attempt_id'])
              and fit_plan.get('cell')==exp['candidate'] and fit_plan.get('expected_session_id')==exp['service_session']
              and bound_plan == Path(doc['source_plan']['path']) and manifest['files'][admission['binding']['generation']] == doc['source_plan']['sha256']
              and all(candidate.get(k)==exp[k] for k in ('run_id','attempt_id','fit_run_id','fit_attempt_id'))
              and candidate.get('service_session')==exp['service_session'], 'admitted run and FIT binding differs')
    receipt = s.read(s.pin(doc['runtime_receipt']))
    instances = receipt['deployment']['instances']
    session = s.read(s.mapped(producer,logical/'serving'/exp['service_session']/'session.json'))
    s.require(session.get('session_id') == exp['service_session'] and session.get('identity',{}).get('runtime_receipt_sha256') == doc['runtime_receipt']['sha256'] and session.get('deployment_instance_ids') == [i['instance_id'] for i in instances], 'resident session differs')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    s.require(re.fullmatch('[0-9a-f-]{36}',doc['boot_id']) and boot == doc['boot_id'], 'current boot differs')
    btime = next(int(x.split()[1]) for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime '))
    locks = s.mapped(producer,logical/'locks',writable=True); out = s.launcher.absolute(doc['state_dir'])
    s.require(out.resolve() == out and not out.is_relative_to(locks) and len(out.parts) >= 4, 'independent settlement directory required')
    for mount in producer['mounts']:
        source = Path(mount['source']).resolve()
        if not mount['readonly']:
            s.require(not out.is_relative_to(source) and not source.is_relative_to(out), 'settlement overlaps producer writable mounts')
    out.mkdir(parents=True,exist_ok=True); s.sync(out.parent)
    with (out/'settlement.guard').open('a+b') as own:
        import fcntl
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with s.guard(locks/'generation.guard') as (check_guard, gi):
            terminal = s.producer_terminal(doc,producer)
            s.require(terminal['state']['ExitCode'] != 0 and s.stamp(terminal['state']['StartedAt']) >= btime, 'producer success or different boot')
            start = s.read(s.mapped(producer,Path(values['--out'])/'started.json',writable=True))
            s.require(start.get('candidate') == exp['candidate'] and start.get('study_commit') == exp['study_commit'], 'producer startup differs')
            lease_path = locks/'flight.lock'; saved = out/'preserved-flight.lock.json'; audit = out/'audit.json'; done = out/'complete.json'
            if done.exists():
                result=s.read(done); identity={'config_sha256':s.launcher.digest(path),'tool_sha256':s.launcher.digest(__file__)}
                prior=s.read(audit)
                s.require(result.get('identity')==identity and prior.get('identity')==identity
                          and prior.get('lease_sha256')==result.get('lease_sha256') and prior.get('terminal')==terminal
                          and prior.get('guard')==gi and result.get('container_id')==terminal['container_id']
                          and s.launcher.digest(saved)==result['lease_sha256'] and s.launcher.digest(audit)==result['audit_sha256'] and not lease_path.exists(), 'completed settlement changed'); return result
            s.require(lease_path.exists() or audit.exists(), 'missing durable audit for detached preserved lease')
            source = lease_path if lease_path.exists() else saved
            s.require(source.is_file() and not source.is_symlink(), 'lease absent or redirected')
            blob=source.read_bytes(); lease=json.loads(blob)
            s.require(lease.get('schema')=='saea-generation-lease-v1' and lease.get('run_id')==exp['run_id'] and lease.get('attempt_id')==exp['attempt_id'] and lease.get('run_ids')==[exp['run_id']] and lease.get('session_id')==exp['service_session'] and lease.get('study_head')==exp['study_commit'], 'lease identity differs')
            pid=lease.get('pid'); s.require(type(pid) is int and pid>0 and not Path('/proc',str(pid)).exists() and start.get('pid')==pid, 'owner PID alive, reused or differs')
            s.require(s.stamp(terminal['state']['StartedAt']) <= start.get('time',0) <= s.stamp(lease['started_at']) <= s.stamp(terminal['state']['FinishedAt']), 'startup or lease outside container lifetime')
            missing=absent(producer,logical,exp); quiet=s.transition.quiescence(instances)
            s.require(s.producer_terminal(doc,producer)==terminal and absent(producer,logical,exp)==missing and source.read_bytes()==blob, 'custody changed during settlement')
            check_guard(); digest=hashlib.sha256(blob).hexdigest(); identity={'config_sha256':s.launcher.digest(path),'tool_sha256':s.launcher.digest(__file__)}
            record={'schema':SCHEMA,'identity':identity,'boot_id':boot,'guard':gi,'terminal':terminal,'lease_sha256':digest,'absent_paths':missing,'quiescence':quiet,'observed_at':time.time()}
            if not saved.exists():
                from speech_aware_evidence_acquisition.core.atomic_file import publish_new
                publish_new(saved,blob)
            s.require(saved.read_bytes()==blob, 'preserved lease differs')
            if not audit.exists(): s.queue.atomic(audit,record)
            else:
                prior=s.read(audit); s.require(prior['identity']==identity and prior['lease_sha256']==digest and prior['terminal']==terminal and prior['guard']==gi, 'prior audit differs')
            if lease_path.exists():
                s.require(lease_path.read_bytes()==blob and absent(producer,logical,exp)==missing, 'lease or custody changed')
                s.require(not Path('/proc',str(pid)).exists() and s.producer_terminal(doc,producer)==terminal, 'producer changed before settlement')
                s.transition.quiescence(instances)
                check_guard(); lease_path.unlink(); s.sync(locks)
            result={'schema':SCHEMA,'identity':identity,'lease_sha256':digest,'audit_sha256':s.launcher.digest(audit),'container_id':terminal['container_id'],'settled':True,'new_model_calls':0}
            s.queue.atomic(done,result); return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    print(json.dumps(execute(p.parse_args().config),sort_keys=True))
