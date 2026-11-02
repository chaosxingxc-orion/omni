"""Adopt a live TF-GRPO worker, then resume one verified terminal predecessor.

Run in a detached host-PID formal container. Live observation never restarts a
worker. Unknown generation/FIT outcomes and spent earlier ancestors are held.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
ROOT=Path(__file__).resolve().parents[2]
TARGET=Path(sys.argv[sys.argv.index('--repo')+1]).resolve() if __name__=='__main__' and '--repo' in sys.argv else ROOT
sys.path[:0]=[str(TARGET/'src'),str(TARGET/'scripts/analysis')]
from speech_aware_evidence_acquisition.contracts import ExecutionPlan, assert_execution_environment
from speech_aware_evidence_acquisition.core.contact_journal import NativeContactReplay, ContactJournal, canonical, digest
from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
from speech_aware_evidence_acquisition.core.flight.native_recovery import load_native_recovery, select_recovery_cell
from a0_recovery_queue import read, save, exclusive, process_identity, process_command, container_identity, require_terminal
from native_recovery_receipt import prepare
import tfgrpo_learning_job as job
ARM='prepared-context-text'


def assert_live(identity):
    current=process_identity(identity['pid'])
    if current is None or current['state']=='Z' or any(current[k]!=identity[k] for k in ('pid','start_ticks','boot_id')):
        raise ValueError('worker identity is no longer live')


def quiescence(ports):
    """Observe server queues and sockets, not only dead client processes."""
    if ports != [8200,8201]:raise ValueError('fixed two-instance topology required')
    gauges={}
    for port in ports:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics',timeout=10) as response:
            lines=response.read().decode().splitlines()
        counts={}
        for metric in ('vllm:num_requests_running','vllm:num_requests_waiting'):
            values=[float(line.rsplit(' ',1)[1]) for line in lines if line.startswith(metric+'{') or line.startswith(metric+' ')]
            if not values or any(value!=0 for value in values):raise ValueError('serving has active, waiting, or unreadable requests')
            counts[metric]=sum(values)
        gauges[str(port)]=counts
    connections=[]
    for table in ('tcp','tcp6'):
        for line in Path('/proc/net',table).read_text().splitlines()[1:]:
            fields=line.split()
            if fields[3]=='01' and any(int(endpoint.rsplit(':',1)[1],16) in ports for endpoint in fields[1:3]):
                connections.append({'table':table,'local':fields[1],'remote':fields[2]})
    if connections:raise ValueError('established inference connections remain')
    return {'queues':gauges,'established_connections':connections,'time':time.time()}


def inspect_fit(data, spec):
    root=data/'prepared'/spec['run_id']/spec['attempt_id']
    evidence=[]
    for endpoint in root.glob('*.endpoint.json'):
        path=endpoint.with_name(endpoint.name.removesuffix('.endpoint.json')+'.request-fit.json')
        if not path.is_file():raise ValueError('unknown tokenizer outcome; no automatic resend')
        document=read(path)
        if document.get('output',{}).get('every_request_fits') is not True:
            raise ValueError('failed FIT needs engineering correction, not automatic retry')
        evidence.append({'endpoint_sha256':digest(endpoint.read_bytes()),'artifact_sha256':digest(path.read_bytes())})
    return evidence


def reconcile(data, source, target, out, worker_identity, wrapper_identity, fit_spec, *, observe=quiescence):
    """Require exact terminal identities and quiescence under stable guard."""
    terminal=[require_terminal(i) for i in (worker_identity,wrapper_identity) if i is not None]
    journal=ContactJournal(data,source,ARM)
    states=[journal.inspect(read(p)['request_id'])['state'] for p in journal.root.glob('*.intent.json')]
    if any(s!='observed' for s in states):raise ValueError('unknown generation outcome; never resend')
    fit=inspect_fit(data,fit_spec)
    ap=data/'attempts'/source.run_id/(source.attempt_id+'.json')
    state=read(ap)
    if state['state'] not in ('RESERVED','OPENED','ABORTED'):raise ValueError('terminal attempt is not recoverable')
    guard=data/'locks/generation.guard'; metadata=data/'locks/flight.lock'
    with exclusive(guard):
        for identity in (worker_identity,wrapper_identity):
            if identity is not None:require_terminal(identity)
        lease_blob=metadata.read_bytes() if metadata.exists() else None
        lease=json.loads(lease_blob) if lease_blob else None
        if lease is not None and any(lease.get(k)!=v for k,v in {
            'run_id':source.run_id,'attempt_id':source.attempt_id,'pid':worker_identity['pid']}.items()):
            raise ValueError('foreign lease remains intact')
        quiet=observe([8200,8201])
        if state['state']!='ABORTED':
            from speech_aware_evidence_acquisition.core.attempts import AttemptHandle
            AttemptHandle(ap,source.run_id,source.attempt_id).mark_aborted('supervisor verified exact worker terminal and serving quiescence')
        replay=NativeContactReplay(data,source,replace(source,attempt_id=target),ARM)
        if any(replay.source.inspect(rid)['state']!='observed' for rid in replay.request_ids):
            raise ValueError('unknown source outcome')
        result=prepare(data,source,target,ARM,out/'recovery.json')
        save(out/'reconciliation.json',{'schema':'saea-tfgrpo-terminal-reconciliation-v1','terminal':terminal,
            'quiescence':quiet,'source':replay.source_binding(),'fit':fit,'recovery':result,
            'lease_sha256':digest(lease_blob) if lease_blob else None,'unknown_auto_retry':False})
        if lease_blob is not None:
            if metadata.read_bytes()!=lease_blob:raise ValueError('lease changed while held')
            os.rename(metadata,out/'preserved-flight.lock.json')
            for directory in (out,metadata.parent):
                fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
                try:os.fsync(fd)
                finally:os.close(fd)
    return result['declaration']


def admit(data, cell, declaration, out, *, registered=None):
    """Add exact fresh-attempt authority to the unchanged registered cell."""
    from speech_aware_evidence_acquisition.core.flight.campaign import CampaignCell,declared_readiness
    cell=select_recovery_cell(cell,declaration,data_root=data)
    fit_run,fit_attempt=cell.run_id+'-FIT','learning-fit-v1'
    authority=job.AUTH
    decision='Continue identical registered H1 learning after verified terminal predecessor; preserve known responses and all costs.'
    if registered is not None:
        if (set(registered) != {'authorization_basis','fit'}
                or not isinstance(registered['authorization_basis'],str)
                or not registered['authorization_basis']
                or not isinstance(registered['fit'],dict)):
            raise ValueError('registered recovery requires exact authority and FIT identity')
        fit_run=registered['fit']['run_id'];fit_attempt=registered['fit']['attempt_id']
        authority=registered['authorization_basis']
        decision='Continue unchanged registered workload after verified terminal predecessor; preserve known responses and all costs.'
    did=cell.run_id+'-'+cell.attempt_id.upper()+'-ADMISSION'
    path=data/'coordination/decisions'/(did+'.json')
    doc={'schema':'saea-campaign-decision-v1','status':'decided','decision_id':did,
        'eligible_run_ids':[cell.run_id,fit_run],'eligible_attempt_ids':[cell.attempt_id,fit_attempt],
        'authorization_basis':authority,'owner':'root','coordinator':'Codex',
        'decision':decision,
        'recovery':declaration,'prior_readiness':cell.document['readiness']}
    if path.exists():
        if read(path)!=doc:raise ValueError('existing recovery authority differs')
    else:save(path,doc)
    document=dict(cell.document);ready=dict(document['readiness'])
    ready['requirements']=[*ready.get('requirements',[]),{'kind':'decision','relpath':str(path.relative_to(data)),
        'sha256':digest(path.read_bytes()),'expected_decision_id':did}]
    document['readiness']=ready;cell=CampaignCell.from_document(document)
    if declared_readiness(cell,data).state!='ready':raise ValueError('exact successor remains unready')
    save(out/'admission.json',{'decision':str(path),'sha256':digest(path.read_bytes())})
    return cell


def worker(repo,data,out,prepared_job,recovery):
    assert_execution_environment();commit=require_clean_study_repo(repo)
    complete,reg,_=job.verify_job(repo,data,prepared_job)
    from speech_aware_evidence_acquisition.core.flight.campaign import load_campaign_cells,run_campaign
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    from tfrl_current_generation import SESSION
    from run_real_engine_candidate import assert_campaign_success
    name=complete.get('job',job.NAME);run_id=complete.get('run_id',job.RUN)
    if not os.environ.get('SAEA_DEPLOYMENT_PROFILE'):raise ValueError('recovery worker requires declared deployment profile')
    cell,=load_campaign_cells(repo,name)
    declaration={'path':str(recovery),'sha256':digest(recovery.read_bytes())}
    cell=admit(data,cell,declaration,out)
    if (data/'attempts'/run_id/(cell.attempt_id+'.json')).exists():raise ValueError('existing successor attempt; never relaunch')
    save(out/'started.json',process_identity(os.getpid()))
    runtime=build_flight_runtime(repo,data,model=reg['model_id'],dataset=cell.document['dataset_fragment'],
        baseline=name,experiment=name,run_id=run_id,attempt_id=cell.attempt_id,
        runtime_root=data/'work/scratch/runtime-root',lock_path=repo.parents[1]/'docs/datasets.lock.json',native_contact_recovery=declaration)
    result=run_campaign([runtime],cells=[cell],repo_root=repo,data_root=data,owner='Codex-TFGRPO-supervisor',expected_session_id=complete.get('service_session',SESSION))
    assert_campaign_success(result,run_id)
    save(out/'complete.json',{'result':result,'study_commit':commit,'candidate_accepted':False})


def adopt(repo,data,out,prepared_job,legacy,source_repo,wrapper_pid,container_id,container_pid,source_script,wrapper_script):
    assert_execution_environment();head=require_clean_study_repo(repo)
    job.verify_job(repo,data,prepared_job)
    original=read(legacy/'started.json');source_head=require_clean_study_repo(source_repo)
    if original['study_commit']!=source_head or original['pid']!=container_pid:
        raise ValueError('original worker commit or Docker init identity differs')
    pid=original['pid'];identity=process_identity(pid);wrapper=process_identity(wrapper_pid)
    if not identity or not wrapper:raise ValueError('adoption requires live identities')
    assert_live(identity);assert_live(wrapper);container_identity(pid,container_id)
    command=process_command(pid)
    expected=[b'--repo',str(source_repo).encode(),b'--data',str(data).encode(),b'--out',str(legacy).encode(),b'--job',str(prepared_job).encode()]
    if len(command)<4 or command[1]!=str(source_script).encode() or command[2]!=b'run' or command[3:]!=expected:
        raise ValueError('original worker command differs')
    if process_command(wrapper_pid)[1:] != [str(wrapper_script).encode()]:
        raise ValueError('detached wrapper command differs')
    boot_epoch=time.time()-float(Path('/proc/uptime').read_text().split()[0])
    if boot_epoch+identity['start_ticks']/os.sysconf('SC_CLK_TCK')>original['time']+1:
        raise ValueError('worker PID born after original start')
    recovery=read(legacy/'prepared.json')['recovery']
    recovery_doc=read(Path(recovery['path']));plan=ExecutionPlan(**recovery_doc['target_plan'])
    load_native_recovery(recovery,data_root=data,plan=plan,arm=ARM)
    if plan.attempt_id!=original['attempt_id']:raise ValueError('source plan differs')
    from speech_aware_evidence_acquisition.core.config import compose
    from speech_aware_evidence_acquisition.core.flight.campaign import load_campaign_cells
    source_cell,=load_campaign_cells(source_repo,job.NAME)
    cfg=compose(source_repo,source_cell.document['model_fragment'],source_cell.document['dataset_fragment'],job.NAME,job.NAME).values
    registration={'repo':str(repo),'source_repo':str(source_repo),'data':str(data),'job':str(prepared_job),
        'legacy':str(legacy),'source_commit':source_head,'target_commit':head,'worker':identity,'wrapper':wrapper,
        'worker_command':[s.decode() for s in command],'wrapper_command':[s.decode() for s in process_command(wrapper_pid)],
        'container_id':container_id,'plan':asdict(plan),'fit':cfg['tfgrpo_contact_fit'],
        'source_script':{'path':str(source_script),'sha256':digest(source_script.read_bytes())},
        'wrapper_script':{'path':str(wrapper_script),'sha256':digest(wrapper_script.read_bytes())},
        'legacy_started_sha256':digest((legacy/'started.json').read_bytes()),'coordinator_sha256':digest(Path(__file__).read_bytes()),
        'max_automatic_recoveries':1,'interval_seconds':600}
    with exclusive(out/'queue.lock'):
        if {p.name for p in out.iterdir()}!={'queue.lock'}:raise ValueError('existing supervisor custody')
        assert_live(identity);assert_live(wrapper)
        save(out/'registration.json',registration)
    return registration


def current_command_matches(command, *, source_repo, data, prepared_job, legacy, after):
    expected=[b'scripts/analysis/tfgrpo_learning_job.py',b'run',b'--repo',str(source_repo).encode(),
        b'--data',str(data).encode(),b'--job',str(prepared_job).encode(),b'--out',str(legacy).encode(),b'--after',str(after).encode()]
    if command[1:]!=expected:raise ValueError('current learning worker command differs')


def adopt_current(repo,data,out,prepared_job,legacy,wrapper_pid,container_id,container_pid,wrapper_script,after):
    """Adopt a current registered first attempt without inventing old recovery metadata."""
    assert_execution_environment();head=require_clean_study_repo(repo)
    complete,reg,_=job.verify_job(repo,data,prepared_job)
    original=read(legacy/'started.json')
    if original['study_commit']!=head or original['job']!=complete['job']:raise ValueError('current worker commit/job differs')
    job.verify_release(after,data)
    if original['predecessor_release_sha256']!=digest(after.read_bytes()):raise ValueError('release source changed')
    identity=process_identity(container_pid);wrapper=process_identity(wrapper_pid)
    if not identity or not wrapper:raise ValueError('adoption requires live processes')
    assert_live(identity);assert_live(wrapper);container_identity(container_pid,container_id)
    command=process_command(container_pid)
    current_command_matches(command,source_repo=repo,data=data,prepared_job=prepared_job,legacy=legacy,after=after)
    if process_command(wrapper_pid)[1:]!=[str(wrapper_script).encode()]:raise ValueError('current wrapper command differs')
    plan=ExecutionPlan(**read(repo/'docs/readiness'/complete['job']/'generation-plan.json')['execution_plan']);plan.validate()
    if plan.run_id!=complete['run_id'] or plan.attempt_id!='learning-v1':raise ValueError('current registered plan differs')
    exp=read(repo/'configs/experiment'/(complete['job']+'.json'))
    registration={'repo':str(repo),'source_repo':str(repo),'data':str(data),'job':str(prepared_job),'legacy':str(legacy),
        'source_commit':head,'target_commit':head,'worker':identity,'wrapper':wrapper,'container_id':container_id,
        'worker_command':[v.decode() for v in command],'wrapper_command':[v.decode() for v in process_command(wrapper_pid)],
        'plan':asdict(plan),'fit':exp['tfgrpo_contact_fit'],
        'source_script':{'path':str(repo/'scripts/analysis/tfgrpo_learning_job.py'),'sha256':digest((repo/'scripts/analysis/tfgrpo_learning_job.py').read_bytes())},
        'wrapper_script':{'path':str(wrapper_script),'sha256':digest(wrapper_script.read_bytes())},
        'legacy_started_sha256':digest((legacy/'started.json').read_bytes()),'coordinator_sha256':digest(Path(__file__).read_bytes()),
        'max_automatic_recoveries':1,'interval_seconds':600}
    with exclusive(out/'queue.lock'):
        if {p.name for p in out.iterdir()}!={'queue.lock'}:raise ValueError('existing supervisor custody')
        assert_live(identity);assert_live(wrapper);save(out/'registration.json',registration)
    return registration


def supervise(repo,data,out):
    assert_execution_environment()
    with exclusive(out/'queue.lock'):
        reg=read(out/'registration.json')
        if (reg['repo']!=str(repo) or reg['data']!=str(data) or require_clean_study_repo(repo)!=reg['target_commit']
                or digest(Path(__file__).read_bytes())!=reg['coordinator_sha256']):raise ValueError('supervisor pin changed')
        legacy=Path(reg['legacy']);plan=ExecutionPlan(**reg['plan'])
        if digest((legacy/'started.json').read_bytes())!=reg['legacy_started_sha256']:raise ValueError('source startup receipt changed')
        for key in ('source_script','wrapper_script'):
            if digest(Path(reg[key]['path']).read_bytes())!=reg[key]['sha256']:raise ValueError('adopted launch script changed')
        while True:
            try:
                require_terminal(reg['worker']);require_terminal(reg['wrapper']);break
            except ValueError:time.sleep(600)
        if (legacy/'complete.json').exists():
            from run_real_engine_candidate import assert_campaign_success
            complete=read(legacy/'complete.json');assert_campaign_success(complete['result'],plan.run_id)
            if read(data/'attempts'/plan.run_id/(plan.attempt_id+'.json'))['state']!='FINALIZED':raise ValueError('completion lacks finalized attempt')
            save(out/'complete.json',complete);return complete
        own=out/'successor';own.mkdir(exist_ok=True)
        if not (own/'recovery.json').exists():
            reconcile(data,plan,plan.attempt_id+'-continuation-1',own,reg['worker'],reg['wrapper'],reg['fit'])
        else:
            if not (own/'reconciliation.json').exists() or (data/'locks/flight.lock').exists():
                raise ValueError('partial reconciliation; no launch')
        if not (own/'launch-intent.json').exists():
            save(own/'launch-intent.json',{'recovery_sha256':digest((own/'recovery.json').read_bytes())})
            cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--repo',str(repo),'--data',str(data),
                '--out',str(own),'--job',reg['job'],'--recovery',str(own/'recovery.json')]
            with (own/'worker.log').open('ab') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                code=child.wait()
            save(own/'terminal.json',{'pid':child.pid,'exit_code':code})
        if not (own/'started.json').exists():raise ValueError('unknown launch outcome; never duplicate')
        while True:
            try:require_terminal(read(own/'started.json'));break
            except ValueError:time.sleep(600)
        if not (own/'complete.json').exists():raise ValueError('second interruption requires complete ancestor reconciliation')
        result=read(own/'complete.json');save(out/'complete.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['adopt','adopt-current','supervise','worker'])
    for key in ('repo','data','out'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('job','legacy','source-repo','recovery','source-script','wrapper-script','after'):p.add_argument('--'+key,type=Path)
    for key in ('wrapper-pid','container-pid'):p.add_argument('--'+key,type=int)
    p.add_argument('--container-id');a=p.parse_args()
    if not a.out.resolve().is_relative_to(a.data.resolve()/'work/codex'):raise ValueError('owned output required')
    try:
        if a.mode=='adopt':result=adopt(a.repo,a.data,a.out,a.job,a.legacy,a.source_repo,a.wrapper_pid,a.container_id,a.container_pid,a.source_script,a.wrapper_script)
        elif a.mode=='adopt-current':result=adopt_current(a.repo,a.data,a.out,a.job,a.legacy,a.wrapper_pid,a.container_id,a.container_pid,a.wrapper_script,a.after)
        elif a.mode=='worker':result=worker(a.repo,a.data,a.out,a.job,a.recovery)
        else:result=supervise(a.repo,a.data,a.out)
    except BaseException as error:
        if a.mode in ('worker','supervise'):
            stamp=str(time.time_ns())
            save(a.out/('held-'+stamp+'.json'),{'type':type(error).__name__,'error':str(error),'automatic_retry':False})
        raise
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
