"""Detached A0 coordinator: terminal-child reconciliation and one safe replay.

Run this entrypoint in the formal host-PID container under nohup/setsid. The
existing campaign owns generation leases and scoring. Unknown contacts and a
second interrupted predecessor chain are held, never restarted as fresh work.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis')]
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal, NativeContactReplay, canonical, digest
from speech_aware_evidence_acquisition.contracts import ExecutionPlan
from native_recovery_receipt import prepare


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    publish_new(path,canonical(value))


def read(path): return json.loads(path.read_bytes())


def process_identity(pid):
    """Linux host PID namespace identity; PID reuse is not worker liveness."""
    try:
        raw=Path(f'/proc/{pid}/stat').read_text()
        rest=raw[raw.rindex(')')+2:].split()
        return {'pid':pid,'start_ticks':int(rest[19]),'state':rest[0],
                'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except FileNotFoundError:return None


def require_terminal(started):
    current=process_identity(started['pid'])
    if current is not None and current['state']!='Z' and all(current[k]==started[k] for k in ('pid','start_ticks','boot_id')):
        raise ValueError('worker still live; observation timeout never authorizes replay')
    return {'started':started,'observed':current,'observed_at':time.time(),'terminal':True}


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def inspect_contacts(data, plan, arm):
    journal=ContactJournal(data,plan,arm)
    states=[]
    for path in sorted(journal.root.glob('*.intent.json')):
        value=read(path);record=journal.inspect(value['request_id'])
        states.append({'request_id':value['request_id'],'sample_id':value['sample_id'],'state':record['state']})
    return states


def reconcile(data, source, target, arm, root, started, fit_spec=None):
    """Abort only a dead exact worker; preserve lease bytes and source spend.

    A known-response first predecessor is the only replay shape supported by
    NativeContactReplay. A later chain is recorded and held before mutation.
    """
    terminal=require_terminal(started)
    journal=ContactJournal(data,source,arm)
    if (data/'runs'/source.run_id/source.attempt_id/'resume-declaration.json').exists():
        raise ValueError('repeated interruption requires multi-predecessor reconciliation; source ancestry preserved')
    states=inspect_contacts(data,source,arm)
    if not states or any(s['state']!='observed' for s in states):
        raise ValueError('missing or unknown generation contact; no automatic resend')
    # The worker completed its FIT before every sent generation request. Extra
    # successful FITs are reusable; unresolved tokenizer attempts remain held by
    # the original registered FIT gate, never granted a new tokenizer budget.
    if fit_spec is not None:
        fit_root=data/'prepared'/fit_spec['run_id']/fit_spec['attempt_id']
        custody={Path(read(p)['receipt']['path']).resolve() for p in journal.root.glob('*.fit.json')}
        for endpoint in fit_root.glob('*.contacts.jsonl'):
            artifact=endpoint.with_name(endpoint.name.removesuffix('.contacts.jsonl'))
            if not artifact.is_file() or artifact.resolve() not in custody:
                raise ValueError('unresolved FIT endpoint or missing custody; preserve lease for reconciliation')
    attempt=data/'attempts'/source.run_id/(source.attempt_id+'.json')
    document=read(attempt)
    if document['state']=='FINALIZED':raise ValueError('finalized source is never replayed')
    if document['state'] not in ('RESERVED','OPENED','ABORTED'):raise ValueError('unrecognized source attempt state')
    guard=data/'locks/generation.guard'; metadata=data/'locks/flight.lock'
    with exclusive(guard):
        # Even with the kernel lock free, exact owner identity and contact
        # accounting are required before removing stale lease metadata.
        lease=read(metadata) if metadata.exists() else None
        if lease is not None and any(lease.get(k)!=v for k,v in {
            'run_id':source.run_id,'attempt_id':source.attempt_id,'pid':started['pid']}.items()):
            raise ValueError('lease belongs to another worker; preserve it')
        if document['state']!='ABORTED':
            from speech_aware_evidence_acquisition.core.attempts import AttemptHandle
            AttemptHandle(attempt,source.run_id,source.attempt_id).mark_aborted('queue verified exact worker terminal; observed contacts retained')
        # Verifies source actual usage exactly equals all journaled intent costs.
        replay=NativeContactReplay(data,source,replace(source,attempt_id=target),arm)
        result=prepare(data,source,target,arm,root/'recovery.json')
        evidence={'schema':'saea-a0-terminal-reconciliation-v1','terminal':terminal,
                  'source':replay.source_binding(),'contacts':states,'lease':lease,
                  'recovery':result,'unknown_auto_retry':False}
        save(root/'reconciliation.json',evidence)
        if lease is not None:
            if read(metadata)!=lease:raise ValueError('lease changed during reconciliation')
            os.replace(metadata,root/'preserved-flight.lock.json')
            fd=os.open(metadata.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
    return result['declaration']


def worker(repo,data,name,out,recovery=None):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.flight.campaign import load_campaign_cells,run_campaign
    from speech_aware_evidence_acquisition.core.flight.native_recovery import select_recovery_cell
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    from speech_aware_evidence_acquisition.core.flight.cli import assert_prepared_artifacts_current
    from run_real_engine_candidate import assert_campaign_success
    from registered_workload import validate_workload
    assert_execution_environment();commit=require_clean_study_repo(repo)
    save(out/'started.json',process_identity(os.getpid()))
    spec=read(repo/'docs/readiness'/name/'candidate.json')
    if spec.get('workload_kind')!='tfrl-hardset-a0-observation':raise ValueError('only registered hardset A0 workload supported')
    rows=validate_workload(repo,data,name,spec)
    if os.environ.get('SAEA_DEPLOYMENT_PROFILE')!=spec['profile']:raise ValueError('deployment profile differs')
    cell,=load_campaign_cells(repo,name)
    if recovery:cell=select_recovery_cell(cell,recovery,data_root=data)
    runtime=build_flight_runtime(repo,data,model=cell.document['model_fragment'],dataset=cell.document['dataset_fragment'],baseline=cell.document['baseline_fragment'],experiment=name,run_id=cell.run_id,attempt_id=cell.attempt_id,runtime_root=data/'work/scratch/runtime-root',lock_path=repo.parents[1]/'docs/datasets.lock.json',**({'native_contact_recovery':recovery} if recovery else {}))
    if (data/'attempts'/cell.run_id/(cell.attempt_id+'.json')).exists() or (data/'runs'/cell.run_id/cell.attempt_id).exists():
        raise ValueError('target already has state outside this worker; never adopt it')
    base=read(repo/'configs/baseline'/(cell.document['baseline_fragment']+'.json'))
    save(out/'fit-spec.json',base['native_contact_fit'])
    assert_prepared_artifacts_current([runtime]);runtime.preflight()
    save(out/'plan.json',asdict(runtime._plan))
    result=run_campaign([runtime],cells=[cell],repo_root=repo,data_root=data,owner='Codex-hardset-a0-queue',expected_session_id=spec['service_session'])
    save(out/'campaign.json',result);assert_campaign_success(result,cell.run_id)
    save(out/'complete.json',{'rows':rows,'attempt_id':cell.attempt_id,'commit':commit,'result':result})


def supervise(repo,data,name,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    out=out.resolve()
    if not out.is_relative_to((data/'work/codex').resolve()):raise ValueError('owned queue directory required')
    out.mkdir(parents=True,exist_ok=True)
    with exclusive(out/'queue.lock'):
        registration={'repo':str(repo.resolve()),'candidate':name,'data':str(data.resolve()),'max_automatic_recoveries':1}
        if (out/'registration.json').exists():
            if read(out/'registration.json')!=registration:raise ValueError('queue identity changed')
        else:save(out/'registration.json',registration)
        recovery=None
        for index in range(2):
            own=out/f'attempt-{index}'; own.mkdir(exist_ok=True)
            if (own/'complete.json').exists():return read(own/'complete.json')
            if not (own/'launch-intent.json').exists():
                save(own/'launch-intent.json',{'index':index,'recovery':recovery})
                command=[sys.executable,str(Path(__file__).resolve()),'worker','--repo',str(repo),'--data',str(data),'--candidate',name,'--out',str(own)]
                if recovery:
                    command+=['--recovery',str(out/'attempt-0/recovery.json')]
                with (own/'worker.log').open('ab') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    code=child.wait()
                save(own/'terminal.json',{'pid':child.pid,'exit_code':code,'waited':True})
            # Restart only observes an existing worker; a missing startup receipt
            # is ambiguous and must never create a second worker.
            if not (own/'started.json').exists():raise ValueError('launch outcome lacks worker identity; no automatic relaunch')
            started=read(own/'started.json')
            while True:
                try:
                    require_terminal(started)
                    break
                except ValueError:
                    # A restarted coordinator observes the same exact worker;
                    # ten-minute polls never manufacture a terminal event.
                    time.sleep(600)

            if (own/'complete.json').exists():return read(own/'complete.json')
            if index:
                save(out/'chain-held.json',{'reason':'second interruption needs multi-predecessor reconciliation','attempts':[str(out/'attempt-0'),str(own)],'automatic_retry':False})
                raise ValueError('recovery chain held with both predecessor records intact')
            plan=ExecutionPlan(**read(own/'plan.json'))
            if (own/'recovery.json').exists():
                path=own/'recovery.json';recovery={'path':str(path),'sha256':digest(path.read_bytes())}
                if not (own/'reconciliation.json').exists():raise ValueError('partial reconciliation; inspect preserved lease before proceeding')
                if (data/'locks/flight.lock').exists():raise ValueError('lease still present after reconciliation; inspect before proceeding')
            else:recovery=reconcile(data,plan,plan.attempt_id+'-recovery-1','raw-speech-agent-native-tools',own,started,read(own/'fit-spec.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['supervise','worker'])
    for k in ('repo','data','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--candidate',required=True);p.add_argument('--recovery',type=Path)
    a=p.parse_args()
    if a.mode=='worker':
        declaration=({'path':str(a.recovery.resolve()),'sha256':digest(a.recovery.read_bytes())} if a.recovery else None)
        worker(a.repo,a.data,a.candidate,a.out,declaration)
    else:print(json.dumps(supervise(a.repo,a.data,a.candidate,a.out)))


if __name__=='__main__':main()
