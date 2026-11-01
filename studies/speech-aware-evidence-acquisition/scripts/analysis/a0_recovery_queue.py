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
# A separately deployed coordinator must execute the frozen target runtime.
# Never import its own evolving worktree's runtime into an existing flight.
TARGET_ROOT=Path(sys.argv[sys.argv.index('--repo')+1]).resolve() if __name__=='__main__' and '--repo' in sys.argv else ROOT
sys.path[:0]=[str(TARGET_ROOT/'src'),str(TARGET_ROOT/'scripts/analysis')]
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



def process_command(pid):
    return Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')[:-1]


def container_identity(pid, container_id):
    if not isinstance(container_id,str) or len(container_id)!=64 or any(c not in '0123456789abcdef' for c in container_id):
        raise ValueError('exact Docker container ID required')
    if container_id not in Path(f'/proc/{pid}/cgroup').read_text():
        raise ValueError('worker cgroup does not match observed container')
    return container_id


def adopt(repo,data,name,out,*,run,wrapper_pid,container_id,container_pid):
    """Attach to a live exact legacy dispatch; never launch a worker here.

    Docker identity comes from host inspect, but its ID must occur in the live
    worker cgroup. Both worker and detached wrapper must remain the same live
    processes through publication. No historical PID alone permits adoption.
    """
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    assert_execution_environment()
    repo,data,out,run=(p.resolve() for p in (repo,data,out,run))
    if not out.is_relative_to(data/'work/codex') or not run.is_relative_to(data/'work/codex'):
        raise ValueError('owned adoption directories required')
    commit=require_clean_study_repo(repo)
    original=read(run/'started.json')
    original_hash=digest((run/'started.json').read_bytes())
    spec=read(repo/'docs/readiness'/name/'candidate.json')
    if (original.get('study_commit')!=commit or original.get('candidate')!=name
            or original.get('rows')!=439 or original.get('profile')!=spec['profile']
            or spec.get('workload_kind')!='tfrl-hardset-a0-observation'):
        raise ValueError('legacy worker registration or frozen commit differs')
    pid=original['pid']
    if type(original.get('time')) not in (int,float) or not 0<original['time']<=time.time():
        raise ValueError('original run start timestamp required')
    if pid!=container_pid:raise ValueError('Docker init PID differs from registered worker')
    started=process_identity(pid);wrapper=process_identity(wrapper_pid)
    if not started or not wrapper or started['state']=='Z' or wrapper['state']=='Z':
        raise ValueError('adoption requires live worker and wrapper')
    boot_epoch=time.time()-float(Path('/proc/uptime').read_text().split()[0])
    if boot_epoch+started['start_ticks']/os.sysconf('SC_CLK_TCK')>original['time']+1:
        raise ValueError('worker PID was born after original start receipt')
    expected=[b'scripts/analysis/run_real_engine_candidate.py',b'--repo',str(repo).encode(),b'--data',str(data).encode(),b'--candidate',name.encode(),b'--out',str(run).encode()]
    if process_command(pid)[1:]!=expected:raise ValueError('worker command differs from exact legacy dispatch')
    wrapper_command=process_command(wrapper_pid)
    if wrapper_command[1:]!=[str(repo/'scripts/flights/nx/hardset_a0_launch.sh').encode(),b'dispatch']:
        raise ValueError('wrapper command differs from detached dispatch')
    container_identity(pid,container_id)
    registered=read(repo/'docs/readiness'/name/'generation-plan-registered-v1.json')
    plan=ExecutionPlan(**registered['execution_plan'])
    if (plan.run_id,plan.attempt_id)!=(spec['run_id'],spec['attempt_id']):
        raise ValueError('adopted plan differs from candidate')
    fit=read(repo/'configs/baseline'/(spec['baseline']+'.json'))['native_contact_fit']
    registration={'repo':str(repo),'candidate':name,'data':str(data),'max_automatic_recoveries':1}
    evidence={'schema':'saea-a0-live-adoption-v1','legacy_run':str(run),'legacy_started_sha256':original_hash,
        'source_commit':commit,'worker':started,'wrapper':wrapper,'container_id':container_id,
        'container_pid':container_pid,'observed_at':time.time(),
        'coordinator_sha256':digest(Path(__file__).read_bytes())}
    with exclusive(out/'queue.lock'):
        if any(out.iterdir()):
            # The lock file itself is the sole allowed pre-existing entry.
            if {p.name for p in out.iterdir()}!={'queue.lock'}:raise ValueError('queue already has custody; supervise it instead')
        if process_identity(pid)!=started or process_identity(wrapper_pid)!=wrapper:
            # Process state can change between R/S naturally; compare identity
            # separately below instead of requiring equal scheduling states.
            for identity in (started,wrapper):
                now=process_identity(identity['pid'])
                if not now or now['state']=='Z' or any(now[k]!=identity[k] for k in ('pid','start_ticks','boot_id')):
                    raise ValueError('worker changed while adopting')
        if digest((run/'started.json').read_bytes())!=original_hash:
            raise ValueError('original started receipt changed while adopting')
        own=out/'attempt-0'
        # Publication is deliberately fail-closed: partial adoption cannot
        # create a new worker; recovery requires all custody records below.
        save(out/'registration.json',registration)
        save(own/'launch-intent.json',{'index':0,'recovery':None,'adopted':True})
        save(own/'started.json',started);save(own/'plan.json',asdict(plan));save(own/'fit-spec.json',fit)
        save(own/'adopted.json',evidence)
    return evidence


def adopted_completion(own,data,plan):
    """Only trust complete legacy campaign after exact worker/wrapper terminal."""
    evidence=read(own/'adopted.json')
    if evidence['coordinator_sha256']!=digest(Path(__file__).read_bytes()):raise ValueError('adopted coordinator changed')
    require_terminal(evidence['worker']);require_terminal(evidence['wrapper'])
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    registration=read(own.parent/'registration.json')
    if require_clean_study_repo(Path(registration['repo']))!=evidence['source_commit']:
        raise ValueError('frozen adopted runtime changed')
    run=Path(evidence['legacy_run'])
    if digest((run/'started.json').read_bytes())!=evidence['legacy_started_sha256']:
        raise ValueError('adopted worker receipt changed')
    if not (run/'complete.json').exists():return None
    from run_real_engine_candidate import assert_campaign_success
    complete=read(run/'complete.json');assert_campaign_success(complete['result'],plan.run_id)
    attempt=read(data/'attempts'/plan.run_id/(plan.attempt_id+'.json'))
    if attempt['state']!='FINALIZED':raise ValueError('legacy completion lacks finalized attempt')
    return {'adopted':True,'rows':439,'source_commit':evidence['source_commit'],'legacy_complete_sha256':digest((run/'complete.json').read_bytes()),'result':complete['result']}


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

            if (own/'adopted.json').exists():
                evidence=read(own/'adopted.json')
                while True:
                    try:require_terminal(evidence['wrapper']);break
                    except ValueError:time.sleep(600)
                completed=adopted_completion(own,data,ExecutionPlan(**read(own/'plan.json')))
                if completed is not None:
                    save(own/'complete.json',completed);return completed
            elif read(own/'launch-intent.json').get('adopted'):
                raise ValueError('partial adoption custody; never dispatch recovery')
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['supervise','worker','adopt'])
    for k in ('repo','data','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--candidate',required=True);p.add_argument('--recovery',type=Path)
    p.add_argument('--legacy-run',type=Path);p.add_argument('--wrapper-pid',type=int);p.add_argument('--container-id');p.add_argument('--container-pid',type=int)
    a=p.parse_args()
    if a.mode=='adopt':
        print(json.dumps(adopt(a.repo,a.data,a.candidate,a.out,run=a.legacy_run,wrapper_pid=a.wrapper_pid,container_id=a.container_id,container_pid=a.container_pid)))
        return
    if a.mode=='worker':
        declaration=({'path':str(a.recovery.resolve()),'sha256':digest(a.recovery.read_bytes())} if a.recovery else None)
        worker(a.repo,a.data,a.candidate,a.out,declaration)
    else:print(json.dumps(supervise(a.repo,a.data,a.candidate,a.out)))


if __name__=='__main__':main()
