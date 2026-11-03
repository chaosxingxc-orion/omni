"""Preserve a dead worker's lease only after exact observed-contact reconciliation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def settle(repo, data, out, container, expected_run, expected_attempt, expected_pid, plan, plan_sha256, arm='prepared-context-text', runtime_receipt=None, runtime_receipt_sha256=None):
    sys.path[:0]=[str(repo/'src'),str(repo/'scripts/analysis')]
    from tfgrpo_recovery_queue import exclusive,quiescence
    from speech_aware_evidence_acquisition.contracts import ExecutionPlan
    from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    read=lambda p:json.loads(p.read_bytes())
    raw=lambda d:(json.dumps(d,sort_keys=True,indent=2)+'\n').encode()
    if out.exists() or not out.resolve().is_relative_to((data/'work/codex').resolve()):raise ValueError('fresh owned reconciliation required')
    doc,=json.loads(subprocess.check_output(['docker','inspect',container]))
    if doc['State']['Running'] or doc['State']['Status']!='exited' or doc['State']['Pid']!=0:raise ValueError('exact worker must have exited')
    if Path('/proc',str(expected_pid)).exists():raise ValueError('original PID still exists; inspect reuse before proceeding')
    lease=data/'locks/flight.lock';guard=data/'locks/generation.guard'
    with exclusive(guard):
        blob=lease.read_bytes();record=json.loads(blob)
        if any(record.get(k)!=v for k,v in {'run_id':expected_run,'attempt_id':expected_attempt,'pid':expected_pid}.items()):raise ValueError('lease identity differs')
        state=read(data/'attempts'/expected_run/(expected_attempt+'.json'))
        if state['state'] not in ('ABORTED','FINALIZED'):raise ValueError('terminal attempt required')
        if hashlib.sha256(plan.read_bytes()).hexdigest()!=plan_sha256:raise ValueError('registered plan changed')
        registered=read(plan)
        execution=ExecutionPlan(**{k:v for k,v in registered['execution_plan'].items() if k not in ('schema','study')},
            **({'owner_allowance':registered['owner_allowance']} if 'owner_allowance' in registered else {}))
        if (execution.run_id,execution.attempt_id)!=(expected_run,expected_attempt):raise ValueError('plan identity differs')
        journal=ContactJournal(data,execution,arm)
        intents=list(journal.root.glob('*.intent.json'))
        if len(intents)!=state['actual']['calls']:raise ValueError('attempt and contacts differ')
        observed=[]
        for p in intents:
            request=read(p)['request_id'];contact=journal.inspect(request)
            if contact['state']!='observed':raise ValueError('unknown request; no lease settlement')
            observed.append(request)
        if runtime_receipt is None:
            if runtime_receipt_sha256 is not None:raise ValueError('receipt path required with digest')
            quiet=quiescence([8200,8201])
        else:
            receipt_blob=Path(runtime_receipt).read_bytes()
            if hashlib.sha256(receipt_blob).hexdigest()!=runtime_receipt_sha256:
                raise ValueError('runtime receipt changed')
            session_id=record.get('session_id')
            if not isinstance(session_id,str) or Path(session_id).name!=session_id or session_id in ('.','..'):
                raise ValueError('lease must identify a confined resident session')
            session=read(data/'serving'/session_id/'session.json')
            if (session.get('session_id')!=session_id
                    or session.get('identity',{}).get('runtime_receipt_sha256')!=runtime_receipt_sha256):
                raise ValueError('lease resident differs from pinned deployment')
            receipt=json.loads(receipt_blob)
            instances=receipt['deployment']['instances']
            if session.get('deployment_instance_ids')!=[item['instance_id'] for item in instances]:
                raise ValueError('resident deployment membership differs')
            from transition_serving_budget import quiescence as deployment_quiescence
            quiet=deployment_quiescence(instances)
        if lease.read_bytes()!=blob:raise ValueError('lease changed')
        out.mkdir(parents=True)
        publish_new(out/'reconciliation.json',raw({'container':doc['Id'],'state':doc['State'],
            'attempt':state,'observed_requests':observed,'quiescence':quiet,'lease_sha256':hashlib.sha256(blob).hexdigest(),'no_model_calls':True}))
        publish_new(out/'preserved-flight.lock.json',blob)
        if lease.read_bytes()!=blob:raise ValueError('lease changed before release')
        lease.unlink()
        for p in (out,lease.parent):
            fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
    return {'settled':True,'observed_contacts':len(observed),'new_model_calls':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','data','out','plan'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('container','expected-run','expected-attempt'):p.add_argument('--'+k,required=True)
    p.add_argument('--expected-pid',type=int,required=True)
    p.add_argument('--plan-sha256',required=True)
    p.add_argument('--arm',default='prepared-context-text')
    p.add_argument('--runtime-receipt',type=Path)
    p.add_argument('--runtime-receipt-sha256')
    print(json.dumps(settle(**vars(p.parse_args()))))
