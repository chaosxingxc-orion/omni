"""Resume a registered real workload using explicit native recovery custody."""
import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import sys


def main(repo,data,candidate,out,target_attempt):
    sys.path[:0]=[str(repo/'src'),str(repo/'scripts/analysis')]
    from speech_aware_evidence_acquisition.contracts import ExecutionPlan,assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    from speech_aware_evidence_acquisition.core.flight.campaign import load_campaign_cells,run_campaign
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    from native_recovery_receipt import prepare
    from registered_workload import validate_workload, assert_execution_binding
    from tfgrpo_recovery_queue import admit
    from run_real_engine_candidate import assert_campaign_success
    assert_execution_environment();head=require_clean_study_repo(repo)
    read=lambda p:json.loads(p.read_bytes())
    save=lambda p,d:publish_new(p,(json.dumps(d,sort_keys=True,indent=2)+'\n').encode())
    if out.exists() or not out.resolve().is_relative_to((data/'work/codex').resolve()):raise ValueError('fresh owned recovery output required')
    package=repo/'docs/readiness'/candidate;spec=read(package/'candidate.json')
    rows=validate_workload(repo,data,candidate,spec)
    assert_execution_binding(spec)
    if os.environ.get('SAEA_DEPLOYMENT_PROFILE')!=spec['profile']:raise ValueError('declared deployment differs')
    registered=read(package/'generation-plan-registered-v1.json');original=registered['execution_plan']
    allowed={f.name for f in fields(ExecutionPlan)}
    source=ExecutionPlan(**{k:v for k,v in original.items() if k in allowed},
        owner_allowance=registered.get('owner_allowance'))
    source.validate(source_root=repo)
    base=read(repo/f'configs/baseline/{candidate}.json');arm=base['arm']
    out.mkdir(parents=True)
    recovery=prepare(data,source,target_attempt,arm,out/'recovery.json')['declaration']
    cell,=load_campaign_cells(repo,candidate)
    fit=base.get('live_phase_fit',base.get('native_contact_fit'))
    authority={'authorization_basis':spec['authority'],'fit':fit}
    cell=admit(data,cell,recovery,out,registered=authority)
    runtime=build_flight_runtime(repo,data,model=cell.document['model_fragment'],dataset=cell.document['dataset_fragment'],
        baseline=candidate,experiment=candidate,run_id=cell.run_id,attempt_id=cell.attempt_id,
        runtime_root=data/'work/scratch/runtime-root',lock_path=repo.parents[1]/'docs/datasets.lock.json',native_contact_recovery=recovery)
    save(out/'started.json',{'study_commit':head,'rows':rows,'attempt_id':target_attempt,'recovery':recovery})
    result=run_campaign([runtime],cells=[cell],repo_root=repo,data_root=data,owner='Codex-registered-recovery',expected_session_id=spec['service_session'])
    save(out/'campaign.json',result);assert_campaign_success(result,cell.run_id)
    save(out/'complete.json',{'rows':rows,'result':result,'prior_costs_preserved':True})
    return {'completed':True,'rows':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','data','out'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('candidate','target-attempt'):p.add_argument('--'+k,required=True)
    print(json.dumps(main(**vars(p.parse_args()))))
