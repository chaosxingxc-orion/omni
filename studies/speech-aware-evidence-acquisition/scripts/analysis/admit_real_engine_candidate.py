"""Bind a registered real workload to an observed resident and refresh shared preparation.

Owner authorization is already recorded. This command makes no inference calls;
the normal clean-tree/preflight/lease/FIT checks still precede dispatch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/flights/direct'),str(ROOT/'scripts/flights')]
from register_real_engine_candidate import read,sha,write


def admit(repo,data,name):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.session.serving import find_resident_session,assert_identity_matches
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    from speech_aware_evidence_acquisition.core.flight.repin import repin_package,rewrite_dependent_hashes
    from refresh_registered_plan_pins import refresh_package
    from thinking_system_mint_receipt import mint
    assert_execution_environment()
    pkg=repo/'docs/readiness'/name;spec=read(pkg/'candidate.json')
    exp=read(repo/'configs/experiment'/f'{name}.json');cell,=exp['cells']
    if cell['readiness']['state']!='held':raise ValueError('candidate not held')
    from registered_workload import validate_workload
    rows=validate_workload(repo,data,name,spec)
    receipt=repo/'docs'/spec['runtime_receipt']
    if not receipt.is_file():raise ValueError('observed runtime receipt missing')
    lock=repo.parents[1]/'docs/datasets.lock.json'
    runtime=build_flight_runtime(repo,data,model=cell['model_fragment'],dataset=cell['dataset_fragment'],baseline=cell['baseline_fragment'],experiment=name,run_id=cell['run_id'],attempt_id=cell['attempt_id'],runtime_root=data/'work/scratch/runtime-root',lock_path=lock)
    resident=find_resident_session(data)
    if resident is None or resident[1]['session_id']!=spec['service_session']:raise ValueError('expected resident absent')
    assert_identity_matches(resident[1],runtime.session_required_identity())
    from speech_aware_evidence_acquisition.core.session.receipt import SessionReceipt
    from speech_aware_evidence_acquisition.core.session.vllm_identity import verify_vllm_session
    doc=read(receipt)
    for instance in doc['deployment']['instances']:
        s=SessionReceipt.from_json(read(data/'serving'/instance['instance_id']/'session-receipt.json'))
        verify_vllm_session(s,doc)
    runs=[spec['run_id'],spec['fit_run_id']]
    draft={'schema':'saea-campaign-decision-v1','status':'DRAFT-NOT-MINTED','decision_id':spec['decision_id'],'eligible_run_ids':runs,'eligible_attempt_ids':[spec['attempt_id'],spec['fit_attempt_id']],'authorization_basis':spec['authority'],'owner':'root','decision':'Execute the exact registered real population through shared c2 runtime under recorded owner authority; no synthetic replay or test tuning.','antecedents':{'owner':{'path':spec['authority'],'sha256':sha(repo/spec['authority'])},'protocol':{'path':str((pkg/'protocol-registered-v1.json').relative_to(repo)),'sha256':sha(pkg/'protocol-registered-v1.json')},'runtime':{'path':str(receipt.relative_to(repo)),'sha256':sha(receipt)}},'service':{'expected_session_id':spec['service_session'],'profile':spec['profile'],'session_sha256':sha(resident[0]/'session.json')},'population':{'count':rows,'split_identity_hash':spec['split_identity_hash'],'test_contact_authorized':False},'budget':{kind:read(pkg/f'{kind}-plan-registered-v1.json')['execution_plan'] for kind in ('generation','fit')}}
    write(pkg/'root-admission-draft-v1.json',draft)
    family=SimpleNamespace(PACKAGE=str(pkg.relative_to(repo)),DECISION_ID=spec['decision_id'],DECISION_RELPATH=spec['decision_relpath'],eligible_run_ids=lambda:runs)
    result=mint(str(repo),str(data),'Codex',False,family=family,tool='scripts/analysis/admit_real_engine_candidate.py')
    before=(data/spec['decision_relpath']).read_bytes()
    repin=repin_package(repo_root=repo,data_root=data,runtime_root=data/'work/scratch/runtime-root',lock_path=lock,experiments=[(name,name)],readiness_dir=pkg,preregistration_path=repo/'docs/readiness'/f'{name}-preregistration.md',decision=(spec['decision_relpath'],spec['decision_id']),set_ready=True,allow_output_change=False,run_check_fn=None)
    refresh_package(repo,pkg)
    rewrite_dependent_hashes(repo_root=repo,readiness_dir=pkg,preregistration_path=repo/'docs/readiness'/f'{name}-preregistration.md',exposure_ledger_path=repo/'docs/exposure-ledger.md',run_ids=set(runs),mapping={})
    if before!=(data/spec['decision_relpath']).read_bytes():raise ValueError('admission mutated')
    write(pkg/'admission-preparation.json',{'new_model_calls':0,'repin':repin,'admission_sha256':hashlib.sha256(before).hexdigest(),'remaining':['commit exact runtime configuration','shared preflight','guarded detached dispatch']})
    return {'candidate':name,'admission_minted':True,'new_model_calls':0,'preflight_pending':True}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--candidate',required=True)
    a=p.parse_args();print(json.dumps(admit(a.repo,a.data,a.candidate)))
