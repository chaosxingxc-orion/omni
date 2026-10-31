"""Bind held A0 continuation to a fresh deployment of its measured profile.

Preserve measured decoding, topology, scientific source and prepared outputs.
This does not deploy, register a resident, admit or execute a model contact.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis'),str(ROOT/'scripts/flights')]
from register_a0_remaining import NAME,RUN
from register_real_engine_candidate import read,sha,write
from bind_a0_selected_profile import verify_selection
from audit_real500_a0_inputs import compare_runtime

RECEIPT='receipts/runtime-a0-remaining353-selected-resident-v1.json'
SESSION='serving-a0-remaining353-selected-resident-v1'


def validate_refresh(spec,cell,old_receipt,new_receipt,transition,runtime):
    if (spec.get('profile_selection_state')!='selected' or cell['readiness']['state']!='held'
            or spec.get('resident_refresh') or spec['runtime_receipt']==RECEIPT):
        raise ValueError('unspent selected held continuation required')
    compare_runtime(old_receipt,new_receipt)
    if old_receipt['deployment']!=new_receipt['deployment']:
        raise ValueError('fresh deployment geometry differs from measured winner')
    if transition['target']!=spec['profile'] or not transition['after']:
        raise ValueError('transition target differs from selected profile')
    expected={i['instance_id'] for i in new_receipt['deployment']['instances']}
    if expected!={i['name'] for i in transition['after']} or expected!=set(runtime['observations']):
        raise ValueError('fresh observed members differ')
    observed={d['Name'].lstrip('/'):d for d in runtime['docker_identity']}
    if set(observed)!=expected:raise ValueError('fresh Docker members differ')
    for name in expected:
        doc=observed[name]
        if (doc['Id']!=runtime['container_ids'][name] or not doc['State']['Running']
                or doc['RestartCount'] or doc['State']['Pid']!=runtime['observations'][name]['pid']
                or doc['Image']!=new_receipt['container']['image_id']
                or doc['Config'].get('Labels',{}).get('saea.deployment.profile')!=spec['profile']):
            raise ValueError('fresh runtime identity differs')


def refresh(repo,data,runtime,transition):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.config import compose,protocol_hash
    from speech_aware_evidence_acquisition.core.flight.repin import repin_package,rewrite_dependent_hashes
    from refresh_registered_plan_pins import refresh_package
    assert_execution_environment()
    pkg=repo/'docs/readiness'/NAME;spec=read(pkg/'candidate.json')
    selection=data/spec['profile_selection']['relpath']
    if sha(selection)!=spec['profile_selection']['sha256']:raise ValueError('selection pin differs')
    decision=read(selection);verify_selection(data,decision)
    if spec['profile']!=decision['profile']:raise ValueError('selected profile differs')
    for run in (RUN,RUN+'-FIT'):
        if any((data/p/run).exists() for p in ('attempts','runs')):raise ValueError('continuation already spent')
    if (data/spec['decision_relpath']).exists():raise ValueError('continuation already admitted')
    exp_path=repo/'configs/experiment'/(NAME+'.json');exp=read(exp_path);cell,=exp['cells']
    model_path=repo/'configs/model'/(cell['model_fragment']+'.json');model=read(model_path)
    if model['runtime_receipt']!=spec['runtime_receipt']:raise ValueError('model/spec receipt mismatch')
    new=runtime/'runtime-receipt.json'
    if sha(new)!=read(runtime/'complete.json')['runtime_receipt_sha256']:raise ValueError('fresh receipt pin differs')
    if not read(transition/'apply.complete.json')['containers']:raise ValueError('transition not complete')
    validate_refresh(spec,cell,read(repo/'docs'/model['runtime_receipt']),read(new),read(transition/'plan.json'),
        {'observations':read(runtime/'observations.json'),'docker_identity':read(runtime/'docker-identity.json'),
         'container_ids':read(transition/'candidate-container-ids.json')})
    archive=data/'work/codex/a0-remaining353-resident-binding-before-v1'
    if archive.exists() or (repo/'docs'/RECEIPT).exists():raise ValueError('refresh already attempted')
    before=compose(repo,cell['model_fragment'],NAME,NAME,NAME)
    old_session=spec['service_session']
    archive.mkdir(parents=True)
    for path in (model_path,exp_path,pkg/'candidate.json',pkg/'protocol-registered-v1.json'):
        target=archive/path.relative_to(repo);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    publish_new(repo/'docs'/RECEIPT,new.read_bytes())
    model['runtime_receipt']=RECEIPT;model_path.write_text(json.dumps(model,indent=2,sort_keys=True)+'\n')
    exp['service_session']=SESSION;exp_path.write_text(json.dumps(exp,indent=2,sort_keys=True)+'\n')
    spec.update(runtime_receipt=RECEIPT,service_session=SESSION,resident_refresh={
        'measured_profile':decision['profile'],'runtime_receipt_sha256':sha(new),
        'transition':str(transition.relative_to(data)),'transition_plan_sha256':sha(transition/'plan.json')})
    (pkg/'candidate.json').write_text(json.dumps(spec,indent=2,sort_keys=True)+'\n')
    protocol=read(pkg/'protocol-registered-v1.json');protocol['resident_refresh']=spec['resident_refresh']
    (pkg/'protocol-registered-v1.json').write_text(json.dumps(protocol,indent=2,sort_keys=True)+'\n')
    current=compose(repo,cell['model_fragment'],NAME,NAME,NAME);split=spec['split_identity_hash']
    mapping={before.config_hash:current.config_hash,
        protocol_hash(before.config_hash,before.values['arm'],split):protocol_hash(current.config_hash,current.values['arm'],split),
        old_session:SESSION};mapping={k:v for k,v in mapping.items() if k!=v}
    prereg=repo/'docs/readiness'/(NAME+'-preregistration.md')
    rewrite_dependent_hashes(repo_root=repo,readiness_dir=pkg,preregistration_path=prereg,
        exposure_ledger_path=repo/'docs/exposure-ledger.md',run_ids={RUN,RUN+'-FIT'},mapping=mapping)
    result=repin_package(repo_root=repo,data_root=data,runtime_root=data/'work/scratch/runtime-root',
        lock_path=repo.parents[1]/'docs/datasets.lock.json',experiments=[(NAME,NAME)],readiness_dir=pkg,
        preregistration_path=prereg,set_ready=False,allow_output_change=False,run_check_fn=None)
    refresh_package(repo,pkg)
    rewrite_dependent_hashes(repo_root=repo,readiness_dir=pkg,preregistration_path=prereg,
        exposure_ledger_path=repo/'docs/exposure-ledger.md',run_ids={RUN,RUN+'-FIT'},mapping={})
    record={'new_model_calls':0,'readiness':'held','repin':result,'resident_refresh':spec['resident_refresh'],
        'remaining':['register fresh resident','admit','commit','full preflight','dispatch']}
    write(pkg/'resident-refresh.json',record);return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','data','runtime','transition'):p.add_argument('--'+key,type=Path,required=True)
    print(json.dumps(refresh(**vars(p.parse_args()))))
