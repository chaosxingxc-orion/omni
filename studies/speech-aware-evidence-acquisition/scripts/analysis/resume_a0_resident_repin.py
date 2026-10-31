"""Finish a recorded held A0 refresh after its model-fragment commit boundary.

The failed refresh's changed model is committed before shared repin, whose clean
scope deliberately excludes model fragments. No deployment or generation here.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/analysis'), str(ROOT/'scripts/flights')]
import refresh_a0_resident_binding as original
read, sha, NAME, RUN = original.read, original.sha, original.NAME, original.RUN


def validate_partial(old_model, model, spec, exp, receipt_sha):
    expected = copy.deepcopy(old_model); expected['runtime_receipt'] = original.RECEIPT
    cell, = exp['cells']
    if model != expected:
        raise ValueError('model changed beyond runtime receipt')
    if (cell['run_id'] != RUN or cell['readiness']['state'] != 'held'
            or exp['service_session'] != original.SESSION
            or spec['runtime_receipt'] != original.RECEIPT or spec['service_session'] != original.SESSION
            or spec['resident_refresh']['runtime_receipt_sha256'] != receipt_sha):
        raise ValueError('partial held refresh identity differs')


def resume(repo, data, runtime, transition):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.flight.repin import repin_package, rewrite_dependent_hashes
    from refresh_registered_plan_pins import refresh_package
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    assert_execution_environment()
    if subprocess.check_output(['git', '-c', 'safe.directory='+str(repo), '-C', str(repo), 'status', '--porcelain']).strip():
        raise ValueError('commit verified partial held state before repinning')
    pkg = repo/'docs/readiness'/NAME
    if (pkg/'resident-refresh.json').exists(): raise ValueError('refresh already completed')
    spec, exp = read(pkg/'candidate.json'), read(repo/'configs/experiment'/(NAME+'.json'))
    cell, = exp['cells']; model_rel = 'configs/model/'+cell['model_fragment']+'.json'
    archive = data/'work/codex/a0-remaining353-resident-binding-before-v1'
    old_spec = read(archive/'docs/readiness'/NAME/'candidate.json')
    old_model, model = read(archive/model_rel), read(repo/model_rel)
    new = runtime/'runtime-receipt.json'
    validate_partial(old_model, model, spec, exp, sha(new))
    selection = data/spec['profile_selection']['relpath']
    if sha(selection) != spec['profile_selection']['sha256']: raise ValueError('selection changed')
    original.verify_selection(data, read(selection))
    if sha(repo/'docs'/original.RECEIPT) != sha(new): raise ValueError('copied runtime receipt changed')
    if sha(new) != read(runtime/'complete.json')['runtime_receipt_sha256']: raise ValueError('capture pin differs')
    if sha(transition/'plan.json') != spec['resident_refresh']['transition_plan_sha256']: raise ValueError('transition changed')
    for run in (RUN, RUN+'-FIT'):
        if any((data/k/run).exists() for k in ('runs','attempts')): raise ValueError('generation already spent')
    if (data/spec['decision_relpath']).exists(): raise ValueError('already admitted')
    original.validate_refresh(old_spec, cell, read(repo/'docs'/old_model['runtime_receipt']), read(new),
        read(transition/'plan.json'), {'observations':read(runtime/'observations.json'),
        'docker_identity':read(runtime/'docker-identity.json'), 'container_ids':read(transition/'candidate-container-ids.json')})
    if not read(transition/'apply.complete.json')['containers']: raise ValueError('deployment incomplete')
    prereg = repo/'docs/readiness'/(NAME+'-preregistration.md')
    result = repin_package(repo_root=repo, data_root=data, runtime_root=data/'work/scratch/runtime-root',
        lock_path=repo.parents[1]/'docs/datasets.lock.json', experiments=[(NAME,NAME)], readiness_dir=pkg,
        preregistration_path=prereg, set_ready=False, allow_output_change=False, run_check_fn=None)
    refresh_package(repo,pkg)
    rewrite_dependent_hashes(repo_root=repo, readiness_dir=pkg, preregistration_path=prereg,
        exposure_ledger_path=repo/'docs/exposure-ledger.md', run_ids={RUN,RUN+'-FIT'}, mapping={})
    record = {'new_model_calls':0, 'readiness':'held', 'repin':result, 'resident_refresh':spec['resident_refresh'],
        'recovered_after_model_commit':True, 'remaining':['register fresh resident','admit','commit','full preflight','dispatch']}
    publish_new(pkg/'resident-refresh.json', (json.dumps(record,indent=2,sort_keys=True)+'\n').encode())
    return record


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','data','runtime','transition'): p.add_argument('--'+key,type=Path,required=True)
    print(json.dumps(resume(**vars(p.parse_args()))))
