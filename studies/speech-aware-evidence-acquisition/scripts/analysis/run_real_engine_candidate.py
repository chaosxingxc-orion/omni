"""Run one registered real workload using shared campaign/lease/checkpoints.

The host launches this once in a detached formal container. No auto-retry,
synthetic request path, alternate gate, or historical response injection exists.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def assert_campaign_success(result, run_id):
    if result.get('pending') or result.get('completed') != [run_id]:
        raise RuntimeError('shared campaign did not complete the candidate')
    states = result.get('states', {})
    state = states.get(run_id, {})
    if state.get('state') != 'scored':
        raise RuntimeError('candidate generation completed but finalization failed: ' + str(state))


def generation_release_document(runtime, *, candidate, spec, commit, rows, elapsed):
    """Bind a scheduling event to sealed bytes; this is not a scoring verdict.

    Called only by the campaign after its generation lease context exits.
    Consumers must still acquire their own lease and satisfy scientific gates.
    """
    if (runtime.run_id, runtime.attempt_id) != (spec['run_id'], spec['attempt_id']):
        raise ValueError('generation release identity differs from candidate')
    seal = runtime.receipts.require('seal')
    if runtime.receipts.has('score'):
        raise ValueError('generation release must precede scoring')
    # Pin immutable inputs, not the mutable run manifest or growing stage ledger.
    pins = {}
    for key in ('generation_manifest', 'score_ledger_cells', 'score_jobs'):
        path = Path(seal[key + '_path'])
        with path.open('rb') as handle:
            actual = hashlib.file_digest(handle, 'sha256').hexdigest()
        if actual != seal[key + '_sha256']:
            raise ValueError('generation release sealed input changed: ' + key)
        pins[key] = {'path': str(path), 'sha256': actual}
    return {
        'schema': 'saea-generation-released-v1',
        'candidate': candidate, 'run_id': runtime.run_id,
        'attempt_id': runtime.attempt_id, 'study_commit': commit,
        'profile': spec['profile'], 'service_session': spec['service_session'],
        'rows': rows, 'released_at_unix': time.time(),
        'campaign_seconds_to_release': elapsed,
        'generation_sealed': True, 'generation_lease_released': True,
        'scoring_complete': False, 'artifacts': pins,
        'seal_receipt': dict(seal),
        'consumer_requires_own_lease_and_readiness': True,
    }


def run(repo,data,name,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.flight.campaign import load_campaign_cells,declared_readiness,run_campaign
    from speech_aware_evidence_acquisition.core.flight.cli import assert_prepared_artifacts_current
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    assert_execution_environment()
    commit=require_clean_study_repo(repo)
    if out.exists() or not out.resolve().is_relative_to((data/'work/codex').resolve()):raise ValueError('fresh owned output required; inspect existing attempts before resume')
    spec=json.loads((repo/'docs/readiness'/name/'candidate.json').read_bytes())
    from registered_workload import validate_workload
    rows=validate_workload(repo,data,name,spec)
    if os.environ.get('SAEA_DEPLOYMENT_PROFILE')!=spec['profile']:raise ValueError('candidate topology differs')
    cells=load_campaign_cells(repo,name);cell,=cells
    if declared_readiness(cell,data).state!='ready':raise ValueError('candidate not ready')
    for rid,aid in [(spec['run_id'],spec['attempt_id']),(spec['fit_run_id'],spec['fit_attempt_id'])]:
        if (data/'attempts'/rid/(aid+'.json')).exists() or (data/'runs'/rid/aid).exists():raise ValueError('existing attempt: no blind replay')
    out.mkdir(parents=True)
    def save(name,doc):publish_new(out/name,(json.dumps(doc,indent=2)+'\n').encode())
    save('started.json',{'pid':os.getpid(),'time':time.time(),'study_commit':commit,'candidate':name,'profile':spec['profile'],'rows':rows})
    try:
        runtime=build_flight_runtime(repo,data,model=cell.document['model_fragment'],dataset=cell.document['dataset_fragment'],baseline=cell.document['baseline_fragment'],experiment=name,run_id=cell.run_id,attempt_id=cell.attempt_id,runtime_root=data/'work/scratch/runtime-root',lock_path=repo.parents[1]/'docs/datasets.lock.json')
        assert_prepared_artifacts_current([runtime])
        started=time.monotonic()
        def released(flight):
            save('generation-released.json', generation_release_document(
                flight, candidate=name, spec=spec, commit=commit, rows=rows,
                elapsed=time.monotonic()-started))
        result=run_campaign([runtime],cells=cells,repo_root=repo,data_root=data,owner='Codex-real500',expected_session_id=spec['service_session'],on_generation_released=released)
        save('campaign.json',result)
        assert_campaign_success(result, cell.run_id)
        save('complete.json',{'candidate':name,'rows':rows,'campaign_wall_seconds':time.monotonic()-started,'includes_preflight_preprocess_scoring':True,'profile':spec['profile'],'result':result})
        return {'candidate':name,'rows':rows,'completed':True}
    except BaseException as error:
        save('failed.json',{'type':type(error).__name__,'error':str(error),'automatic_retry':False,'preserve':'attempt costs, FIT contacts, complete-row checkpoints and unresolved requests'})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','data','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--candidate',required=True);a=p.parse_args()
    print(json.dumps(run(a.repo,a.data,a.candidate,a.out)))
