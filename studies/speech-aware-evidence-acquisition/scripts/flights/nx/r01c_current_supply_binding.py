"""Adopt an existing exact R01C supply and verify shared dispatch identity.

Never rebuild supply, admit a cell or invoke a model. Keep the original derived
configuration and completed preparation separately addressable.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / 'src'), str(Path(__file__).parent)]
import r01c_current_baseline as baseline
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
read, sha, raw = baseline.read, baseline.sha, baseline.raw


def bound_documents(base, exp, completion):
    if (completion['schema'] != 'saea-r01c-current-supply-v1' or completion['rows'] != 625
            or completion['model_calls'] != 0 or completion['readiness'] != 'held'
            or completion['registered'] is not False):
        raise ValueError('completed held625 supply required')
    base, exp = copy.deepcopy(base), copy.deepcopy(exp)
    cell, = exp['cells']
    if (cell['cell_id'] != baseline.CELL or cell['run_id'] != baseline.RUN
            or cell['readiness']['state'] != 'held' or cell['allowed_execution_stages'] != ['check']
            or 'prepared_phase_supply' in base):
        raise ValueError('unbound held construction baseline required')
    pin = completion['supply']
    if pin['relpath'] != f'prepared/{baseline.RUN}/phase-supply-v1.json':
        raise ValueError('own supply required')
    base['prepared_phase_supply'] = pin
    cell['readiness']['requirements'].append({'kind': 'preparation', **pin})
    return base, exp


def bind(repo, data, preparation, derivation, out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    assert_execution_environment(); head = require_clean_study_repo(repo)
    for p in (preparation, derivation, out): baseline.comparator.confined(data, str(p))
    if out.exists() or not out.resolve().is_relative_to(data.resolve() / 'work/codex'):
        raise ValueError('fresh owned binding record required')
    if any((data / kind / baseline.RUN).exists() for kind in ('runs', 'attempts')):
        raise ValueError('construction already spent')
    prep, source = read(preparation), read(derivation)
    for rel, digest in source['output_hashes'].items():
        if sha(repo / rel) != digest: raise ValueError('original derived configuration changed')
    target = baseline.comparator.confined(data, prep['supply']['relpath'])
    if sha(target) != prep['supply']['sha256']: raise ValueError('prepared supply changed')
    # Later implementation commits are allowed; the original preparation commit
    # must still contain exactly the configurations whose hashes were derived.
    for rel, digest in source['output_hashes'].items():
        import hashlib
        old = subprocess.check_output(['git', '-c', 'safe.directory=' + str(repo), '-C', str(repo),
            'show', prep['source_commit'] + ':' + rel])
        if hashlib.sha256(old).hexdigest() != digest: raise ValueError('preparation source differs')
    base_path = repo / f'configs/baseline/{baseline.CELL}.json'
    exp_path = repo / f'configs/experiment/{baseline.CELL}.json'
    base, exp = bound_documents(read(base_path), read(exp_path), prep)
    out.mkdir(parents=True)
    for path in (base_path, exp_path): publish_new(out / path.name.replace('.json', '.' + path.parent.name + '.json'), path.read_bytes())
    publish_new(out / 'intent.json', raw({'source_commit': head, 'preparation_sha256': sha(preparation),
        'derivation_sha256': sha(derivation), 'new_model_calls': 0}))
    base_path.write_bytes(raw(base)); exp_path.write_bytes(raw(exp))
    cell, = exp['cells']
    runtime = build_flight_runtime(repo, data, model=cell['model_fragment'], dataset=cell['dataset_fragment'],
        baseline=cell['baseline_fragment'], experiment=baseline.CELL, run_id=baseline.RUN,
        attempt_id=cell['attempt_id'], runtime_root=data / 'work/scratch/runtime-root',
        lock_path=repo.parents[1] / 'docs/datasets.lock.json')
    checks = runtime.verify_prepared_artifacts()
    if not checks or not all(c['identity_ok'] and c.get('seed_ok', False) for c in checks):
        raise ValueError('shared prepared identity or dispatch seed validation failed')
    result = {'schema': 'saea-r01c-current-supply-binding-v1', 'rows': 625, 'checks': checks,
        'supply': prep['supply'], 'readiness': 'held', 'new_model_calls': 0, 'registered': False,
        'source_commit': head, 'preparation_sha256': sha(preparation), 'derivation_sha256': sha(derivation),
        'output_hashes': {str(p.relative_to(repo)): sha(p) for p in (base_path, exp_path)}}
    publish_new(out / 'complete.json', raw(result)); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('repo', 'data', 'preparation', 'derivation', 'out'): p.add_argument('--' + key, type=Path, required=True)
    print(json.dumps(bind(**vars(p.parse_args()))))
