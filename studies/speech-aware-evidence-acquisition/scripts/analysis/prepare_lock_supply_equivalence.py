"""Rebuild full-roster c2 supply on CPU and compare with pinned c17 output.

Writes only a fresh external scratch tree. No serving, tokenization, scoring,
registration, readiness change, or edits to the predecessor are performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(repo, data, candidate_path, experiment, lock_path, out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    from speech_aware_evidence_acquisition.rails.native_episode_config import load_native_episode_config
    from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
    assert_execution_environment()
    if out.exists() or not out.resolve().is_relative_to((data / 'work/codex').resolve()):
        raise ValueError('fresh owned scratch directory required')
    candidate = json.loads(candidate_path.read_bytes())
    exp_path = repo / 'configs/experiment' / (experiment + '.json')
    exp = json.loads(exp_path.read_bytes()); cell, = exp['cells']
    source_path = repo / 'configs/baseline' / (cell['baseline_fragment'] + '.json')
    model_path = repo / 'configs/model' / (cell['model_fragment'] + '.json')
    if candidate['source_sha256'] != sha(source_path) or candidate['model_sha256'] != sha(model_path):
        raise ValueError('candidate source/model differs from actual experiment')
    baseline = candidate['candidate_baseline']
    pin = baseline['prepared_phase_supply']
    previous = (data / pin['relpath']).resolve()
    if not previous.is_relative_to(data.resolve()) or sha(previous) != pin['sha256']:
        raise ValueError('old supply differs')
    original = json.loads(previous.read_bytes())
    out.mkdir(parents=True)
    publish_new(out / 'intent.json', json.dumps({'candidate_sha256': sha(candidate_path),
        'experiment_sha256': sha(exp_path), 'old_supply_sha256': sha(previous),
        'rows': 1080, 'new_model_calls': 0}).encode())
    stage = out / 'study'; stage.mkdir()
    shutil.copytree(repo / 'configs', stage / 'configs')
    for name in ('src', 'docs', 'scripts'):
        shutil.copytree(repo / name, stage / name)
    baseline = dict(baseline); baseline.pop('prepared_phase_supply')
    (stage / 'configs/baseline' / (cell['baseline_fragment'] + '.json')).write_text(json.dumps(baseline))
    runtime = build_flight_runtime(stage, data, model=cell['model_fragment'], dataset=cell['dataset_fragment'],
        baseline=cell['baseline_fragment'], experiment=experiment, run_id=cell['run_id'],
        attempt_id=cell['attempt_id'], runtime_root=data / 'work/scratch/runtime-root', lock_path=lock_path)
    try:
        result = runtime.prepare_phase_supply(out / 'c2-phase-supply.json', max_workers=16)
        current = json.loads((out / 'c2-phase-supply.json').read_bytes())
        if current['output'] != original['output']:
            raise ValueError('recomputed c2 supply differs from old c17 supply')
        native = load_native_episode_config(runtime._composed.values,
                    default_phase_program(runtime._composed.values['policy']))
        if native.max_contacts != 2 or len(runtime._samples) != 1080:
            raise ValueError('unexpected program or roster')
        report = {'schema': 'saea-lock-supply-equivalence-v1', 'rows': len(runtime._samples),
            'candidate_sha256': sha(candidate_path), 'old_supply_sha256': sha(previous),
            'new_supply_sha256': sha(out / 'c2-phase-supply.json'), 'output_equal': True,
            'split_identity_hash': exp['population']['split_identity_hash'],
            'new_model_calls': 0, 'registered': False,
            'remaining': ['actual P/S request replay', 'fresh serving/FIT registration', 'checkpoint migration']}
        publish_new(out / 'complete.json', json.dumps(report, indent=2).encode())
        return report
    except Exception as error:
        publish_new(out / 'failed.json', json.dumps({'error': str(error), 'automatic_retry': False}).encode())
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'data', 'candidate', 'lock', 'out'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--experiment', required=True)
    a=p.parse_args(); print(json.dumps(run(a.repo,a.data,a.candidate,a.experiment,a.lock,a.out)))
