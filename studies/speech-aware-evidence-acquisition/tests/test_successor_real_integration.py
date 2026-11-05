"""Opt-in CPU integration with externally mounted, pinned real arm inputs.

SAEA_SUCCESSOR_REAL_CONFIGS names a JSON array of finalize config paths.
Only the terminal custody gate is replaced with an explicit fixture; no execution is admitted.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/environment/configured_checkpoint_successor.py'
spec = importlib.util.spec_from_file_location('real_successor_compiler', SCRIPT)
compiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compiler)
sys.path.insert(0, str(ROOT / 'scripts/analysis'))


@pytest.mark.skipif(not os.environ.get('SAEA_SUCCESSOR_REAL_CONFIGS'), reason='real pinned inputs not mounted')
def test_real_successor_compiler(tmp_path, monkeypatch):
    from registered_workload import validate_workload
    helper_path = os.environ.get('SAEA_ADMISSION_HELPER')
    if not helper_path:
        pytest.fail('SAEA_ADMISSION_HELPER must name the fixed external admission source')
    helper_spec = importlib.util.spec_from_file_location('external_successor_admission', helper_path)
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)

    paths = json.loads(os.environ['SAEA_SUCCESSOR_REAL_CONFIGS'])
    assert len(paths) == 2
    monkeypatch.setattr(compiler, 'validate_terminal_custody', lambda **kwargs: {
        'schema': 'saea-checkpoint-terminal-custody-result-v1',
        'ancestors': [{'fixture_only': True}], 'activation_authorized': False})
    for index, config_path in enumerate(paths):
        config = json.loads(Path(config_path).read_bytes())
        config['output'] = str(tmp_path / str(index))
        source_manifest = json.loads(Path(config['assets_manifest']['path']).read_bytes())
        original = json.loads(Path(source_manifest['files'][config['candidate']['relpath']]['path']).read_bytes())
        config['admission'] = {'workspace': str(tmp_path / ('workspace-' + str(index))),
            'decision_relpath': 'coordination/' + config['fragments']['experiment'] + '/successor-decision.json',
            'predecessor_decision': {'relpath': original['decision_relpath'], 'sha256': '0'*64}}
        config_path = tmp_path / (str(index) + '.json')
        config_path.write_text(json.dumps(config))
        result = compiler.execute(config_path)
        assert result['rows'] == 1080 and result['status'] == 'HELD'
        output = Path(config['output'])
        manifest_path = output / 'assets.json'
        manifest = json.loads(manifest_path.read_bytes())
        assert hashlib.sha256((output / config['plan']['relpath']).read_bytes()).hexdigest() == config['plan']['sha256']
        candidate = json.loads((output / config['candidate']['relpath']).read_bytes())
        index_doc = json.loads((output / 'pin-index.json').read_bytes())
        custody = json.loads(Path(index_doc['terminal_custody']['path']).read_bytes())
        assert custody['ancestors'] == [{'fixture_only': True}]
        admission = json.loads(Path(index_doc['admission_config']['path']).read_bytes())
        experiment = json.loads((output / 'configs/experiment' /
                                 (config['fragments']['experiment'] + '.json')).read_bytes())
        repo, data = Path(config['repo']), Path(config['data'])
        with compiler.asset_environment(manifest_path):
            docs = helper.validate_successor_plans(repo, candidate, experiment, admission['plans'])
            assert docs['generation']['execution_plan']['planned_model_calls'] == 6480
            assert docs['fit']['execution_plan']['planned_model_calls'] == 3240
            assert docs['generation']['execution_plan']['planned_speech_audio_seconds'] == 18508
            assert docs['fit']['execution_plan']['planned_gpu_hours'] == 16
            altered = copy.deepcopy(candidate)
            altered['job_config']['sample_ids'][0] += '-changed'
            with pytest.raises(ValueError):
                validate_workload(repo, data, config['fragments']['experiment'], altered)
            altered = copy.deepcopy(candidate)
            altered['fit_attempt_id'] = 'fit-v1'
            with pytest.raises(ValueError):
                validate_workload(repo, data, config['fragments']['experiment'], altered)
        assert all(Path(pin['path']).is_file() for pin in manifest['files'].values())
