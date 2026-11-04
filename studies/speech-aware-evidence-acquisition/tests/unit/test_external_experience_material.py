"""Configured material invokes source replay and publishes pinned workload artifacts."""
import json
import hashlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/analysis')]
import external_experience_material as material
from test_tfgrpo_external_release import material as release_material


@pytest.mark.parametrize('fault', [None, 'source_kind', 'reused_output', 'escaping_output'])
def test_material_build_pins_original_release_and_refuses_unsafe_outputs(release_material, monkeypatch, fault):
    kwargs, candidate, _, _, _, _ = release_material
    repo, data = kwargs['repo'], kwargs['data']
    cfg = {'schema': 'saea-external-experience-material-job-v1', 'source_kind': 'external-tfgrpo-v1',
           'name': 'unit-pair', 'run_id': 'UNIT-PAIR', 'release': {k: str(v) if isinstance(v, Path) else v for k, v in kwargs.items()},
           'runtime_binding': {'model_fragment': 'frozen-actor'}, 'execution': {'admitted_level': 2},
           'params_path': str(data/'params.json'), 'measurement_path': str(data/'measurement.json'),
           'material_relpath': 'configs/evaluation/unit-material.json', 'workload_path': str(data/'workload.json')}
    # Tokenizer measurement has its own tests; this test exercises real source replay and release.
    monkeypatch.setattr(material, 'measure', lambda *args: {'token_counts': {'template': 10}})
    checked = []
    monkeypatch.setattr(material, 'ExternalExperienceWorkload', lambda workload: checked.append(workload))
    if fault == 'source_kind': cfg['source_kind'] = 'unknown'
    if fault == 'reused_output': (data/'params.json').write_text('old')
    if fault == 'escaping_output': cfg['material_relpath'] = '../escape.json'
    if fault:
        with pytest.raises(ValueError): material.build(cfg)
        assert not (repo/'configs/probes/external.txt').exists()
    else:
        result = material.build(cfg)
        assert result['new_model_calls'] == 0 and result['rows'] == 1
        workload = json.loads((data/'workload.json').read_bytes())
        assert workload == checked[0]
        params = json.loads((data/'params.json').read_bytes())
        assert params['candidate']['sha256'] == hashlib.sha256(kwargs['candidate_path'].read_bytes()).hexdigest()
        assert list(candidate['items'].values())[0] in (repo/'configs/probes/external.txt').read_text()
