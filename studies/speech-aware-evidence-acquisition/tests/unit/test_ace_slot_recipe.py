"""Engineering fixture: exercise the offline recipe, not scientific evidence."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from speech_aware_evidence_acquisition.rails.semantic_knowledge_edits import apply_slot_value_edits

from test_semantic_knowledge_edits import rows


STUDY = Path(__file__).resolve().parents[2]


def test_recipe_cli_outputs_valid_single_card_and_rejects_forbidden(tmp_path):
    def put(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding='utf-8')
        return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    base = rows()
    group = 'slurp:train:fixture'
    sample = group + ':recording'
    cfg = {
        'schema': 'saea-ace-slot-recipe-v1',
        'authored': put('authored.json', {'schema': 'saea-authored-semantic-knowledge-source-v1', 'rows': base}),
        'records': put('records.json', [{'source_id': group, 'source_sample_ids': [sample], 'kind': 'engineering fixture'}]),
        'pool': put('pool.json', {'kind': 'engineering fixture'}),
        'forbidden_sample_ids': put('forbidden.json', []),
        'slot_ids': ['date'], 'fields': ['extraction_cues'],
        'max_edits': 1, 'max_field_chars': 300, 'output': str(tmp_path / 'prepared'),
    }
    config = tmp_path / 'recipe.json'
    config.write_text(json.dumps(cfg), encoding='utf-8')
    cmd = [sys.executable, str(STUDY / 'scripts/analysis/ace_slot_recipe.py'), '--config', str(config)]
    env = dict(os.environ, PYTHONPATH=str(STUDY / 'src'))
    completed = subprocess.run(cmd, cwd=STUDY, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    out = Path(cfg['output'])
    dynamic = json.loads((out / 'dynamic.json').read_text())
    messages = json.loads((out / 'messages.json').read_text())
    assert messages == [{'role': 'user', 'content': messages[0]['content']}]
    assert json.loads(messages[0]['content']) == dynamic
    assert dynamic['base_rows'] == base == json.loads((out / 'base.json').read_text())
    contract = json.loads((out / 'contract.json').read_text())
    assert dynamic['edit_contract'] == contract
    for name, pin in json.loads((out / 'pins.json').read_text()).items():
        assert hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest() == pin['sha256']
        assert Path(pin['path']) == out / (name + '.json')
    target = contract['editable'][0]
    proposal = {'schema': 'saea-slot-value-edits-v1', 'base_rows_sha256': contract['base_rows_sha256'],
                'edits': [{**target, 'replacement': ['Keep the complete spoken date.'],
                           'evidence_group_ids': [group]}]}
    result = apply_slot_value_edits(base, proposal, contract=contract,
                                    verified_generation_ref='artifact-run:' + '1' * 64)
    assert result['admission'] == 'candidate-only'
    assert result['changed_fields'] == [{'slot_id': 'date', 'field': 'extraction_cues'}]
    assert result['rows'][0]['intent_bindings'] == base[0]['intent_bindings']
    assert set(result['rows'][0]['source_support']['source_sample_ids']) == set(base[0]['source_support']['source_sample_ids']) | {sample}

    # A different recording of the same utterance is also forbidden.
    cfg['forbidden_sample_ids'] = put('forbidden-overlap.json', [group + ':other-recording'])
    cfg['output'] = str(tmp_path / 'rejected')
    config.write_text(json.dumps(cfg), encoding='utf-8')
    rejected = subprocess.run(cmd, cwd=STUDY, env=env, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert 'construction evidence intersects forbidden samples' in rejected.stderr
    assert not Path(cfg['output']).exists()
