import importlib.util
import json
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/analysis/sealed_coverage_reader.py'
spec = importlib.util.spec_from_file_location('sealed_coverage_reader', SCRIPT)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def pinned(path, value):
    raw = (json.dumps(value) + '\n').encode()
    path.write_bytes(raw)
    return {'path': str(path), 'sha256': reader.sha(raw)}


def make_config(tmp_path):
    ids = [f's{i}' for i in range(1080)]
    rows = [{'sample_id': i, 'text': '{"scenario":"x","action":"y","entities":[]}',
             'native_episode_receipt': {}} for i in ids]
    output = tmp_path / 'outputs.jsonl'
    output.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    manifest = {'run_id': 'R', 'attempt': {'attempt_id': 'generation-v1'}, 'status': 'COMPLETED', 'outputs': {'sha256': reader.sha(output.read_bytes())},
                'cost': {'umbrella_lock_sha256': 'lock'}, 'row_checkpoint': {'complete_rows': 1080, 'unrendered_rows': 0}, 'split': {'ids': ids, 'count': 1080, 'identity_hash': 'split'}}
    cfg = {'schema': reader.SCHEMA, 'manifest': pinned(tmp_path / 'manifest.json', manifest),
           'outputs': {'path': str(output), 'sha256': reader.sha(output.read_bytes())},
           'roster': pinned(tmp_path / 'roster.json', {'items': [{'sample_id': i} for i in ids], 'split_identity_hash': 'split'}),
           'inventory': pinned(tmp_path / 'inventory.json', {'pairs': [['x', 'y']], 'entity_types': ['t'], 'lock_sha256': 'lock'}),
           'scores': None, 'expected_rows': 1080, 'out': str(tmp_path / 'out.json')}
    return cfg, rows


def rewrite(cfg, rows):
    path = Path(cfg['outputs']['path'])
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    cfg['outputs']['sha256'] = reader.sha(path.read_bytes())
    p = Path(cfg['manifest']['path'])
    m = json.loads(p.read_text())
    m['outputs']['sha256'] = cfg['outputs']['sha256']
    cfg['manifest'] = pinned(p, m)


def test_contract_partition_and_failures_preserve_denominator(tmp_path):
    cfg, rows = make_config(tmp_path)
    rows[0]['text'] = '{"scenario":"x","action":"y","entities":[{"type":"t","filler":"v"}]}'
    rows[1]['text'] = ''
    rows[1]['native_episode_receipt'] = {'terminal_disposition': {'stage': 'perception', 'reason': 'finite_completion', 'finish_reason': 'length'}}
    rows[2]['text'] = '{"scenario":"x","scenario":"x","action":"y","entities":[]}'
    rows[3]['text'] = '{"scenario":"x","action":"y","entities":[{"type":"illegal","filler":"v"}]}'
    rows[4]['text'] = '{"scenario":"x","action":"illegal","entities":[]}'
    rewrite(cfg, rows)
    result = reader.evaluate(cfg)
    assert result['partitions'] == {'contract_legal_nonempty_entities': 1, 'empty_answer': 1,
        'nonempty_invalid_structure': 1, 'invalid_label_or_entity_contract': 2,
        'contract_legal_empty_entities': 1075}
    assert result['terminal_failures']['P:length'] == 1
    assert sum(result['partitions'].values()) == 1080


@pytest.mark.parametrize('variant', ['duplicate', 'missing'])
def test_duplicate_and_missing_are_refused(tmp_path, variant):
    cfg, rows = make_config(tmp_path)
    if variant == 'duplicate': rows[-1]['sample_id'] = rows[0]['sample_id']
    else: rows.pop()
    rewrite(cfg, rows)
    with pytest.raises(ValueError): reader.evaluate(cfg)


def test_interrupted_recovery_deduplicates_and_cost_is_unknown():
    parent = {'request_id': 'r1', 'agent_contact_id': 'slot:0'}
    child = {'request_id': 'r2', 'agent_contact_id': 'slot:0:slot-length-repair'}
    ep = {'role_visits': [{'stage': 'slot', 'execution_receipt': {'contacts': [parent]}}],
          'interrupted_role': {'stage': 'slot', 'execution_receipt': {'contacts': [parent, child]}}}
    result = reader.recovery_contacts(ep)
    assert len(result) == 1 and result[0]['request_id'] == 'r2'
    assert result[0]['parent_request_id'] == 'r1' and result[0]['cost'] == 'unknown'
    assert result[0]['local_recovered'] == 'unknown'


def test_unidentified_failure_is_unknown():
    assert reader.terminal_category({'terminal_disposition': {'stage': 'slot', 'reason': 'finite_completion'}}) == 'Slot:unknown'


def test_output_pin_mismatch_refused(tmp_path):
    cfg, _ = make_config(tmp_path)
    cfg['outputs']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='SHA256 differs'): reader.evaluate(cfg)


def test_interrupted_slot_candidate_sources_counted_once():
    first = {'request_id': 'r1', 'agent_contact_id': 'slot:0'}
    retry = {'request_id': 'r2', 'agent_contact_id': 'slot:1'}
    ep = {'interrupted_role': {'stage': 'slot', 'execution_receipt': {'contacts': [retry]}},
          'slot_recovery_pending': {'sources': [
              {'source': first, 'valid_ids': ['a']},
              {'source': retry, 'valid_ids': []}]}}
    found = reader.recovery_contacts(ep)
    assert len(found) == 1 and found[0]['request_id'] == 'r2'
    assert found[0]['local_recovered'] is False and found[0]['cost'] == 'unknown'
