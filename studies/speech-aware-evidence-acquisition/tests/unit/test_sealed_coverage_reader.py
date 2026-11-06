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
    result, uncertain = reader.recovery_contacts(ep)
    assert uncertain == []
    assert len(result) == 1 and result[0]['request_id'] == 'r2'
    assert result[0]['parent_request_id'] == 'r1' and result[0]['cost'] == 'unknown'
    assert result[0]['stage_recovered'] == 'unknown'


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
    found, uncertain = reader.recovery_contacts(ep)
    assert uncertain == []
    assert len(found) == 1 and found[0]['request_id'] == 'r2'
    assert found[0]['local_candidate_recovered'] is False and found[0]['stage_recovered'] == 'unknown' and found[0]['cost'] == 'unknown'


def test_scores_input_rejected(tmp_path):
    cfg, _ = make_config(tmp_path)
    cfg['scores'] = pinned(tmp_path / 'unrelated-score.json', {'accuracy': 1})
    with pytest.raises(ValueError, match='scores not supported'):
        reader.evaluate(cfg)


def test_completed_length_child_stage_is_accepted_but_not_correctness():
    for stage, suffix in [('perception', ':perception-length-repair'), ('slot', ':slot-length-repair')]:
        parent = {'request_id': 'p', 'agent_contact_id': stage + ':0'}
        child = {'request_id': 'c', 'agent_contact_id': stage + ':0' + suffix}
        ep = {'role_visits': [{'stage': stage, 'execution_receipt': {
            'contacts': [parent, child], 'final_contact': child}}]}
        found, unknown = reader.recovery_contacts(ep)
        assert not unknown and len(found) == 1
        assert found[0]['stage_recovered'] is True
        assert found[0]['local_candidate_recovered'] == 'unknown'


def test_slot_candidate_recovered_does_not_mean_stage_recovered():
    first = {'request_id': 'r1', 'agent_contact_id': 'slot:0'}
    retry = {'request_id': 'r2', 'agent_contact_id': 'slot:1'}
    ep = {'interrupted_role': {'stage': 'slot', 'execution_receipt': {'contacts': [retry]}},
          'slot_recovery_pending': {'sources': [{'source': first, 'valid_ids': ['a']},
                                                {'source': retry, 'valid_ids': ['b']}],
                                    'remaining_candidate_ids': ['c']}}
    found, unknown = reader.recovery_contacts(ep)
    assert found[0]['local_candidate_recovered'] is True
    assert found[0]['stage_recovered'] == 'unknown' and not unknown


def test_unattributed_interrupted_contacts_are_retained_without_counting_as_retries():
    ep = {'interrupted_role': {'stage': 'slot', 'execution_receipt': {'contacts': [
        {'request_id': 'r1', 'agent_contact_id': 'slot:0'},
        {'request_id': 'r2', 'agent_contact_id': 'slot:1'}]}}}
    found, unknown = reader.recovery_contacts(ep)
    assert not found and {v['request_id'] for v in unknown} == {'r1', 'r2'}
    assert all(v['attribution'] == 'unknown' for v in unknown)


def test_explicit_slot_sources_candidate_recovery():
    first = {'request_id': 'r1', 'agent_contact_id': 'slot:0'}
    retry = {'request_id': 'r2', 'agent_contact_id': 'slot:1'}
    ep = {'role_visits': [{'stage': 'slot', 'execution_receipt': {'contacts': [first, retry], 'final_contact': retry}}],
          'intent_slot_pipeline': {'coverage_recovery': {'max_extra_contacts': 3},
                                   'slot_sources': [{'source': first, 'valid_ids': ['a']},
                                                    {'source': retry, 'valid_ids': ['b']}]}}
    found, _ = reader.recovery_contacts(ep)
    assert found[0]['local_candidate_recovered'] is True
    assert found[0]['stage_recovered'] == 'unknown'


def test_nonempty_filler_and_valid_empty_entities_are_distinct():
    pairs, types = {('x', 'y')}, {'t'}
    assert reader.classify('{"scenario":"x","action":"y","entities":[]}', pairs, types) == 'contract_legal_empty_entities'
    assert reader.classify('{"scenario":"x","action":"y","entities":[{"type":"t","filler":"v"}]}', pairs, types) == 'contract_legal_nonempty_entities'
    assert reader.classify('{"scenario":"x","action":"y","entities":[{"type":"t","filler":""}]}', pairs, types) == 'nonempty_invalid_structure'

@pytest.mark.parametrize('fault', [None, 'open', 'partial', 'roster', 'checkpoint'])
def test_successor_authentication_and_journal_source(tmp_path, monkeypatch, fault):
    from types import SimpleNamespace
    from speech_aware_evidence_acquisition.core import bundle as bundle_module
    cfg, rows = make_config(tmp_path)
    cfg['schema'] = 'saea-sealed-coverage-successor-v1'
    cfg['lock'] = pinned(tmp_path / 'lock.json', {'datasets': []})
    inventory = json.loads(Path(cfg['inventory']['path']).read_text())
    inventory['lock_sha256'] = cfg['lock']['sha256']
    cfg['inventory'] = pinned(Path(cfg['inventory']['path']), inventory)
    p = Path(cfg['manifest']['path']); manifest = json.loads(p.read_text())
    ids = manifest['split']['ids']; manifest.pop('row_checkpoint')
    manifest['predecessor_attempt'] = {'attempt_id':'original', 'run_id':'R', 'mode':'checkpoint',
        'counts':{'kept':1080,'flown':0,'partial':0}, 'kept_sample_ids':ids}
    cfg['manifest'] = pinned(p, manifest)
    fake = SimpleNamespace(attempt_state='FINALIZED', manifest=manifest,
        outputs_path=Path(cfg['outputs']['path']), outputs_rows=rows, split_ids=set(ids))
    if fault == 'open': fake.attempt_state = 'OPENED'
    if fault == 'partial': manifest['predecessor_attempt']['counts']['partial'] = 1
    if fault == 'roster': manifest['predecessor_attempt']['kept_sample_ids'] = ids[:-1]
    if fault in ('partial','roster'): cfg['manifest'] = pinned(p, manifest)
    def verify(path):
        if fault == 'checkpoint': raise ValueError('checkpoint hash mismatch')
        return fake
    monkeypatch.setattr(bundle_module, 'verify_bundle', verify)
    from speech_aware_evidence_acquisition.core.bundle import git_commit
    monkeypatch.setattr(git_commit, 'require_clean_load_bearing_surface', lambda path: ('test-commit', None))
    manifest['study_commit'] = 'test-commit'
    cfg['manifest'] = pinned(p, manifest)
    cfg['lock']['path'] = '/opt/saea-release/docs/datasets.lock.json'
    original_read_pin = reader.read_pin
    monkeypatch.setattr(reader, 'read_pin', lambda pin: (json.dumps({'datasets': []}) + '\n').encode() if pin is cfg['lock'] else original_read_pin(pin))
    if fault:
        with pytest.raises(ValueError): reader.evaluate(cfg)
    else:
        result = reader.evaluate(cfg)
        assert result['total_rows'] == 1080 and result['journal_source_attempt'] == 'original'
        assert result['partitions']['contract_legal_empty_entities'] == 1080
