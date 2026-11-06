"""Protocol v2 contracts and source replay, with no model or service."""
import copy
import hashlib
import json

import jsonschema
import pytest

from speech_aware_evidence_acquisition.rails.adapters.slurp_intent_slot import classify_slot_coverage, validate_response
from speech_aware_evidence_acquisition.rails.intent_slot_pipeline import (
    SCHEMA_V2, candidates, canonical, derive_answer, digest, dynamic_decision_format,
    dynamic_slot_format, slot_results, validate_evidence_refs, validate_receipt,
)

CFG = {'schema': SCHEMA_V2, 'slot_mode': 'batched', 'max_candidates': 3,
       'candidate_outside_recovery': False, 'duplicate_candidate_policy': 'group_v2'}
PFORMAT = {'json_schema': {'schema': {'type': 'object', 'required': ['candidates'],
    'properties': {'candidates': {'type': 'array', 'items': {'type': 'object',
    'required': ['intent_id', 'distinction', 'evidence_refs', 'unresolved_fields'],
    'properties': {'intent_id': {'enum': ['play_music', 'play_radio']},
    'distinction': {'type': 'string'}, 'evidence_refs': {'type': 'array', 'items': {'type': 'string'}},
    'unresolved_fields': {'type': 'array', 'items': {'type': 'string'}}}}}}}}}
SFORMAT = {'json_schema': {'schema': {'type': 'object', 'required': ['candidates'],
    'properties': {'candidates': {'type': 'array', 'items': {'type': 'object',
    'required': ['candidate_id', 'entities', 'evidence_refs', 'unresolved_fields'],
    'properties': {'candidate_id': {'enum': ['play_music', 'play_radio']},
    'entities': {'type': 'array', 'items': {'type': 'object', 'properties': {
    'type': {'enum': ['person']}, 'filler': {'type': 'string'}}}},
    'evidence_refs': {'type': 'array'}, 'unresolved_fields': {'type': 'array'}}}}}}}}


def row(cid, distinction, refs=()):
    return {'intent_id': cid, 'distinction': distinction, 'evidence_refs': list(refs), 'unresolved_fields': [distinction]}


def inventory(ids=()):
    return {'sample_id': 'sample', 'candidate_ids': list(ids), 'sources': [{'source_id': 'audio', 'kind': 'original-audio'}]}


def test_group_preserves_all_sources_and_rejects_invalid_input():
    raw = [row('play_music', 'first', ['audio']), row('play_music', 'different', ['audio']), row('play_radio', 'other')]
    ctx = {'evidence_inventory': inventory()}
    grouped = validate_response(kind='perception', text=canonical({'candidates': raw}), response_format=PFORMAT, context=ctx, config=CFG)
    assert [r['candidate_id'] for r in grouped] == ['play_music', 'play_radio']
    assert grouped[0]['sources'] == raw[:2]
    invalid = copy.deepcopy(raw); invalid[1]['evidence_refs'] = ['invented']
    with pytest.raises(ValueError, match='reference'):
        validate_response(kind='perception', text=canonical({'candidates': invalid}), response_format=PFORMAT, context=ctx, config=CFG)
    invalid[1]['intent_id'] = 'outside'
    with pytest.raises(ValueError):
        validate_response(kind='perception', text=canonical({'candidates': invalid}), response_format=PFORMAT, context=ctx, config=CFG)
    with pytest.raises(ValueError, match='conflicting'):
        candidates(canonical({'candidates': raw[:2]}), 3, duplicate_policy='identical_v1')


def test_dynamic_slot_coverage_and_decision_schema():
    ids = ['play_music', 'play_radio']
    fmt = dynamic_slot_format(SFORMAT, ids)
    array = fmt['json_schema']['schema']['properties']['candidates']
    assert array['minItems'] == array['maxItems'] == 2
    assert array['items']['properties']['candidate_id']['enum'] == ids
    a = {'candidate_id': ids[0], 'entities': [], 'evidence_refs': [], 'unresolved_fields': []}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({'candidates': [a]}, fmt['json_schema']['schema'])
    with pytest.raises(ValueError):
        slot_results(canonical({'candidates': [a, a]}), ids)
    assert dynamic_decision_format(ids)['json_schema']['schema']['properties']['candidate_id']['enum'] == ids


def test_partial_slot_classification_preserves_valid_source_and_refuses_ambiguity():
    ids = ['play_music', 'play_radio']
    fmt = dynamic_slot_format(SFORMAT, ids)
    context = {'candidates': [{'candidate_id': cid} for cid in ids],
               'evidence_inventory': inventory(ids)}
    valid = {'candidate_id': ids[0], 'entities': [{'type': 'person', 'filler': 'Alice'}],
             'evidence_refs': ['audio'], 'unresolved_fields': []}
    broken = {'candidate_id': ids[1], 'entities': [{'type': 'person', 'filler': ''}],
              'evidence_refs': [], 'unresolved_fields': []}
    report = classify_slot_coverage(text=canonical({'candidates': [valid, broken]}),
                                    response_format=fmt, context=context, config=CFG)
    assert report['valid'] == {ids[0]: valid}
    assert report['missing'] == [ids[1]] and ids[1] in report['invalid']
    broken['entities'][0]['filler'] = 'Bob'
    broken['evidence_refs'] = ['unprovided']
    assert classify_slot_coverage(text=canonical({'candidates': [valid, broken]}),
        response_format=fmt, context=context, config=CFG)['missing'] == [ids[1]]
    for rows in ([valid, valid], [valid, dict(broken, candidate_id='foreign')]):
        with pytest.raises(ValueError):
            classify_slot_coverage(text=canonical({'candidates': rows}),
                                   response_format=fmt, context=context, config=CFG)


def test_raw_three_node_replay_and_tamper():
    raw = [row('play_music', 'first', ['audio']), row('play_music', 'different', ['audio']), row('play_radio', 'other')]
    rows = candidates(canonical({'candidates': raw}), 3, duplicate_policy='group_v2')
    ids = [r['candidate_id'] for r in rows]
    slots = [{'candidate_id': cid, 'entities': [{'type': 'person', 'filler': cid}],
              'evidence_refs': ['audio'], 'unresolved_fields': []} for cid in ids]
    texts = [canonical({'candidates': raw}), canonical({'candidates': slots}), canonical({'candidate_id': ids[1]})]
    records, sources = {}, []
    for i, text in enumerate(texts):
        rid = f'r{i}'; source = {'request_id': rid, 'request_sha256': f'request{i}',
            'response_sha256': f'envelope{i}', 'response_text_sha256': hashlib.sha256(text.encode()).hexdigest()}
        sources.append(source)
        records[rid + '/request'] = {'payload_sha256': source['request_sha256']}
        records[rid + '/response'] = {'text': text, 'response_sha256': source['response_sha256'],
                                      'response_text_sha256': source['response_text_sha256']}
    parsed = slot_results(texts[1], ids)
    answer = derive_answer(ids[1], parsed)
    block = {'schema': SCHEMA_V2, 'slot_mode': 'batched', 'candidate_ids': ids,
             'duplicate_candidate_policy': 'group_v2', 'perception_source': sources[0],
             'slot_sources': [{'candidate_ids': ids, 'source': sources[1], 'result_sha256': digest(parsed)}],
             'decision_source': sources[2], 'selected_candidate_id': ids[1],
             'derived_answer_sha256': hashlib.sha256(answer.encode()).hexdigest(),
             'grouping': {'policy': 'group_v2', 'raw_count': 3,
                 'groups': [{'candidate_id': r['candidate_id'], 'sources': r['sources']} for r in rows]},
             'evidence_bindings': [{'stage': stage, 'inventory': inv,
                 'mappings': validate_evidence_refs(source_rows, inv, expected),
                 'response_text_sha256': sources[i]['response_text_sha256']}
                 for i, (stage, inv, source_rows, expected) in enumerate([
                     ('perception', inventory(), raw, ids), ('slot', inventory(ids), slots, ids)])]}
    receipt = {'intent_slot_pipeline': block, 'accepted_solver_receipt': {}, 'role_visits': [
        {'stage': stage, 'execution_receipt': {'final_contact': src}}
        for stage, src in zip(('perception', 'slot', 'solver'), sources)]}
    assert validate_receipt(receipt, trace_records=records) == answer
    assert json.loads(answer)['entities'] == slots[1]['entities']
    for change in ('slot', 'decision', 'group', 'answer'):
        bad, trace = copy.deepcopy(receipt), copy.deepcopy(records)
        if change == 'slot': trace['r1/response']['text'] += ' '
        if change == 'decision': trace['r2/response']['text'] = canonical({'candidate_id': ids[0]})
        if change == 'group': bad['intent_slot_pipeline']['grouping']['groups'][0]['sources'].pop()
        if change == 'answer': bad['intent_slot_pipeline']['derived_answer_sha256'] = '0' * 64
        with pytest.raises(ValueError): validate_receipt(bad, trace_records=trace)


@pytest.mark.parametrize('mode', ['batched', 'per_candidate'])
def test_v2_preparation_and_recovery_admission(mode):
    import sys
    from pathlib import Path
    from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
    from speech_aware_evidence_acquisition.rails.phase_material import load_phase_material_specs
    from speech_aware_evidence_acquisition.core.flight.native_recovery import native_contact_custody_enabled
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'scripts/analysis'))
    from derive_lock_only_baseline import derive as lock
    from prepare_intent_slot_candidate import derive
    source = json.loads((root / 'configs/baseline/slurp-native-stage3-l338-lever-column-fold-b-baseline-raw-port-v1.json').read_text())
    baseline = lock(source)
    assert baseline['native_episode']['deliver_normaliser']['normaliser'] == 'saea-deliver-normalise-v4g'
    prepared = derive(baseline, mode, 'v2')
    assert prepared['native_episode']['deliver_normaliser'] == baseline['native_episode']['deliver_normaliser']
    program = default_phase_program(prepared['policy'])
    assert len(load_phase_material_specs(prepared, program)) == 5
    assert prepared['native_episode']['intent_slot_pipeline']['schema'] == SCHEMA_V2
    assert prepared['task_instruction_template'].endswith('context_v2.txt')
    prepared.update(transport='vllm', arm='raw-speech-agent-native-tools', phase_tools={})
    assert native_contact_custody_enabled(prepared)


def test_coverage_recovery_preparation_has_separate_identity_and_budget():
    import sys
    from pathlib import Path
    from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
    from speech_aware_evidence_acquisition.rails.native_episode_config import load_native_episode_config
    from speech_aware_evidence_acquisition.core.flight.native_recovery import native_contact_custody_enabled
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'scripts/analysis'))
    from derive_lock_only_baseline import derive as lock
    from prepare_intent_slot_candidate import derive
    source = json.loads((root / 'configs/baseline/slurp-native-stage3-l338-lever-column-fold-b-baseline-raw-port-v1.json').read_text())
    prepared = derive(lock(source), 'batched', 'v2', True, True)
    program = default_phase_program(prepared['policy'])
    assert program.max_contacts == 6
    assert load_native_episode_config(prepared, program).max_contacts == 6
    assert prepared['native_episode']['intent_slot_pipeline']['answer_only_fallback'] == 'off'
    prepared.update(transport='vllm', arm='raw-speech-agent-native-tools', phase_tools={})
    assert native_contact_custody_enabled(prepared)


def test_node_rails_requires_explicit_fallback_policy():
    from speech_aware_evidence_acquisition.rails.intent_slot_pipeline import parse_config
    config = {'schema':'saea-intent-slot-pipeline-v2','slot_mode':'batched','max_candidates':3,
              'candidate_outside_recovery':False,'duplicate_candidate_policy':'group_v2','node_rails':'sdk-handoff-v1'}
    with pytest.raises(ValueError, match='explicit answer_only_fallback'): parse_config(config)
    for choice in ('off', 'budgeted-v1'):
        assert parse_config(dict(config, answer_only_fallback=choice))['answer_only_fallback'] == choice
