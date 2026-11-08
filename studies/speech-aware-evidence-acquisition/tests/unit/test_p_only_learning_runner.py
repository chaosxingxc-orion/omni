import json
from dataclasses import replace
from pathlib import Path

import pytest

from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
from speech_aware_evidence_acquisition.rails.intent_slot_pipeline import canonical
from speech_aware_evidence_acquisition.rails.p_only_learning_scope import PerceptionLearningScope
from test_intent_slot_runner import recovery_harness
from test_native_episode_runner import _wire_response
from test_llama_server_transport import _spawn_live_matching_process  # noqa: F401
from test_rails_runner import _knowledge_snapshot


def overlapping_snapshot(sample_id):
    snapshot = _knowledge_snapshot()
    value = replace(snapshot.values[0], source_sample_ids=('train:t1', sample_id))
    return replace(snapshot, values=(value,), source_sample_ids=('train:t1', sample_id)).validate()


@pytest.mark.parametrize('mode', ['tf-collection', 'candidate-evaluation'])
@pytest.mark.parametrize('bad', [None, 'reference', 'exemplar', 'visible', 'short'])
def test_real_runner_learning_scope_and_repair(tmp_path, mode, bad):
    h, policy = recovery_harness(tmp_path)
    h.repo_root = Path(__file__).resolve().parents[2]
    policy.update(answer_only_fallback='off', perception_length_recovery='compact-reask-v1')
    h.config = replace(h.config, intent_slot_pipeline_json=canonical(policy), perception_only=True)
    snapshot = overlapping_snapshot(h.samples[0].sample_id)
    h.p_only_learning_scope = PerceptionLearningScope(frozenset([h.samples[0].sample_id]),
        {h.samples[0].sample_id: 'weather' if bad == 'short' else 'secret teacher reference sentence for this example'}, snapshot.content_hash(), snapshot, ())
    def supply(phase, context):
        if phase != 'perception':
            raise AssertionError('P-only requested downstream supply')
        return None
    h.supply_for_phase = supply
    if bad == 'short':
        original_material = h.material_for_phase
        def short_material(phase, context, sealed):
            material = original_material(phase, context, sealed)
            return replace(material, instruction=material.instruction + ' weather')
        h.material_for_phase = short_material
    calls = []
    def post(_url, body):
        calls.append(json.loads(body))
        if len(calls) == 1:
            response = json.loads(_wire_response('{"candidates":['))
            response['choices'][0]['finish_reason'] = 'length'
            return json.dumps(response).encode()
        return _wire_response(h._output(('s1', 'perception', 1, 1)))
    h.post_override = post
    if bad == 'visible':
        original = h.p_only_learning_scope.assert_visible
        def check_visible(query_state):
            original(replace(query_state, retrieved_value_ids=('train-exemplars::answer',)))
        object.__setattr__(h.p_only_learning_scope, 'assert_visible', check_visible)
    elif bad and bad != 'short':
        original = h.p_only_learning_scope.assert_request
        def check(prepared, sample_id):
            original(prepared, sample_id)
            if len(calls) == 1:
                body = prepared.rendered_body()
                body['messages'].append({'role':'user', 'content': 'secret teacher reference sentence for this example' if bad == 'reference' else 'train-exemplars'})
                from types import SimpleNamespace
                original(SimpleNamespace(role_request=prepared.role_request, rendered_body=lambda: body), sample_id)
        object.__setattr__(h.p_only_learning_scope, 'assert_request', check)
    try:
        if bad and bad != 'short':
            with pytest.raises(Exception, match='teacher reference|exemplar|visible source'):
                h.run(snapshot=snapshot, decoding_params={'temperature': .7 if mode == 'tf-collection' else 0, 'max_tokens':64})
        else:
            h.run(snapshot=snapshot, decoding_params={'temperature': .7 if mode == 'tf-collection' else 0, 'max_tokens':64})
    finally:
        h.close()
    assert len(calls) == (0 if bad == 'visible' else 1 if bad and bad != 'short' else 2)


def test_default_evaluation_still_rejects_composite_overlap(tmp_path):
    h, _ = recovery_harness(tmp_path)
    snapshot = overlapping_snapshot(h.samples[0].sample_id)
    try:
        with pytest.raises(Exception, match='overlap'):
            h.run(snapshot=snapshot)
    finally:
        h.close()
