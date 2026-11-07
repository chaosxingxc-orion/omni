import copy
import json
from pathlib import Path

import pytest

from speech_aware_evidence_acquisition.rails import tfgrpo_external_reference as ref
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import empty_experience, sha
from speech_aware_evidence_acquisition.rails.grounded_synthesis_fusion import project_group_evidence
from test_tfgrpo_external_reference import material, write, responses
from test_external_teacher import network


def setup_p(tmp_path):
    path, cfg = material(tmp_path)
    batch = ref.read(cfg['source_batch']['path'])
    group = batch['groups'][0]
    group['answer'] = json.dumps({'gold_intent': 'alarm_set'})
    for i, row in enumerate(group['rollouts']):
        row['reward'] = i
        row['critique'] = json.dumps({'recall3': bool(i), 'legal': True, 'top1': bool(i)})
        trajectory = json.loads(row['trajectory'])
        trajectory['scored_answer_after_fixed_normalization'] = trajectory['response_text']
        row['trajectory'] = json.dumps(trajectory)
    batch['common']['reward_mode'] = 'binary-task-exact-v1'
    cfg['source_batch'] = write(Path(cfg['source_batch']['path']), batch)
    source = ref.read(cfg['source']['path'])
    doc = source['document']
    doc.update(source_batch_sha256=sha(batch), reward_mode='binary-task-exact-v1')
    doc['groups'][0].update(group_sha256=sha(group), evidence=project_group_evidence(group, evidence_view='full-query-v1'))
    cfg['source'] = write(Path(cfg['source']['path']), source)
    qid = group['query_id']
    private = {'provenance': {'original_source_sha256': 'a'*64}, 'document': {'groups': [{
        'query_id': qid, 'evidence': {'query': {'sample_id': qid}, 'audio_available_to_reflector': False,
        'reference_feedback_private': {'reference_sentence': 'set an alarm', 'intent': 'alarm_set',
                                       'actor_input_contains_gold_feedback': False}}}]}}
    cfg['teacher_reference'] = write(tmp_path/'reference.json', private)
    cfg['p_only_binary'] = True
    cfg['teacher']['research_prompt'] = ref.pin(ref.ROOT/'configs/prompts/learner/tfrl/tfgrpo-reference-p-only.md')
    write(path, cfg)
    return path, cfg, batch, private


@pytest.mark.parametrize('fault', ['hash', 'missing', 'foreign', 'duplicate', 'sample', 'label'])
def test_reference_rejected(tmp_path, fault):
    path, cfg, batch, private = setup_p(tmp_path)
    rows = private['document']['groups']
    if fault == 'hash':
        Path(cfg['teacher_reference']['path']).write_text('{}')
    else:
        if fault == 'missing': rows.clear()
        elif fault == 'foreign': rows[0]['query_id'] = rows[0]['evidence']['query']['sample_id'] = 'slurp:train:foreign:audio'
        elif fault == 'duplicate': rows.append(copy.deepcopy(rows[0]))
        elif fault == 'sample': rows[0]['evidence']['query']['sample_id'] = 'slurp:train:foreign:audio'
        elif fault == 'label': rows[0]['evidence']['reference_feedback_private']['intent'] = 'foreign'
        cfg['teacher_reference'] = write(Path(cfg['teacher_reference']['path']), private)
        write(path, cfg)
    with pytest.raises(ValueError): ref.prepare(path)
    assert not Path(cfg['output_root']).exists()


def test_configured_reference_prompt_is_pinned_and_prepared(tmp_path):
    path, cfg, _, _ = setup_p(tmp_path)
    default = ref.ROOT/'configs/prompts/learner/tfrl/tfgrpo-reference-p-only.md'
    prompt = tmp_path/'external-policy.md'
    prompt.write_bytes(default.read_bytes() + b'\nApply the configured research instructions.\n')
    cfg['teacher']['research_prompt'] = ref.pin(prompt)
    write(path, cfg)
    ref.prepare(path)
    assert (Path(cfg['output_root'])/'policy.md').read_bytes() == prompt.read_bytes()

    prompt.write_bytes(prompt.read_bytes() + b'Changed after pinning.\n')
    with pytest.raises(ValueError):
        ref.verified_material(cfg)


def test_teacher_only_preserves_original_group_and_request(tmp_path, monkeypatch):
    path, cfg, batch, private = setup_p(tmp_path)
    prepared = ref.prepare(path)
    assert prepared['plan']['rollouts'] == 2
    assert ref.read(Path(cfg['output_root'])/'groups.json') == batch['groups']
    caller = ref.load(cfg['output_root'])[-1]
    calls = network(caller, monkeypatch, responses())
    ref.execute(cfg['output_root'])
    assert len(calls) == 5
    contacts = [ref.read(p) for p in (Path(cfg['output_root'])/'learner/contacts').glob('*.intent.json')]
    for request in contacts:
        text = json.dumps(request['messages'])
        if request['stage'] in ('summary', 'advantage'):
            assert 'set an alarm' in text and 'alarm_set' in text
            assert 'independent listening' in text
        else:
            assert 'set an alarm' not in text
    assert 'set an alarm' not in json.dumps(batch)
