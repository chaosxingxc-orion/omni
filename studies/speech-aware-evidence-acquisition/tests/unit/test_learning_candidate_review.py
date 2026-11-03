"""The exporter must keep human/model judgments pending and native checks mandatory."""
import copy
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / 'scripts/analysis'))
import learning_candidate_review as m
from test_learning_candidate_release import reviewed, put
from test_tfgrpo_consumption import release


@pytest.fixture
def packet(reviewed, monkeypatch):
    repo, data, params, candidate, rel, review, document = reviewed
    candidate['updates'] = []
    candidate.pop('sha256'); candidate['sha256'] = m.sha(candidate)
    params['candidate'] = put(data, 'candidate.json', candidate)
    monkeypatch.setattr(m.consumption, 'verify_learning_seal', lambda *a, **kw: object())
    loaded = {'document': document, 'known_usage': candidate['learning_usage'],
              'historical_usage': {'calls': 2, 'prompt_tokens': 20, 'completion_tokens': 10}}
    document['parent'] = {'items': {}}
    calls = []
    def verify(**kwargs):
        calls.append(kwargs)
        return loaded
    monkeypatch.setattr(m.release, 'verify_ace_source', verify)
    monkeypatch.setattr(m.release, 'verify_reflexion_source', verify)
    return repo, data, candidate, document, calls


@pytest.mark.parametrize('strategy', [m.release.STRATEGY, m.release.REFLEXION_STRATEGY])
def test_export_pending_all_items_conflicts_evidence_and_costs(packet, strategy):
    repo, data, candidate, document, calls = packet
    candidate['strategy'] = strategy
    candidate['unresolved_conflicts'] = [{'query_id': 'slurp:train:000001:real', 'reason': 'unresolved'}]
    candidate.pop('sha256'); candidate['sha256'] = m.sha(candidate)
    put(data, 'candidate.json', candidate)
    before = (data / 'candidate.json').read_bytes()
    out = data / 'work/codex/review'
    receipt = m.export_review(repo, data, candidate_path=data/'candidate.json',
                              manifest_path=data/'manifest.json', out=out)
    assert len(calls) == 1 and calls[0]['candidate'] == candidate
    assert receipt['new_model_calls'] == 0 and not receipt['candidate_deployment_accepted']
    assert (data / 'candidate.json').read_bytes() == before
    review = json.loads((out/'review.pending.json').read_bytes())
    assert review['decision'] == 'pending' and set(review['items']) == set(candidate['items'])
    assert all(i['decision'] == 'pending' and i['evidence'] == [] for i in review['items'].values())
    assert set(review['conflicts']) == {m.sha(candidate['unresolved_conflicts'][0])}
    manifest = json.loads((out/'release.pending.json').read_bytes())
    assert manifest['review'] is None
    with pytest.raises(ValueError, match='independent development review'):
        m.release.verify_review(review, release=manifest, candidate=candidate, document=document)
    evidence = json.loads((out/'evidence.json').read_bytes())
    assert evidence['groups'] == document['groups'] and evidence['items'] == candidate['items']
    assert evidence['cumulative_known_usage'] == candidate['cumulative_known_usage']
    assert not evidence['semantic_support_verified']
    with pytest.raises(ValueError, match='fresh'):
        m.export_review(repo, data, candidate_path=data/'candidate.json',
                        manifest_path=data/'manifest.json', out=out)


@pytest.mark.parametrize('failure', ['seal', 'source', 'mutation', 'output'])
def test_export_rejects_before_publishing_on_custody_failure(packet, monkeypatch, failure):
    repo, data, candidate, document, calls = packet
    out = data/'work/codex/review'
    def reject(*a, **kw):
        raise ValueError('native failure')
    if failure == 'seal': monkeypatch.setattr(m.consumption, 'verify_learning_seal', reject)
    elif failure == 'source': monkeypatch.setattr(m.release, 'verify_ace_source', reject)
    elif failure == 'mutation':
        original = m.release.verify_ace_source
        def mutate(**kw):
            result = original(**kw); (data/'candidate.json').write_text('{}'); return result
        monkeypatch.setattr(m.release, 'verify_ace_source', mutate)
    else: out = data/'outside'
    with pytest.raises(ValueError):
        m.export_review(repo, data, candidate_path=data/'candidate.json',
                        manifest_path=data/'manifest.json', out=out)
    assert not out.exists()


def annotated_packet(packet, operation='always_drop'):
    from speech_aware_evidence_acquisition.rails.slot_rule_review import SCHEMA
    from speech_aware_evidence_acquisition.rails.slot_rule_admission import CANDIDATE_SCHEMA
    repo, data, candidate, document, calls = packet
    group = document['groups'][0]
    group['evidence'] = {'reference': {'scenario': 'news', 'action': 'query',
        'entities': [{'type': 'news_topic', 'filler': 'topic'}]}}
    key = next(iter(candidate['items']))
    annotations = {'schema': SCHEMA, 'candidate': m.pin(data, data/'candidate.json'),
        'document_sha256': m.sha(document), 'items': {key: {
            'content_sha256': m.sha(candidate['items'][key]), 'rule': {
                'schema': CANDIDATE_SCHEMA, 'intent_scope': [{'scenario': 'news', 'action': 'query'}],
                'condition': None, 'operation': operation, 'slot': 'news_topic',
                'support_refs': [group['query_id']]}}}}
    put(data, 'rules.json', annotations)
    return annotations, key


@pytest.mark.parametrize('operation,status', [('always_drop', 'rejected'),
                                              ('always_require', 'static_consistency_only')])
def test_optional_static_annotations_never_change_release_or_candidate(packet, operation, status):
    repo, data, candidate, document, calls = packet
    annotations, key = annotated_packet(packet, operation)
    original = (data/'candidate.json').read_bytes()
    out = data/'work/codex/static-review'
    receipt = m.export_review(repo, data, candidate_path=data/'candidate.json',
        manifest_path=data/'manifest.json', out=out, rules_path=data/'rules.json')
    assert len(calls) == 1
    static = json.loads((out/'slot-rules.static.json').read_bytes())
    assert static['items'][key]['static_result']['status'] == status
    assert static['candidate'] == annotations['candidate']
    assert static['document_sha256'] == m.sha(document)
    assert static['evidence_pool']['rows'][0]['source_sha256'] == m.sha(document['groups'][0])
    assert not static['items'][key]['semantic_translation_verified']
    assert not static['semantic_support_verified'] and not static['candidate_deployment_accepted']
    assert (data/'candidate.json').read_bytes() == original
    assert 'slot-rules.static.json' in receipt['artifacts']
    review = json.loads((out/'review.pending.json').read_bytes())
    manifest = json.loads((out/'release.pending.json').read_bytes())
    assert review['decision'] == 'pending' and manifest['review'] is None
    with pytest.raises(ValueError, match='independent development review'):
        m.release.verify_review(review, release=manifest, candidate=candidate, document=document)


@pytest.mark.parametrize('fault', ['candidate', 'document', 'content', 'item', 'test', 'foreign_ref', 'prose'])
def test_static_annotations_cannot_substitute_items_or_train_evidence(packet, fault):
    repo, data, candidate, document, calls = packet
    annotations, key = annotated_packet(packet)
    if fault == 'candidate': annotations['candidate']['sha256'] = '0'*64
    elif fault == 'document': annotations['document_sha256'] = '0'*64
    elif fault == 'content': annotations['items'][key]['content_sha256'] = '0'*64
    elif fault == 'item': annotations['items']['foreign'] = annotations['items'].pop(key)
    elif fault == 'test':
        document['groups'][0]['query_id'] = 'slurp:test:0'
        annotations['document_sha256'] = m.sha(document)
    elif fault == 'foreign_ref': annotations['items'][key]['rule']['support_refs'] = ['slurp:train:foreign']
    else: annotations['items'][key]['rule'] = 'Always drop news_topic'
    put(data, 'rules.json', annotations)
    out = data/'work/codex/refused'
    with pytest.raises(ValueError):
        m.export_review(repo, data, candidate_path=data/'candidate.json',
            manifest_path=data/'manifest.json', out=out, rules_path=data/'rules.json')
    assert not out.exists()


def test_static_annotations_do_not_bypass_native_verifier(packet, monkeypatch):
    repo, data, candidate, document, calls = packet
    annotated_packet(packet)
    def refuse(**kwargs): raise ValueError('native source mismatch')
    monkeypatch.setattr(m.release, 'verify_ace_source', refuse)
    out = data/'work/codex/refused'
    with pytest.raises(ValueError, match='native source'):
        m.export_review(repo, data, candidate_path=data/'candidate.json',
            manifest_path=data/'manifest.json', out=out, rules_path=data/'rules.json')
    assert not out.exists()
