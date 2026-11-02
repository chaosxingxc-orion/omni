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
