"""External wire to actor mutation: no native teacher impersonation."""
import json
from pathlib import Path
import sys

import pytest
sys.path.insert(0, str(Path(__file__).parents[2] / 'scripts/analysis'))
import external_exploration as exploration
from gepa_external_mutation import verify_external_mutation
from test_external_exploration import job, write
from test_external_teacher import network
from test_solver_prompt_candidate import fixture


def material(tmp_path, monkeypatch):
    contract, parent, mutation, _ = fixture()
    path, root = job(tmp_path)
    cfg = exploration.read(path)
    source_path = exploration.checked(cfg['source'])
    source = exploration.read(source_path)
    source['document']['groups'][0]['query_id'] = 'slurp:train:1'
    cfg['source'] = write(source_path, source)
    cfg['query_ids'] = ['slurp:train:1']
    cfg['context'] = write(tmp_path / 'context-input.json', {'contract': contract, 'parent': parent})
    cfg['profile'] = exploration.pin(exploration.ROOT / 'configs/learners/tfrl/gepa.json')
    write(path, cfg)
    exploration.prepare(path)
    _, _, caller, request = exploration.load_prepared(root)
    second = dict(mutation, solver_instruction='Check contrary supplied evidence before choosing.')
    response = {'id': 'external-gepa', 'model': 'external-model', 'choices': [
        {'finish_reason': 'stop', 'message': {'content': json.dumps({'algorithm': 'gepa', 'candidates': [mutation, second]})}}],
        'usage': {'prompt_tokens': 100, 'completion_tokens': 30}}
    network(caller, monkeypatch, [response]); caller(request)
    exploration.collect(root)
    spec = {'schema': 'saea-gepa-external-mutation-v1', 'prepared': exploration.pin(root / 'prepared.json'),
            'collected': exploration.pin(root / 'collected.json'), 'candidate_index': 1}
    return spec, contract, parent, root, caller, request


def test_verified_external_candidate_without_native_fit(tmp_path, monkeypatch):
    spec, contract, parent, root, caller, request = material(tmp_path, monkeypatch)
    result = verify_external_mutation(spec, contract=contract, parent=parent)
    assert json.loads(result['mutation_text'])['solver_instruction'] == 'Check contrary supplied evidence before choosing.'
    assert result['usage']['calls'] == 1
    assert result['receipt_kind'] == 'external-chat-v1' and result['native_fit_completed'] is False
    assert result['candidate_accepted'] is False


@pytest.mark.parametrize('fault', ['parent', 'index_bool', 'index_range', 'proposal_tamper', 'wire_tamper'])
def test_external_source_rejections(tmp_path, monkeypatch, fault):
    spec, contract, parent, root, caller, request = material(tmp_path, monkeypatch)
    if fault == 'parent': parent = dict(parent, candidate_sha256='f' * 64)
    elif fault == 'index_bool': spec['candidate_index'] = True
    elif fault == 'index_range': spec['candidate_index'] = 2
    elif fault == 'proposal_tamper':
        d = exploration.read(root / 'collected.json'); d['proposals']['candidates'][1]['solver_instruction'] = 'Not from teacher'
        spec['collected'] = write(root / 'collected.json', d)
    else:
        directory, _ = caller._binding(request)
        (directory / 'response/response.bin').write_text('{}')
    with pytest.raises(Exception): verify_external_mutation(spec, contract=contract, parent=parent)


def test_derived_citations_are_deterministic_and_reject_unknown():
    from gepa_external_mutation import normalize_citations
    groups = [{'query_id': 'slurp:train:000001:audio-42'}, {'query_id': 'slurp:train:000002:audio-43'}]
    source = {'diagnosis': 'See slurp:train:000001.', 'evidence_query_ids': ['slurp:train:000002:audio-43'], 'perception_instruction': 'unchanged'}
    result, proof = normalize_citations(source, groups)
    assert result['diagnosis'] == 'See slurp:train:000001:audio-42.'
    assert result['evidence_query_ids'] == sorted(g['query_id'] for g in groups)
    assert result['perception_instruction'] == source['perception_instruction']
    assert proof['original_candidate_sha256'] != proof['derived_candidate_sha256']
    with pytest.raises(ValueError):
        normalize_citations(dict(source, diagnosis='slurp:train:000003'), groups)
    with pytest.raises(ValueError):
        normalize_citations(source, groups + [{'query_id': 'slurp:train:000001:audio-other'}])


def test_derived_wire_replay_and_tamper(tmp_path, monkeypatch):
    spec, contract, parent, root, caller, request = material(tmp_path, monkeypatch)
    source = {'schema': 'saea-gepa-derived-citation-v1', 'prepared': spec['prepared'],
              'candidate_index': 1, 'request_sha256': exploration.sha(request),
              'response_sha256': caller.reconcile(request)['response']['wire_response_sha256']}
    result = verify_external_mutation(source, contract=contract, parent=parent)
    assert result['receipt_kind'] == 'external-chat-derived-citation-v1'
    with pytest.raises(ValueError):
        verify_external_mutation(dict(source, response_sha256='0' * 64), contract=contract, parent=parent)
    with pytest.raises(ValueError):
        verify_external_mutation(source, contract=contract, parent=dict(parent, candidate_sha256='f' * 64))
    directory, _ = caller._binding(request)
    (directory / 'response/response.bin').write_text('{}')
    with pytest.raises(Exception):
        verify_external_mutation(source, contract=contract, parent=parent)
