import copy

import pytest

from test_psd_construction_campaign import config
from psd_construction_campaign import compile_campaign, check_compiled
from p_only_workload import PerceptionOnlyWorkload, verify_vector_selection
import hashlib
import json


def learning_config():
    campaign = config()
    job = campaign['workload']
    job['schema'] = 'saea-perception-only-workload-v1'
    job['sample_ids'] = job['sample_ids'][:80]
    job.pop('node_arm')
    job.pop('pipeline')
    job.update(parent_baseline={}, parent_registration={}, axis={'kind': 'control'},
               learning_authorization={'relpath': 'feedback.json', 'sha256': 'a' * 64})
    campaign['collection']['seeds'] = [101, 102, 103, 104]
    return campaign


def test_four_independent_p_only_jobs_keep_common_science():
    c = learning_config()
    doc = compile_campaign(c)
    check_compiled(doc)
    assert doc['budget']['normal_generation_calls'] == 320
    assert len({m['fit_run_id'] for m in doc['members']}) == 4
    jobs = [m['job_config'] for m in doc['members']]
    assert len({j['run_id'] for j in jobs}) == len({j['name'] for j in jobs}) == 4
    assert [j['collection_decoding']['seed'] for j in jobs] == [101, 102, 103, 104]
    normalized = [{k: v for k, v in j.items() if k not in ('name', 'run_id', 'collection_decoding')} for j in jobs]
    assert all(j == normalized[0] for j in normalized)
    assert all(PerceptionOnlyWorkload(j).ARMS == ['perception'] for j in jobs)
    changed = copy.deepcopy(doc)
    changed['members'][1]['job_config']['sample_ids'][0] = 'different'
    with pytest.raises(ValueError):
        check_compiled(changed)


@pytest.mark.parametrize('change', ['seeds', 'temperature', 'duplicate', 'axis', 'missing_auth'])
def test_reject_invalid_learning_campaign(change):
    c = learning_config()
    if change == 'seeds': c['collection']['seeds'] = [101, 102, 103, 103]
    if change == 'temperature': c['collection']['temperature'] = 0
    if change == 'duplicate': c['workload']['sample_ids'][1] = c['workload']['sample_ids'][0]
    if change == 'axis': c['workload']['axis'] = {'kind': 'breadth', 'value': 12}
    if change == 'missing_auth': c['workload'].pop('learning_authorization')
    with pytest.raises(ValueError):
        compile_campaign(c)


def test_vector_view_must_preserve_source_bytes_and_order(tmp_path, monkeypatch):
    import p_only_workload as workload
    source = tmp_path / 'source.jsonl'
    rows = [json.dumps({'id': str(i), 'vector': [i]}).encode() + b'\n' for i in range(3)]
    source.write_bytes(b''.join(rows))
    monkeypatch.setitem(workload.CONSTRUCTION_VECTORS, 'perception', ('source.jsonl', hashlib.sha256(source.read_bytes()).hexdigest()))
    view = tmp_path / 'view.jsonl'
    view.write_bytes(rows[2] + rows[0])
    verify_vector_selection(tmp_path, {'relpath': 'view.jsonl'}, 'perception', ['2', '0'])
    view.write_bytes(rows[0] + rows[2])
    with pytest.raises(ValueError, match='exact construction625'):
        verify_vector_selection(tmp_path, {'relpath': 'view.jsonl'}, 'perception', ['2', '0'])
