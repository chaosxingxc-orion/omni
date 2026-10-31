"""Complete source-bound vector conversion; no historical fixture writes."""
import copy
import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/flights/nx'))
sys.path.insert(0, str(REPO / 'scripts'))
import fold_a_retrieval_inputs as prep


def put(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(prep.raw(document))


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    repo, data = tmp_path / 'repo', tmp_path / 'data'
    rows = [{'id': f'slurp:train:{i:06}:recording', 'audio_sha256': f'{i:064x}',
             'source_frames': 4, 'source_sample_rate_hz': 16000,
             'segments': [{'id': f'span-{i}', 'start_frame': 0, 'end_frame': 4,
                           'vectors': {'content': [0.25, -0.0, 0.875]}}]} for i in range(1080)]
    put(repo / prep.ROSTER, {'items': [{'sample_id': row['id']} for row in rows]})
    put(repo / prep.DATASET, {'items': []})
    query_rel = f'prepared/{prep.RUN}/queries.jsonl'
    query = data / query_rel; query.parent.mkdir(parents=True)
    query.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    index_rel = f'prepared/{prep.RUN}/index.json'
    put(data / index_rel, {'output': {'rows': rows}})
    put(data / 'coordination/sources.json', {'rows': [{k:v for k,v in row.items() if k!='segments'} for row in rows]})
    protocol = {'dimension': 3, 'query_rows_relpath': query_rel, 'output_relpath': index_rel,
        'roster_sha256': prep.sha(repo / prep.ROSTER), 'encoder': {'sha256': 'e' * 64},
        'sources': {'path': '${SAEA_DATA_ROOT}/coordination/sources.json',
                    'sha256': prep.sha(data / 'coordination/sources.json')}}
    put(repo / prep.PROTOCOL, protocol)
    receipt = {'status': 'SEALED', 'run_id': prep.RUN, 'attempt_id': 'encode-v1',
        'study_head': 'encoder-head', 'protocol_sha256': prep.sha(repo / prep.PROTOCOL),
        'encoder_inventory_sha256': protocol['encoder']['sha256'],
        'source_inventory_sha256': protocol['sources']['sha256'],
        'query_rows_path': str(query), 'query_rows_sha256': prep.sha(query),
        'index_sha256': prep.sha(data / index_rel)}
    put(data / prep.RECEIPT, receipt)
    put(data / 'attempts' / prep.RUN / 'encode-v1.json', {'state': 'FINALIZED'})
    expected = data / 'expected.json'; put(expected, {'source_commit': 'encoder-head'})
    parent = data / prep.PARENT
    put(parent / 'status.json', {'state': 'done', 'pid': 12})
    put(parent / 'started.json', {'pid': 12, 'expected': prep.read(expected)})
    put(parent / 'formal.exit.json', {'exit': 0})
    put(parent / 'complete.json', {'rows': 1080, 'run_id': prep.RUN,
        'receipt': str(data / prep.RECEIPT), 'receipt_sha256': prep.sha(data / prep.RECEIPT)})
    monkeypatch.setattr(prep.subprocess, 'check_output', lambda args,**kw: 'tool-head' if 'rev-parse' in args else b'')
    return repo, data, expected, rows


def test_real_builders_preserve_every_vector_and_refuse_replay(inputs):
    repo, data, expected, rows = inputs
    before = (data / prep.RECEIPT).read_bytes()
    out = data / 'output'
    result = prep.prepare(repo, data, expected, out, 'tool-head')
    assert result['rows'] == 1080 and result['values_unchanged']
    assert result['model_contacts'] == result['encoder_contacts'] == 0
    assert len(result['pins']) == 6
    assert '-0.0' in (data / prep.CONTENT).read_text()
    assert (data / prep.RECEIPT).read_bytes() == before
    assert prep.read(data / prep.PROVENANCE)['parent_sha256'] == prep.sha(data / prep.SEGMENTS)
    with pytest.raises(ValueError, match='fresh external'):
        prep.prepare(repo, data, expected, out, 'tool-head')
    with pytest.raises(ValueError, match='existing derivative'):
        prep.prepare(repo, data, expected, data / 'different-output', 'tool-head')


@pytest.mark.parametrize('fault', ['running', 'exit', 'receipt', 'source-commit', 'attempt', 'query', 'index', 'source', 'roster'])
def test_incomplete_or_changed_encoder_never_creates_derivatives(inputs, fault):
    repo, data, expected, _ = inputs
    parent = data / prep.PARENT
    if fault == 'running': put(parent / 'status.json', {'state': 'running', 'pid': 12})
    elif fault == 'exit': put(parent / 'formal.exit.json', {'exit': 1})
    elif fault == 'receipt':
        doc = prep.read(parent / 'complete.json'); doc['receipt_sha256'] = 'wrong'; put(parent / 'complete.json', doc)
    elif fault == 'source-commit': put(expected, {'source_commit': 'wrong'})
    elif fault == 'attempt': put(data / 'attempts' / prep.RUN / 'encode-v1.json', {'state': 'OPENED'})
    else:
        protocol = prep.read(repo / prep.PROTOCOL)
        path = {'query': data / protocol['query_rows_relpath'], 'index': data / protocol['output_relpath'],
                'source': data / 'coordination/sources.json', 'roster': repo / prep.ROSTER}[fault]
        path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError):
        prep.prepare(repo, data, expected, data / 'output', 'tool-head')
    assert not (data / 'output').exists()
    assert not (data / prep.SEGMENTS).exists()


@pytest.mark.parametrize('fault', ['duplicate', 'missing', 'audio', 'partial-span', 'multiple-segments', 'width', 'nan', 'bool'])
def test_rows_require_exact_source_and_complete_finite_vectors(inputs, fault):
    _, _, _, source_rows = inputs
    rows = copy.deepcopy(source_rows)
    if fault == 'duplicate': rows[-1] = rows[0]
    elif fault == 'missing': rows.pop()
    elif fault == 'audio': rows[0]['audio_sha256'] = 'f' * 64
    elif fault == 'partial-span': rows[0]['segments'][0]['end_frame'] = 3
    elif fault == 'multiple-segments': rows[0]['segments'] *= 2
    elif fault == 'width': rows[0]['segments'][0]['vectors']['content'].pop()
    elif fault == 'nan': rows[0]['segments'][0]['vectors']['content'][0] = float('nan')
    elif fault == 'bool': rows[0]['segments'][0]['vectors']['content'][0] = True
    with pytest.raises(ValueError): prep.verify_rows(rows, source_rows, 3)


def test_wrong_source_commit_fails_before_outputs(inputs):
    repo, data, expected, _ = inputs
    with pytest.raises(ValueError, match='pinned clean'):
        prep.prepare(repo, data, expected, data / 'output', 'wrong')


def test_path_escape_refused(tmp_path):
    with pytest.raises(ValueError): prep.confined(tmp_path, '../escape')


def test_partial_alias_is_preserved_and_never_replayed(inputs, monkeypatch):
    import build_slurp_query_content_derivative as derivative
    repo, data, expected, _ = inputs
    def fail(*args, **kwargs): raise RuntimeError('simulated interruption after alias')
    monkeypatch.setattr(derivative, 'build_query_content_derivative', fail)
    out = data / 'output'
    with pytest.raises(RuntimeError, match='simulated interruption'):
        prep.prepare(repo, data, expected, out, 'tool-head')
    alias = (data / prep.SEGMENTS).read_bytes()
    assert prep.read(out / 'failed.json')['automatic_retry'] is False
    assert not (out / 'complete.json').exists()
    with pytest.raises(ValueError, match='existing derivative'):
        prep.prepare(repo, data, expected, data / 'new-output', 'tool-head')
    assert (data / prep.SEGMENTS).read_bytes() == alias
