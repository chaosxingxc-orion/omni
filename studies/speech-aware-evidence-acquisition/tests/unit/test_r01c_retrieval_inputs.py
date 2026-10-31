"""Reject incomplete or substituted encoder artifacts before derivative writes."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/flights/nx'))
import r01c_retrieval_inputs as prep


def put(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(prep.raw(doc))


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    repo, data = tmp_path / 'repo', tmp_path / 'data'
    rows = [{'id': f'slurp:train:{i}:recording', 'audio_sha256': f'{i:064x}',
        'source_frames': 4, 'source_sample_rate_hz': 16000,
        'segments': [{'id': f'span-{i}', 'start_frame': 0, 'end_frame': 4,
            'vectors': {'content': [0.25, -0.0, 0.875]}}]} for i in range(625)]
    inventory = {'rows': [{k:v for k,v in r.items() if k!='segments'} for r in rows]}
    put(data / 'sources.json', inventory); put(data / 'encoder.json', {})
    query_rel = f'prepared/{prep.RUN}/queries.jsonl'
    query = data / query_rel; query.parent.mkdir(parents=True)
    query.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    index_rel = f'prepared/{prep.RUN}/index.json'
    put(data / index_rel, {'output': {'rows': rows}})
    protocol = {'dimension': 3, 'query_rows_relpath': query_rel, 'output_relpath': index_rel,
        'cpu_isolation': {'cpu_ids': [192]},
        'sources': {'path': '${SAEA_DATA_ROOT}/sources.json', 'sha256': prep.sha(data/'sources.json')},
        'encoder': {'path': '${SAEA_DATA_ROOT}/encoder.json', 'sha256': prep.sha(data/'encoder.json')}}
    put(repo / prep.PROTOCOL, protocol)
    put(repo / prep.ROSTER, {'items': [{'sample_id':r['id']} for r in rows]})
    put(repo / prep.DATASET, {})
    protocol.update(expected_unique_spans=625, roster_sha256=prep.sha(repo/prep.ROSTER))
    put(repo / prep.PROTOCOL, protocol)
    plan = {'run_id': prep.RUN, 'attempt_id':'encode-v1', 'split_identity_hash':'split',
        'protocol_hash': prep.sha(repo / prep.PROTOCOL)}
    put(repo / prep.REGISTRATION, {'plans':[{'execution_plan':plan}]})
    runtime = data / 'runtime.json'
    put(runtime, {'study_head':'encoder-head', 'plan':plan,
        'protocol':{'path':str(repo/prep.PROTOCOL), 'sha256':prep.sha(repo/prep.PROTOCOL)}})
    receipt = {'status':'SEALED', 'run_id':prep.RUN, 'attempt_id':'encode-v1',
        'protocol_sha256':prep.sha(repo/prep.PROTOCOL), 'study_head':'encoder-head',
        'encoder_inventory_sha256':protocol['encoder']['sha256'],
        'source_inventory_sha256':protocol['sources']['sha256'],
        'cpu_isolation':protocol['cpu_isolation'], 'logical_encoder_contacts':625,
        'query_rows_path':str(query), 'query_rows_sha256':prep.sha(query),
        'index_sha256':prep.sha(data/index_rel)}
    put(data/prep.RECEIPT,receipt)
    put(data/'attempts'/prep.RUN/'encode-v1.json',{'state':'FINALIZED'})
    exit_path=data/'exit';exit_path.write_text('0\n')
    monkeypatch.setattr(prep.subprocess, 'check_output', lambda args,**kw:'encoder-head' if 'rev-parse' in args else b'')
    monkeypatch.setattr(prep, 'validate_inventory', lambda *args:({'split_identity_hash':'split'},inventory,625,1))
    return repo,data,runtime,prep.sha(runtime),exit_path,rows


def test_complete625_uses_actual_isolated_receipt(inputs):
    repo,data,runtime,digest,exit_path,rows=inputs
    config,actual=prep.source_config(repo,data,runtime,digest,exit_path)
    assert actual==rows and config['expected_rows']==625
    assert config['source_receipts'][0]['relpath']==prep.RECEIPT
    assert 'CPU-ISOLATED' in prep.RECEIPT


@pytest.mark.parametrize('fault',['exit','runtime','receipt-head','contacts','attempt','source','query','index','registration'])
def test_reject_changed_or_incomplete_encoder(inputs,fault):
    repo,data,runtime,digest,exit_path,rows=inputs
    if fault=='exit':exit_path.write_text('1')
    elif fault=='runtime':runtime.write_bytes(runtime.read_bytes()+b' ')
    elif fault in ('receipt-head','contacts'):
        p=data/prep.RECEIPT;doc=prep.read(p)
        doc['study_head' if fault=='receipt-head' else 'logical_encoder_contacts']='wrong'
        put(p,doc)
    elif fault=='attempt':put(data/'attempts'/prep.RUN/'encode-v1.json',{'state':'OPENED'})
    elif fault=='registration':put(repo/prep.REGISTRATION,{'plans':[{'execution_plan':{}}]})
    else:
        protocol=prep.read(repo/prep.PROTOCOL)
        p={'source':data/'sources.json','query':data/protocol['query_rows_relpath'],
           'index':data/protocol['output_relpath']}[fault]
        p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError):prep.source_config(repo,data,runtime,digest,exit_path)
    assert not (data/prep.SEGMENTS).exists()


def test625_cannot_pass_fold_a_default_or_duplicate_source(inputs):
    *_,rows=inputs
    with pytest.raises(ValueError):prep.verify_rows(rows,rows,3)
    sources=copy.deepcopy(rows);sources[-1]=sources[0]
    with pytest.raises(ValueError):prep.verify_rows(rows,sources,3,625)


def test_real_builders_preserve625_and_refuse_replay(inputs):
    repo,data,runtime,digest,exit_path,rows=inputs
    out=data/'work/codex/derived'
    result=prep.prepare(repo,data,runtime,digest,exit_path,out)
    assert result['rows']==625 and result['values_unchanged']
    assert result['new_model_contacts']==result['new_encoder_contacts']==0
    assert result['encoder_run_id']==prep.RUN and len(result['pins'])==6
    flat=[json.loads(line) for line in (data/prep.CONTENT).read_text().splitlines()]
    assert {r['id']:r['vectors']['content'] for r in flat}=={
        r['id']:r['segments'][0]['vectors']['content'] for r in rows}
    assert '-0.0' in (data/prep.CONTENT).read_text()
    with pytest.raises(ValueError,match='fresh owned'):
        prep.prepare(repo,data,runtime,digest,exit_path,out)
    with pytest.raises(ValueError,match='existing derivative'):
        prep.prepare(repo,data,runtime,digest,exit_path,data/'work/codex/other')


def test_partial_derivative_kept_without_encoder_replay(inputs,monkeypatch):
    sys.path.insert(0,str(ROOT/'scripts'))
    import build_slurp_query_content_derivative as derivative
    repo,data,runtime,digest,exit_path,_=inputs
    def fail(*a,**kw):raise RuntimeError('interruption')
    monkeypatch.setattr(derivative,'build_query_content_derivative',fail)
    out=data/'work/codex/partial'
    with pytest.raises(RuntimeError):prep.prepare(repo,data,runtime,digest,exit_path,out)
    before=(data/prep.SEGMENTS).read_bytes()
    assert prep.read(out/'failed.json')['automatic_retry'] is False
    with pytest.raises(ValueError):prep.prepare(repo,data,runtime,digest,exit_path,data/'work/codex/retry')
    assert (data/prep.SEGMENTS).read_bytes()==before
