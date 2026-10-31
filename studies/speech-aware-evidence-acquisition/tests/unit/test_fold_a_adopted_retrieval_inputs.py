import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/flights/nx'))
import fold_a_adopted_retrieval_inputs as m


@pytest.fixture
def source(tmp_path,monkeypatch):
    repo=tmp_path/'repo';data=tmp_path/'data'
    def put(root,rel,doc):
        p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(m.raw(doc));return m.sha(p)
    rows=[{'id':f'row{i}','audio_sha256':f'{i:064x}','source_frames':4,'source_sample_rate_hz':16000,
           'segments':[{'id':str(i),'start_frame':0,'end_frame':4,'vectors':{'content':[.25,-0.0]}}]} for i in range(1080)]
    roster={'items':[{'sample_id':r['id']} for r in rows],'split_identity_hash':'split'}
    roster_sha=put(repo,m.ROSTER,roster);dataset_sha=put(repo,m.DATASET,{})
    inv={'rows':[{k:v for k,v in r.items() if k!='segments'} for r in rows],
         'roster_sha256':roster_sha,'split_identity_hash':'split','dataset_config':{'relpath':m.DATASET,'sha256':dataset_sha,'canonical_runtime_view_sha256':'a'*64}}
    invsha=put(data,'sources.json',inv);encsha=put(data,'encoder.json',{})
    q=data/'prepared/queries.jsonl';q.parent.mkdir(parents=True);q.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    ish=put(data,'prepared/index.json',{'output':{'rows':rows}})
    protocol={'dimension':2,'cpu_isolation':{},'sources':{'path':'${SAEA_DATA_ROOT}/sources.json','sha256':invsha},
        'encoder':{'path':'${SAEA_DATA_ROOT}/encoder.json','sha256':encsha},'roster_sha256':roster_sha,
        'expected_unique_spans':1080,'query_rows_relpath':'prepared/queries.jsonl','output_relpath':'prepared/index.json'}
    psha=put(repo,m.PROTOCOL,protocol)
    plan={'run_id':m.RUN,'attempt_id':'encode-v1','protocol_hash':psha,'split_identity_hash':'split'}
    put(repo,m.REGISTRATION,{'plans':[{'execution_plan':plan}]})
    receipt={'status':'SEALED','run_id':m.RUN,'attempt_id':'encode-v1','study_head':'head',
        'protocol_sha256':psha,'encoder_inventory_sha256':encsha,'source_inventory_sha256':invsha,
        'logical_encoder_contacts':1080,'cpu_isolation':{},'query_rows_path':str(q),'query_rows_sha256':m.sha(q),'index_sha256':ish}
    rsha=put(data,m.RECEIPT,receipt)
    put(data,f'attempts/{m.RUN}/encode-v1.json',dict(state='FINALIZED',run_id=m.RUN,attempt_id='encode-v1'))
    adoption={'schema':'saea-cpu-preparation-adoption-v1','status':'ADOPTED-SEALED','source_run_id':m.RUN,
        'source_attempt_id':'encode-v1','source_study_head':'head','source_receipt_sha256':rsha,'rows':1080,
        'logical_encoder_contacts':1080,'model_contacts_added':0,'encoder_contacts_added':0,
        'replayed':False,'downstream_adopted':True,'query_rows_sha256':m.sha(q),'index_sha256':ish}
    put(data,m.ADOPTION,adoption)
    monkeypatch.setattr(m.subprocess,'check_output',lambda args,**kw:'head' if 'rev-parse' in args else b'')
    return repo,data,rows,put


@pytest.mark.parametrize('fault',[None,'adoption','receipt','query','index','inventory','registration','attempt'])
def test_adopted_encoder_source_validation(source,fault):
    repo,data,rows,put=source
    paths={'adoption':data/m.ADOPTION,'receipt':data/m.RECEIPT,'query':data/'prepared/queries.jsonl',
        'index':data/'prepared/index.json','inventory':data/'sources.json','registration':repo/m.REGISTRATION,
        'attempt':data/f'attempts/{m.RUN}/encode-v1.json'}
    if fault:
        p=paths[fault]
        if fault in ('query','index','inventory'):p.write_bytes(p.read_bytes()+b' ')
        else:
            d=m.read(p)
            if fault=='adoption':d['downstream_adopted']=False
            elif fault=='receipt':d['logical_encoder_contacts']=1079
            elif fault=='registration':d['plans'][0]['execution_plan']['run_id']='wrong'
            else:d['state']='OPENED'
            p.write_text(json.dumps(d))
        with pytest.raises(ValueError):m.source_config(repo,data)
    else:
        alias,actual=m.source_config(repo,data)
        assert actual==rows and alias['expected_rows']==1080


def test_shared_derivatives_preserve_all1080_values_without_reencoding(source):
    repo,data,rows,put=source
    out=data/'work/codex/derived';result=m.prepare(repo,data,out)
    assert result['rows']==1080 and result['new_model_contacts']==result['new_encoder_contacts']==0
    flat=[json.loads(line) for line in (data/m.CONTENT).read_text().splitlines()]
    assert {r['id']:r['vectors']['content'] for r in flat}=={r['id']:r['segments'][0]['vectors']['content'] for r in rows}
    with pytest.raises(ValueError,match='fresh owned'):m.prepare(repo,data,out)
