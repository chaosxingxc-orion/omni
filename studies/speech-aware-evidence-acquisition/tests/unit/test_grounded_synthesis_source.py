import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[2]/'scripts/analysis'))
import grounded_synthesis_source as m
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import empty_experience, canonical


def fixture(tmp, monkeypatch):
    group={'query_id':'slurp:train:1'}
    op={'operation':'ADD','id':None,'content':'Use grounded evidence'}
    hist={'source_run_id':'earlier','source_attempt_id':'learning-v1',
          'original_total_usage':{'calls':308,'prompt_tokens':1000,'completion_tokens':200}}
    hist['native_source_usage']=[{'run_id':'earlier','attempt_id':'learning-v1',
                                 'usage':dict(hist['original_total_usage'])}]
    batch={'schema':'saea-tfgrpo-batch-v1','groups':[group],'model_id':'m',
           'parent':empty_experience('m'),'policy_sha256':'a'*64,'rollout_kind':'shared-perception-adaptation',
           'common':{},'bounds':{'max_operations':32},'final_import':{'relpath':'old','sha256':'b'*64}}
    usage={'calls':1,'prompt_tokens':10,'completion_tokens':2}
    cand={**empty_experience('m'),'batch_sha256':m.sha(batch),'historical_stage_custody':hist,
          'planned_learning_calls':1,'accounting':{'attempted_contacts':1,'observed_contacts':1,
          'unknown_contacts':0,'known_usage':usage},'learning_usage':usage}
    def write(path,doc):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(canonical(doc))
        return {'relpath':str(path.relative_to(tmp)),'sha256':m.sha(doc)}
    cand['sha256']=m.sha({k:v for k,v in cand.items() if k!='sha256'})
    contact=m.sha({'stage':'group_update','identity':group['query_id']})
    rp=tmp/'runs/earlier/learning-v1/tfgrpo/learner/contacts'/(contact+'.response.json')
    response={'response':{'text':json.dumps([op])}}
    response_pin=write(rp,response)
    ops=[{**op,'group_contact':contact,'operation_index':0,'operation_sha256':m.sha(op),
          'response_file_sha256':response_pin['sha256']}]
    spec={'schema':'saea-grounded-synthesis-source-v1',
          'candidate':write(tmp/'learner/candidate.json',cand),
          'source_batch':write(tmp/'learner/batch.json',batch),
          'source_manifest':write(tmp/'manifest.json',{}),'operations':write(tmp/'ops.json',ops)}
    seals=[]
    def verify(*a,**kw):
        seals.append(True)
        return SimpleNamespace(attempt_state='FINALIZED', manifest={
            'attempt':{'run_id':'successful','attempt_id':'learning-v1'}})
    monkeypatch.setattr(m,'verify_learning_seal',verify)
    monkeypatch.setattr(m,'load_import',lambda *a,**kw:([op],hist))
    return spec,seals,rp,ops,cand


def test_preserves_success_and_all_failed_ancestry_cost(tmp_path,monkeypatch):
    spec,seals,*_=fixture(tmp_path,monkeypatch)
    result=m.load_source(spec,data_root=tmp_path)
    assert seals==[True]
    assert result['original_total_usage']=={'calls':309,'prompt_tokens':1010,'completion_tokens':202}
    assert result['source_attempt_state']=='FINALIZED'
    assert [r['usage']['calls'] for r in result['native_source_usage']]==[308,1]
    assert result['unknown_contacts']==0
    assert not result['successful_source_reopened'] and not result['generation_authorized']


@pytest.mark.parametrize('fault',['seal','source','operation','response','unknown','historical_cost',
                                 'duplicate_source','dropped_cost'])
def test_rejects_invalid_lineage_or_dropped_cost(tmp_path,monkeypatch,fault):
    spec,_,rp,ops,cand=fixture(tmp_path,monkeypatch)
    def repin(key,doc):
        p=tmp_path/spec[key]['relpath'];p.write_bytes(canonical(doc));spec[key]['sha256']=m.sha(doc)
    if fault=='seal':
        def reject(*a,**kw):raise ValueError('seal rejected')
        monkeypatch.setattr(m,'verify_learning_seal',reject)
    elif fault=='source':(tmp_path/spec['source_batch']['relpath']).write_text('{}')
    elif fault=='operation':
        ops[0]['content']='unsupported';ops[0]['operation_sha256']=m.sha({k:ops[0][k] for k in ('operation','id','content')});repin('operations',ops)
    elif fault=='response':rp.write_text('{}')
    elif fault in ('duplicate_source','dropped_cost'):
        hist=cand['historical_stage_custody']
        if fault=='duplicate_source':hist['native_source_usage'].append(copy.deepcopy(hist['native_source_usage'][0]))
        else:hist['native_source_usage'][0]['usage']['calls']-=1
        cand['sha256']=m.sha({k:v for k,v in cand.items() if k!='sha256'});repin('candidate',cand)
    else:
        if fault=='unknown':cand['accounting']['unknown_contacts']=1
        else:cand['historical_stage_custody']=copy.deepcopy(cand['historical_stage_custody']);cand['historical_stage_custody']['original_total_usage']['calls']=307
        cand['sha256']=m.sha({k:v for k,v in cand.items() if k!='sha256'});repin('candidate',cand)
    with pytest.raises((ValueError,KeyError)):
        m.load_source(spec,data_root=tmp_path)
