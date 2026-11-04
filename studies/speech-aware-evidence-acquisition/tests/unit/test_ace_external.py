import copy
import json
from pathlib import Path
import shutil

import pytest
from speech_aware_evidence_acquisition.rails import ace_external as ace, external_exploration as external
from speech_aware_evidence_acquisition.rails import tfgrpo_consumption as consumption
from speech_aware_evidence_acquisition.rails.tfgrpo_external_release import ROSTER, DATASET
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha
from test_external_exploration import job, write
from test_external_teacher import network

ROOT=Path(__file__).parents[2]


def prepared(tmp_path,monkeypatch,source_id='slurp:train:one'):
    path,root=job(tmp_path);cfg=external.read(path);source=external.read(cfg['source']['path'])
    source['document']['groups'][0]['query_id']=source_id;source['document']['model_id']='actor'
    cfg['source']=write(Path(cfg['source']['path']),source);cfg['query_ids']=[source_id]
    context={'schema':'saea-ace-offline-additive-context-v1','editable_memory':{'version':'empty-v1','items':{},'prior_updates':[]},
             'frozen_knowledge':{'source_sha256':cfg['source']['sha256'],'editable':False},'operations_allowed':['ADD'],
             'actor_model_id':'actor','bounds':{'max_operations_per_candidate':2}}
    cfg['context']=write(tmp_path/'context-input.json',context);cfg['profile']=external.pin(ROOT/'configs/learners/tfrl/ace.json')
    write(path,cfg);external.prepare(path)
    op={'operation':'ADD','target_id':None,'condition':'Conflicting intent evidence is present','content':'Compare the action boundary before choosing.','exception':'When evidence is absent retain uncertainty.'}
    candidate={'id':'one','reflection':{'hypothesis':'Knowledge is available but not consumed','evidence_query_ids':[source_id],'counterexample':'No available evidence'},'operations':[op,copy.deepcopy(op)],'validation':'Target repairs and unrelated controls'}
    second=copy.deepcopy(candidate);second['id']='two';second['operations'][0]['content']='Use the supplied slot boundary.'
    response={'id':'ace-response','model':'external-model','choices':[{'finish_reason':'stop','message':{'content':json.dumps({'algorithm':'ace','candidates':[candidate,second]})}}],'usage':{'prompt_tokens':100,'completion_tokens':90}}
    _,_,caller,request=external.load_prepared(root);network(caller,monkeypatch,[response]);caller(request);external.collect(root)
    spec={'schema':'saea-ace-external-compile-v1','prepared':external.pin(root/'prepared.json'),'collected':external.pin(root/'collected.json'),'candidate_index':0}
    return spec,root


def test_real_wire_compile_deduplicates_without_losing_provenance(tmp_path,monkeypatch):
    spec,root=prepared(tmp_path,monkeypatch);candidate=ace.compile_preview(spec)
    assert len(candidate['items'])==1 and list(candidate['item_operation_indices'].values())==[[0,1]]
    assert candidate['exposure']['slurp:train:one']['rollout_count']==2
    assert len(candidate['exposure']['slurp:train:one']['request_sha256'])==2
    assert candidate['teacher_model_id']=='external-model' and candidate['model_id']=='actor'
    assert candidate['learning_usage']['calls']==1 and not candidate['programmatic_applicability_gate']
    from speech_aware_evidence_acquisition.rails.external_teacher import ExternalTeacher
    monkeypatch.setattr(ExternalTeacher,'__call__',lambda *a:pytest.fail('unexpected dispatch'))
    assert ace.compile_preview(spec)==candidate


@pytest.mark.parametrize('fault',['wire','proposal','source','bool_index','tfgrpo','parent','modify','foreign','sample_id','missing_exception'])
def test_rejects_untrusted_ace_material(tmp_path,monkeypatch,fault):
    spec,root=prepared(tmp_path,monkeypatch)
    if fault=='wire':
        _,_,caller,request=external.load_prepared(root);directory,_=caller._binding(request);(directory/'response/response.bin').write_text('{}')
    elif fault=='proposal':
        doc=external.read(root/'collected.json');doc['proposals']['candidates'][0]['operations'][0]['content']='forged';spec['collected']=write(root/'collected.json',doc)
    elif fault=='source':Path(external.read(root/'prepared.json')['source']['path']).write_text('{}')
    elif fault=='bool_index':spec['candidate_index']=True
    elif fault=='tfgrpo':spec['schema']='saea-tfgrpo-external-compile-v1'
    else:
        c=ace.compile_preview(spec);context=external.read(root/'context.json');proposal=copy.deepcopy(c['proposal'])
        if fault=='parent':
            context['editable_memory']['items']={'invented':'rule'}
            with pytest.raises(ValueError):ace.validate_context(context,c['source'])
            return
        if fault=='modify':proposal['operations'][0]['operation']='MODIFY'
        if fault=='foreign':proposal['reflection']['evidence_query_ids']=['slurp:train:foreign']
        if fault=='sample_id':proposal['operations'][0]['content']='memorize slurp:train:one'
        if fault=='missing_exception':proposal['operations'][0].pop('exception')
        with pytest.raises(ValueError):ace.validated_items(proposal,context,c['learned_query_ids'])
        return
    with pytest.raises((ValueError,KeyError)):ace.compile_preview(spec)


@pytest.fixture
def release_material(tmp_path,monkeypatch):
    repo=tmp_path/'repo';data=tmp_path/'data';data.mkdir()
    for rel in (ROSTER,DATASET):
        p=repo/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,p)
    rows=external.read(repo/DATASET)['items'];roster=external.read(repo/ROSTER)['items']
    source=rows[0]['sample_id'];target=roster[0]['sample_id'];selected=roster[1]['sample_id']
    spec,_=prepared(data,monkeypatch,source);candidate=ace.compile_preview(spec)
    def dp(name,value):
        p=data/name;write(p,value);return {'relpath':name,'sha256':consumption.digest(p)}
    def rp(rel):return {'relpath':rel,'sha256':consumption.digest(repo/rel)}
    doc={'schema':ace.SCHEMA,'candidate':dp('candidate.json',candidate),'source_manifest':dp('spec.json',spec),'consumer_roster':rp(ROSTER),'learning_dataset':rp(DATASET),'consumer_ids':[target],'selection_ids':[selected]}
    review={'schema':ace.REVIEW_SCHEMA,'decision':'eligible-for-development-evaluation','reviewer':'unit reviewer','procedure':'inspect complete actual exposure','bindings':{k:v for k,v in doc.items() if k!='schema'},'items':{k:{'content_sha256':sha(v),'decision':'supported','evidence':[{'query_id':source,'group_sha256':'a'*64}],'reason':'conditional evidence use'} for k,v in candidate['items'].items()}}
    doc['review']=dp('review.json',review);dp('release.json',doc)
    base=repo/'configs/probes/base.txt';base.parent.mkdir(parents=True);base.write_text('Solver baseline')
    kwargs=dict(candidate_path=data/'candidate.json',source_manifest=data/'spec.json',candidate_release=data/'release.json',repo=repo,data=data,base_template='probes/base.txt',template='probes/ace.txt',consumer_ids=[target],model_id='actor')
    return kwargs,candidate,doc,review,source,selected,dp


def test_released_ace_reaches_shared_solver_template_and_pair(release_material,monkeypatch):
    k,c,_,_,source,_,_=release_material
    monkeypatch.setattr(consumption,'verify_learning_seal',lambda *a,**kw:pytest.fail('ACE impersonates native seal'))
    params=ace.prepare(**k)
    assert params['source_kind']=='external-ace-v1'
    text=(k['repo']/'configs/probes/ace.txt').read_text();assert 'ACE reviewed' in text and source not in text
    values={'native_episode':{'consumption_module':{'components':[{'component_id':consumption.COMPONENT,'state':'on','params':params}]}},'phase_material':{'solver':{'instruction_template':params['template']['relpath']}}}
    assert consumption.verify_for_flight(values,repo=k['repo'],data=k['data'],model_id='actor',consumer_ids=k['consumer_ids'])==c['sha256']
    params['target_arm']='learned';values['phase_material']['solver']['instruction_template']=params['base_template']['relpath']
    values['native_episode']['roles']={'solver':{'template_variants':{'node':'solver','arms':[{'arm_id':'learned','instruction_template':{'relpath':'configs/'+params['template']['relpath'],'sha256':params['template']['sha256']},'deliver':False},{'arm_id':'lock','deliver':True}]}}}
    assert consumption.verify_for_flight(values,repo=k['repo'],data=k['data'],model_id='actor',consumer_ids=k['consumer_ids'])==c['sha256']


@pytest.mark.parametrize('fault',['missing_review','tfgrpo_review','source_kind','template','selection','empty_selection','learned','foreign_evidence','unsupported'])
def test_release_rejects_unreviewed_or_leaking_ace(release_material,fault):
    k,c,doc,review,source,selected,dp=release_material
    if fault in ('source_kind','template'):
        params=ace.prepare(**k)
        if fault=='source_kind':params['source_kind']='external-tfgrpo-v1'
        else:(k['repo']/'configs/probes/ace.txt').write_text('changed')
        with pytest.raises(ValueError):consumption.verify(params,repo=k['repo'],data=k['data'],model_id='actor',consumer_ids=k['consumer_ids'])
        return
    if fault=='missing_review':k.pop('candidate_release')
    if fault=='tfgrpo_review':review['schema']='saea-tfgrpo-external-review-v1'
    if fault=='selection':doc['selection_ids']=k['consumer_ids'];review['bindings']['selection_ids']=k['consumer_ids']
    if fault=='empty_selection':doc['selection_ids']=[];review['bindings']['selection_ids']=[]
    if fault=='learned':k['consumer_ids']=[source];doc['consumer_ids']=[source];review['bindings']['consumer_ids']=[source]
    if fault=='foreign_evidence':review['items'][next(iter(c['items']))]['evidence'][0]['query_id']=selected
    if fault=='unsupported':review['items'][next(iter(c['items']))]['decision']='unsupported'
    doc['review']=dp('review.json',review);dp('release.json',doc)
    with pytest.raises((ValueError,TypeError)):ace.prepare(**k)


def test_workload_metadata_does_not_mislabel_ace():
    import sys
    sys.path.insert(0,str(ROOT/'scripts/analysis'))
    from external_experience_workload import algorithm_metadata
    assert algorithm_metadata('external-ace-v1')['id']=='ace-external-additive-offline-v1'
    with pytest.raises(ValueError):algorithm_metadata('unknown')
