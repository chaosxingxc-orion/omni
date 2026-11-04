import copy
import json
from pathlib import Path
import shutil

import pytest
from speech_aware_evidence_acquisition.rails import reflexion_external as reflexion, external_exploration as external
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
    context={'schema':'saea-reflexion-offline-transfer-context-v1','memory':{'version':'empty-v1','items':[]},
             'frozen_knowledge':{'source_sha256':cfg['source']['sha256'],'editable':False},
             'actor_model_id':'actor','required_transfer_scope':'cross_task_adaptation','retry_history':{'available':False}}
    cfg['context']=write(tmp_path/'context-input.json',context);cfg['profile']=external.pin(ROOT/'configs/learners/tfrl/reflexion.json')
    write(path,cfg);external.prepare(path)
    candidate={'id':'one','hypothesis':'Knowledge is available but not consumed','evidence_query_ids':[source_id],
               'memory':{'trigger':'Conflicting intent evidence is present','action':'Compare the action boundary before choosing.','exception':'When evidence is absent retain uncertainty.'},
               'retry_protocol':'Frozen paired cross-task evaluation','validation':'Target repairs and unrelated controls'}
    second=copy.deepcopy(candidate);second['id']='two';second['memory']['action']='Use the supplied slot boundary.'
    response={'id':'reflexion-response','model':'external-model','choices':[{'finish_reason':'stop','message':{'content':json.dumps({'algorithm':'reflexion','candidates':[candidate,second]})}}],'usage':{'prompt_tokens':100,'completion_tokens':90}}
    _,_,caller,request=external.load_prepared(root);network(caller,monkeypatch,[response]);caller(request);external.collect(root)
    spec={'schema':'saea-reflexion-external-compile-v1','prepared':external.pin(root/'prepared.json'),'collected':external.pin(root/'collected.json'),'candidate_index':0}
    return spec,root


def test_real_wire_compile_deduplicates_without_losing_provenance(tmp_path,monkeypatch):
    spec,root=prepared(tmp_path,monkeypatch);candidate=reflexion.compile_preview(spec)
    assert len(candidate['items'])==1 and list(candidate['item_source_fields'].values())==[['memory']]
    assert candidate['transfer_scope']['authority']=='derived-from-prepared-context'
    assert 'transfer_scope' not in candidate['proposal']
    assert candidate['exposure']['slurp:train:one']['rollout_count']==2
    assert len(candidate['exposure']['slurp:train:one']['request_sha256'])==2
    assert candidate['teacher_model_id']=='external-model' and candidate['model_id']=='actor'
    assert candidate['learning_usage']['calls']==1 and not candidate['programmatic_applicability_gate']
    from speech_aware_evidence_acquisition.rails.external_teacher import ExternalTeacher
    monkeypatch.setattr(ExternalTeacher,'__call__',lambda *a:pytest.fail('unexpected dispatch'))
    assert reflexion.compile_preview(spec)==candidate


@pytest.mark.parametrize('fault',['wire','proposal','source','bool_index','tfgrpo','parent','scope','foreign','sample_id','missing_exception'])
def test_rejects_untrusted_reflexion_material(tmp_path,monkeypatch,fault):
    spec,root=prepared(tmp_path,monkeypatch)
    if fault=='wire':
        _,_,caller,request=external.load_prepared(root);directory,_=caller._binding(request);(directory/'response/response.bin').write_text('{}')
    elif fault=='proposal':
        doc=external.read(root/'collected.json');doc['proposals']['candidates'][0]['memory']['action']='forged';spec['collected']=write(root/'collected.json',doc)
    elif fault=='source':Path(external.read(root/'prepared.json')['source']['path']).write_text('{}')
    elif fault=='bool_index':spec['candidate_index']=True
    elif fault=='tfgrpo':spec['schema']='saea-tfgrpo-external-compile-v1'
    else:
        c=reflexion.compile_preview(spec);context=external.read(root/'context.json');proposal=copy.deepcopy(c['proposal'])
        if fault=='parent':
            context['memory']['items']=['invented']
            with pytest.raises(ValueError):reflexion.validate_context(context,c['source'])
            return
        if fault=='scope':proposal['transfer_scope']='same_task_retry'
        if fault=='foreign':proposal['evidence_query_ids']=['slurp:train:foreign']
        if fault=='sample_id':proposal['memory']['action']='memorize slurp:train:one'
        if fault=='missing_exception':proposal['memory'].pop('exception')
        with pytest.raises(ValueError):reflexion.validated_items(proposal,context,c['learned_query_ids'])
        return
    with pytest.raises((ValueError,KeyError)):reflexion.compile_preview(spec)


@pytest.fixture
def release_material(tmp_path,monkeypatch):
    repo=tmp_path/'repo';data=tmp_path/'data';data.mkdir()
    for rel in (ROSTER,DATASET):
        p=repo/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,p)
    rows=external.read(repo/DATASET)['items'];roster=external.read(repo/ROSTER)['items']
    source=rows[0]['sample_id'];target=roster[0]['sample_id'];selected=roster[1]['sample_id']
    spec,_=prepared(data,monkeypatch,source);candidate=reflexion.compile_preview(spec)
    def dp(name,value):
        p=data/name;write(p,value);return {'relpath':name,'sha256':consumption.digest(p)}
    def rp(rel):return {'relpath':rel,'sha256':consumption.digest(repo/rel)}
    doc={'schema':reflexion.SCHEMA,'candidate':dp('candidate.json',candidate),'source_manifest':dp('spec.json',spec),'consumer_roster':rp(ROSTER),'learning_dataset':rp(DATASET),'consumer_ids':[target],'selection_ids':[selected]}
    review={'schema':reflexion.REVIEW_SCHEMA,'decision':'eligible-for-development-evaluation','reviewer':'unit reviewer','procedure':'inspect complete actual exposure','bindings':{k:v for k,v in doc.items() if k!='schema'},'items':{k:{'content_sha256':sha(v),'decision':'supported','evidence':[{'query_id':source,'group_sha256':'a'*64}],'reason':'conditional evidence use'} for k,v in candidate['items'].items()}}
    doc['review']=dp('review.json',review);dp('release.json',doc)
    base=repo/'configs/probes/base.txt';base.parent.mkdir(parents=True);base.write_text('Solver baseline')
    kwargs=dict(candidate_path=data/'candidate.json',source_manifest=data/'spec.json',candidate_release=data/'release.json',repo=repo,data=data,base_template='probes/base.txt',template='probes/reflexion.txt',consumer_ids=[target],model_id='actor')
    return kwargs,candidate,doc,review,source,selected,dp


def test_released_reflexion_reaches_shared_solver_template_and_pair(release_material,monkeypatch):
    k,c,_,_,source,_,_=release_material
    monkeypatch.setattr(consumption,'verify_learning_seal',lambda *a,**kw:pytest.fail('Reflexion impersonates native seal'))
    params=reflexion.prepare(**k)
    assert params['source_kind']=='external-reflexion-v1'
    text=(k['repo']/'configs/probes/reflexion.txt').read_text();assert 'Reflexion reviewed' in text and source not in text
    values={'native_episode':{'consumption_module':{'components':[{'component_id':consumption.COMPONENT,'state':'on','params':params}]}},'phase_material':{'solver':{'instruction_template':params['template']['relpath']}}}
    assert consumption.verify_for_flight(values,repo=k['repo'],data=k['data'],model_id='actor',consumer_ids=k['consumer_ids'])==c['sha256']
    params['target_arm']='learned';values['phase_material']['solver']['instruction_template']=params['base_template']['relpath']
    values['native_episode']['roles']={'solver':{'template_variants':{'node':'solver','arms':[{'arm_id':'learned','instruction_template':{'relpath':'configs/'+params['template']['relpath'],'sha256':params['template']['sha256']},'deliver':False},{'arm_id':'lock','deliver':True}]}}}
    assert consumption.verify_for_flight(values,repo=k['repo'],data=k['data'],model_id='actor',consumer_ids=k['consumer_ids'])==c['sha256']


@pytest.mark.parametrize('fault',['missing_review','tfgrpo_review','source_kind','template','selection','empty_selection','learned','foreign_evidence','unsupported'])
def test_release_rejects_unreviewed_or_leaking_reflexion(release_material,fault):
    k,c,doc,review,source,selected,dp=release_material
    if fault in ('source_kind','template'):
        params=reflexion.prepare(**k)
        if fault=='source_kind':params['source_kind']='external-tfgrpo-v1'
        else:(k['repo']/'configs/probes/reflexion.txt').write_text('changed')
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
    with pytest.raises((ValueError,TypeError)):reflexion.prepare(**k)


def test_workload_metadata_does_not_mislabel_reflexion():
    import sys
    sys.path.insert(0,str(ROOT/'scripts/analysis'))
    from external_experience_workload import algorithm_metadata
    assert algorithm_metadata('external-reflexion-v1')['id']=='reflexion-external-memory-transfer-offline-v1'
    with pytest.raises(ValueError):algorithm_metadata('unknown')


@pytest.mark.parametrize('context', [{}, {'required_transfer_scope': 'cross_task_adaptation'},
    {'required_transfer_scope': 'same_task_retry', 'retry_history': {'available': False}},
    {'required_transfer_scope': 'cross_task_adaptation', 'retry_history': {'available': True}}])
def test_missing_teacher_scope_requires_explicit_original_offline_context(tmp_path, monkeypatch, context):
    spec, root = prepared(tmp_path, monkeypatch)
    doc = external.read(root / 'collected.json')['proposals']
    profile = external.read(root / 'profile.json')
    with pytest.raises(ValueError, match='algorithm-specific candidate fields'):
        external.validate_proposals(json.dumps(doc), profile, context, ['slurp:train:one'])


def projection_for(proposal):
    fields = {key: {'source_sha256': sha(value), 'spans': [[0, len(value)]]}
              for key, value in proposal['memory'].items()}
    fields['action']['spans'] = [[0, len(proposal['memory']['action']) - 1]]
    return {'schema': 'saea-reflexion-runtime-projection-v1', 'proposal_sha256': sha(proposal),
            'author': 'research reviewer', 'rationale': 'isolate conditional check',
            'decision': 'eligible-for-development-review', 'memory': fields}


def test_projected_reflexion_replays_wire_and_keeps_original(tmp_path, monkeypatch):
    spec, root = prepared(tmp_path, monkeypatch)
    original = reflexion.compile_preview(spec)
    projection = projection_for(original['proposal'])
    spec['runtime_projection'] = write(tmp_path/'projection.json', projection)
    candidate = reflexion.compile_preview(spec)
    assert candidate['proposal'] == original['proposal']
    assert candidate['runtime_projection'] == projection
    assert candidate['runtime_proposal']['memory']['action'] == original['proposal']['memory']['action'][:-1]
    assert candidate['learned_query_ids'] == original['learned_query_ids']
    assert candidate['exposure'] == original['exposure']
    assert candidate['wire_response_sha256'] == original['wire_response_sha256']
    assert candidate['items'] != original['items']
    assert candidate['semantic_validation'] == 'pending' and not candidate['deployment_accepted']
    assert candidate['intervention_kind'] == 'researcher-reviewed-source-substrings'
    # Changing a projection after it has been pinned must fail replay.
    write(tmp_path/'projection.json', {**projection, 'author': 'changed'})
    with pytest.raises(ValueError): reflexion.compile_preview(spec)


@pytest.mark.parametrize('fault', ['proposal', 'field', 'missing', 'insert', 'overlap', 'reverse', 'bool', 'empty', 'unchanged', 'foreign_schema'])
def test_projection_rejects_unbound_or_rewritten_memory(tmp_path, monkeypatch, fault):
    spec, _ = prepared(tmp_path, monkeypatch)
    proposal = reflexion.compile_preview(spec)['proposal']
    doc = projection_for(proposal)
    if fault == 'proposal': doc['proposal_sha256'] = '0' * 64
    elif fault == 'field': doc['memory']['action']['source_sha256'] = '0' * 64
    elif fault == 'missing': doc['memory'].pop('exception')
    elif fault == 'insert': doc['memory']['action']['text'] = 'invented'
    elif fault == 'overlap': doc['memory']['action']['spans'] = [[0, 8], [4, 10]]
    elif fault == 'reverse': doc['memory']['action']['spans'] = [[4, 10], [0, 3]]
    elif fault == 'bool': doc['memory']['action']['spans'] = [[False, 3]]
    elif fault == 'empty': doc['memory']['exception']['spans'] = []
    elif fault == 'unchanged': doc['memory']['action']['spans'] = [[0, len(proposal['memory']['action'])]]
    elif fault == 'foreign_schema': doc['schema'] = 'saea-ace-runtime-projection-v1'
    spec['runtime_projection'] = write(tmp_path/'projection.json', doc)
    with pytest.raises(ValueError): reflexion.compile_preview(spec)


def test_projection_cannot_bypass_source_replay_of_a_release(release_material):
    k, original, doc, review, source, selected, dp = release_material
    spec_path = k['source_manifest']
    spec = external.read(spec_path)
    projection = projection_for(original['proposal'])
    spec['runtime_projection'] = write(k['data']/'projection.json', projection)
    write(spec_path, spec)
    candidate = reflexion.compile_preview(spec)
    doc['candidate'] = dp('candidate.json', candidate)
    doc['source_manifest'] = dp('spec.json', spec)
    review['bindings'] = {key: value for key, value in doc.items() if key not in ('schema', 'review')}
    review['items'] = {key: {'content_sha256': sha(value), 'decision': 'supported',
                           'evidence': [{'query_id': source, 'group_sha256': 'a'*64}],
                           'reason': 'isolated source substring'} for key, value in candidate['items'].items()}
    doc['review'] = dp('review.json', review); dp('release.json', doc)
    params = reflexion.prepare(**k)
    assert consumption.verify(params, repo=k['repo'], data=k['data'], model_id='actor', consumer_ids=k['consumer_ids']) == candidate['sha256']
    assert source not in (k['repo']/'configs/probes/reflexion.txt').read_text()
