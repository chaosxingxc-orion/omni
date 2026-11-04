"""Complete wire-backed multistage replay, release isolation, and real Flight routing."""
import copy
import json
from pathlib import Path
import shutil
import sys

import pytest
from speech_aware_evidence_acquisition.rails import tfgrpo_reference_release as adapter
from speech_aware_evidence_acquisition.rails import tfgrpo_external_reference as ref
from speech_aware_evidence_acquisition.rails import tfgrpo_consumption as consumption
from speech_aware_evidence_acquisition.rails.tfgrpo_external_release import ROSTER, DATASET
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha
from test_tfgrpo_external_reference import material, responses, write
from test_external_teacher import network
ROOT = Path(__file__).parents[2]


def completed(tmp_path, monkeypatch, source_id=None):
    path, cfg = material(tmp_path)
    if source_id:
        old = cfg['query_ids'][0]
        batch = json.loads(Path(cfg['source_batch']['path']).read_text().replace(old, source_id))
        source = json.loads(Path(cfg['source']['path']).read_text().replace(old, source_id))
        group = batch['groups'][0]
        source['document']['source_batch_sha256'] = sha(batch)
        source['document']['groups'][0]['group_sha256'] = sha(group)
        source['document']['groups'][0]['evidence'] = ref.project_group_evidence(group, evidence_view='full-query-v1')
        cfg['query_ids'] = [source_id]
        cfg['source_batch'] = write(Path(cfg['source_batch']['path']), batch)
        cfg['source'] = write(Path(cfg['source']['path']), source)
        write(path, cfg)
    ref.prepare(path)
    root = Path(cfg['output_root']); calls = network(ref.load(root)[-1], monkeypatch, responses())
    ref.run(root)
    spec = {'schema':'saea-tfgrpo-reference-compile-v1', 'prepared':ref.pin(root/'prepared.json'),
            'collected':ref.pin(root/'collected.json')}
    return spec, root, calls


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_complete_stage_replay_read_only_accounting_and_identity(tmp_path, monkeypatch):
    spec, root, calls = completed(tmp_path, monkeypatch)
    before = snapshot(root)
    c = adapter.compile_preview(spec)
    assert len(calls) == 5 and snapshot(root) == before
    assert c['strategy'] == adapter.STRATEGY and c['plan']['rollouts'] == 2
    assert len(c['contacts']) == 5 and c['learning_usage']['calls'] == 5
    assert not c['native_learning_seal'] and c['new_model_calls'] == 0
    assert c['items'] == ref.read(root/'learner/candidate.json')['items']
    assert c == adapter.compile_preview(spec)
    assert c['usage_identity'] == adapter.compile_preview(spec)['usage_identity']


@pytest.mark.parametrize('fault', ['wire','missing','unknown','candidate','algorithm','record','incomplete','intent','source'])
def test_untrusted_or_incomplete_stage_rejected_without_dispatch(tmp_path, monkeypatch, fault):
    spec, root, calls = completed(tmp_path, monkeypatch)
    if fault == 'wire': next((root/'teacher').glob('*/response/response.bin')).write_text('{}')
    elif fault == 'missing': next((root/'learner/contacts').glob('*.response.json')).unlink()
    elif fault == 'unknown': next((root/'teacher').glob('*/response/result.json')).unlink()
    elif fault == 'candidate':
        p = root/'learner/candidate.json'; c = ref.read(p)
        c['items'][next(iter(c['items']))] = 'Forged teacher knowledge'
        c['sha256'] = sha({k:v for k,v in c.items() if k != 'sha256'}); write(p, c)
        record = ref.read(root/'collected.json'); record['candidate'] = ref.pin(p); record['candidate_sha256'] = c['sha256']
        spec['collected'] = write(root/'collected.json', record)
    elif fault == 'algorithm':
        p=root/'prepared.json'; doc=ref.read(p);doc['algorithm']='tfgrpo-external-relative-feedback-v1'
        spec['prepared']=write(p,doc)
    elif fault == 'record':
        p=root/'collected.json';doc=ref.read(p);doc['contacts'].pop();spec['collected']=write(p,doc)
    elif fault == 'intent':
        p=next((root/'learner/contacts').glob('*.intent.json'));doc=ref.read(p);doc['messages'][0]['content']='forged';write(p,doc)
    elif fault == 'source': Path(ref.read(root/'job.json')['source']['path']).write_text('{}')
    else:
        # A collected pin cannot substitute for missing completed candidate bytes.
        (root/'learner/candidate.json').unlink()
    before=snapshot(root)
    with pytest.raises((ValueError, FileNotFoundError)): adapter.compile_preview(spec)
    assert len(calls)==5 and snapshot(root)==before


@pytest.fixture
def release_material(tmp_path, monkeypatch):
    repo=tmp_path/'repo';data=tmp_path/'data';data.mkdir()
    for rel in (ROSTER,DATASET):
        p=repo/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,p)
    rows=ref.read(repo/DATASET)['items'];roster=ref.read(repo/ROSTER)['items']
    source=rows[0]['sample_id'];target=roster[0]['sample_id'];selected=roster[1]['sample_id']
    spec,_,_=completed(data,monkeypatch,source);candidate=adapter.compile_preview(spec)
    def dp(name,value):
        p=data/name;write(p,value);return {'relpath':name,'sha256':consumption.digest(p)}
    def rp(rel):return {'relpath':rel,'sha256':consumption.digest(repo/rel)}
    doc={'schema':adapter.SCHEMA,'candidate':dp('candidate.json',candidate),'source_manifest':dp('spec.json',spec),
         'consumer_roster':rp(ROSTER),'learning_dataset':rp(DATASET),'consumer_ids':[target],'selection_ids':[selected]}
    review={'schema':adapter.REVIEW_SCHEMA,'decision':'eligible-for-development-evaluation','reviewer':'unit reviewer',
        'procedure':'inspect complete source and exact text','bindings':{k:v for k,v in doc.items() if k!='schema'},
        'items':{k:{'content_sha256':sha(v),'decision':'supported','evidence':[{'query_id':source,'group_sha256':candidate['evidence_groups'][source]}],
                    'reason':'conditional evidence use'} for k,v in candidate['items'].items()}}
    doc['review']=dp('review.json',review);dp('release.json',doc)
    base=repo/'configs/probes/base.txt';base.parent.mkdir(parents=True);base.write_text('Solver baseline')
    kwargs=dict(candidate_path=data/'candidate.json',source_manifest=data/'spec.json',candidate_release=data/'release.json',repo=repo,data=data,
        base_template='probes/base.txt',template='probes/reference.txt',consumer_ids=[target],model_id='frozen-actor')
    return kwargs,candidate,doc,review,source,selected,dp


def test_reviewed_reference_reaches_actual_flight_template_and_workload(release_material, monkeypatch):
    k,c,*_=release_material
    monkeypatch.setattr(consumption,'verify_learning_seal',lambda *a,**kw:pytest.fail('reference impersonates native seal'))
    params=adapter.prepare(**k)
    assert params['source_kind']==adapter.KIND
    values={'native_episode':{'consumption_module':{'components':[{'component_id':consumption.COMPONENT,'state':'on','params':params}]}},
            'phase_material':{'solver':{'instruction_template':params['template']['relpath']}}}
    assert consumption.verify_for_flight(values,repo=k['repo'],data=k['data'],model_id=k['model_id'],consumer_ids=k['consumer_ids'])==c['sha256']
    params['target_arm']='learned';values['phase_material']['solver']['instruction_template']=params['base_template']['relpath']
    values['native_episode']['roles']={'solver':{'template_variants':{'node':'solver','arms':[
        {'arm_id':'learned','instruction_template':{'relpath':'configs/'+params['template']['relpath'],'sha256':params['template']['sha256']},'deliver':False},
        {'arm_id':'lock','deliver':True}]}}}
    assert consumption.verify_for_flight(values,repo=k['repo'],data=k['data'],model_id=k['model_id'],consumer_ids=k['consumer_ids'])==c['sha256']
    sys.path.insert(0,str(ROOT/'scripts/analysis'))
    from external_experience_workload import algorithm_metadata
    assert algorithm_metadata(adapter.KIND)['id']==adapter.STRATEGY
    values['phase_material']['solver']['instruction_template']='probes/foreign.txt'
    with pytest.raises(ValueError):consumption.verify_for_flight(values,repo=k['repo'],data=k['data'],model_id=k['model_id'],consumer_ids=k['consumer_ids'])


@pytest.mark.parametrize('fault', ['review','kind','selection','learned','audio','utterance','unsupported','runtime'])
def test_release_isolation_and_algorithm_custody(release_material,fault):
    k,c,doc,review,source,selected,dp=release_material
    if fault=='kind':
        params=adapter.prepare(**k);params['source_kind']='external-tfgrpo-v1'
        with pytest.raises(ValueError):consumption.verify(params,repo=k['repo'],data=k['data'],model_id=k['model_id'],consumer_ids=k['consumer_ids'])
        return
    if fault=='review':review['schema']='saea-tfgrpo-external-review-v1'
    elif fault=='selection':doc['selection_ids']=k['consumer_ids']
    elif fault=='learned':doc['selection_ids']=[source]
    elif fault=='unsupported':next(iter(review['items'].values()))['decision']='unsupported'
    elif fault=='runtime':k['runtime_model_id']='different-runtime'
    else:
        dataset=ref.read(k['repo']/DATASET);source_row=next(r for r in dataset['items'] if r['sample_id']==source)
        target_row=next(r for r in dataset['items'] if r['sample_id']==k['consumer_ids'][0])
        if fault=='audio':target_row['audio_sha256']=source_row['audio_sha256']
        else:
            # Preserve distinct query/audio but make the consumer share the learned utterance.
            new_id=':'.join(source.split(':')[:3])+':other-audio';old=target_row['sample_id'];target_row['sample_id']=new_id
            roster=ref.read(k['repo']/ROSTER)
            next(r for r in roster['items'] if r['sample_id']==old)['sample_id']=new_id
            write(k['repo']/ROSTER,roster);doc['consumer_roster']['sha256']=consumption.digest(k['repo']/ROSTER)
            k['consumer_ids']=[new_id];doc['consumer_ids']=[new_id]
        write(k['repo']/DATASET,dataset);doc['learning_dataset']['sha256']=consumption.digest(k['repo']/DATASET)
    review['bindings']={key:v for key,v in doc.items() if key not in ('schema','review')}
    doc['review']=dp('review.json',review);dp('release.json',doc)
    with pytest.raises(ValueError):adapter.prepare(**k)


def test_incomplete_actual_provider_response_never_compiles(tmp_path, monkeypatch):
    path,cfg=material(tmp_path);ref.prepare(path);root=Path(cfg['output_root'])
    response=responses()[0];response['choices'][0]['finish_reason']='length'
    calls=network(ref.load(root)[-1],monkeypatch,[response])
    with pytest.raises(ValueError,match='incomplete'):ref.run(root)
    spec={'schema':'saea-tfgrpo-reference-compile-v1','prepared':ref.pin(root/'prepared.json'),
          'collected':{'path':str(root/'collected.json'),'sha256':'a'*64}}
    with pytest.raises((ValueError,FileNotFoundError)):adapter.compile_preview(spec)
    assert len(calls)==1 and not (root/'learner/candidate.json').exists()


def test_runtime_alias_keeps_reference_actor(release_material):
    k,c,doc,review,_,_,dp=release_material
    actor={'model_lock_key':'frozen','server_args':['--max-model-len','65536'],
           'decoding_params':{'temperature':0,'seed':42,'max_tokens':16384}}
    runtime=copy.deepcopy(actor);runtime['server_args'][1]='49152';runtime['decoding_params']['max_tokens']=4096
    model_dir=k['repo']/'configs/model';model_dir.mkdir(parents=True)
    pins={}
    for key,name,model in [('actor_model',k['model_id'],actor),('runtime_model','tp4-runtime',runtime)]:
        path=model_dir/(name+'.json');write(path,model)
        pins[key]={'relpath':'configs/model/'+name+'.json','sha256':consumption.digest(path)}
    doc.update(runtime_model_id='tp4-runtime',deployment_binding=dp('deployment.json',{'schema':'saea-external-actor-deployment-v1',**pins}))
    review['bindings']={key:value for key,value in doc.items() if key not in ('schema','review')}
    doc['review']=dp('review.json',review);dp('release.json',doc)
    k.update(runtime_model_id='tp4-runtime',deployment_binding=k['data']/'deployment.json')
    params=adapter.prepare(**k)
    assert consumption.verify(params,repo=k['repo'],data=k['data'],model_id='tp4-runtime',consumer_ids=k['consumer_ids'])==c['sha256']
    with pytest.raises(ValueError):consumption.verify(params,repo=k['repo'],data=k['data'],model_id=k['model_id'],consumer_ids=k['consumer_ids'])


def test_shared_reference_cost_is_identical_across_evaluations(release_material):
    k,c,*_=release_material
    params=adapter.prepare(**k)
    sys.path.insert(0,str(ROOT/'scripts/analysis'))
    from external_experience_read import shared_learning_cost
    values={'native_episode':{'consumption_module':{'components':[{'component_id':consumption.COMPONENT,'state':'on','params':params}]}}}
    cost=shared_learning_cost(values,k['data'])
    assert cost['learning_usage']==c['learning_usage'] and cost['usage_identity']==c['usage_identity']
    assert cost['included_in_evaluation_total_cost'] is False
    assert cost==shared_learning_cost(values,k['data'])
