"""Actual stage orchestration and wire custody with fake HTTP, no research calls."""
import copy
import hashlib
import json
from pathlib import Path
import pytest

from speech_aware_evidence_acquisition.rails import tfgrpo_external_reference as ref
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import empty_experience, sha, DENSE_REWARD
from speech_aware_evidence_acquisition.rails.grounded_synthesis_fusion import project_group_evidence
from test_external_teacher import teacher, network, answer


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    return ref.pin(path)


def material(tmp_path):
    t = teacher(tmp_path, analysis_budget=True)
    actor='frozen-actor'; parent=empty_experience(actor); qid='slurp:train:one:audio-one'
    query={'sample_id':qid,'audio_available_to_reflector':False,'history':[],
           'supplied_evidence':['Boundary: choose the spoken action.'],'task_instruction':'Extract intent and slots.'}
    rows=[]
    for i, intent in enumerate((0,1)):
        scored=json.dumps({'scenario':'one','action':'two','entities':[]})
        critique={'scenario_action_exact':intent,'slot_f1_exact_tp':0,'slot_f1_exact_fp':0,
                  'slot_f1_exact_fn':0,'reward':0.5*intent+0.5,'reward_mode':DENSE_REWARD,
                  'scored_answer_sha256':hashlib.sha256(scored.encode()).hexdigest()}
        trajectory={'request':query,'response_text':scored,'scored_answer_after_fixed_normalization':scored}
        rows.append({'trajectory_id':f't{i}','trajectory':json.dumps(trajectory),'critique':json.dumps(critique),
                     'reward':critique['reward'],'model_id':actor,'policy_sha256':'a'*64,
                     'experience_sha256':parent['sha256']})
    group={'query_id':qid,'query':json.dumps(query),'answer':'Privileged TRAIN reference','rollouts':rows}
    batch={'model_id':actor,'parent':parent,'policy_sha256':'a'*64,
           'rollout_kind':'shared-perception-adaptation','common':{'reward_mode':DENSE_REWARD},'groups':[group]}
    document={'source_batch_sha256':sha(batch),'model_id':actor,'parent':parent,
              'rollout_parent_sha256':parent['sha256'],'rollout_policy_sha256':'a'*64,
              'rollout_kind':batch['rollout_kind'],'reward_mode':DENSE_REWARD,
              'groups':[{'query_id':qid,'group_sha256':sha(group),
                         'evidence':project_group_evidence(group,evidence_view='full-query-v1')}]}
    cfg={'schema':ref.SCHEMA,'source':write(tmp_path/'source.json',{'document':document}),
         'source_batch':write(tmp_path/'batch.json',batch),'parent':write(tmp_path/'parent.json',parent),
         'query_ids':[qid],'actor_model_id':actor,'teacher':t.config,
         'template':{'upstream_commit':ref.UPSTREAM_COMMIT,'source_sha256':ref.SOURCE_SHA256},
         'stage_completion_caps':{s:131072 for s in ref.STAGES},
         'bounds':{'max_calls':5,'concurrency':1,'max_operations':2,'max_items':48,
                   'max_characters':10000,'num_experiences':2,'max_completion_tokens_total':5*131072},
         'objectives':{'agent_objective':'Intent and slots','learning_objective':'Conditional knowledge'},
         'output_root':str(tmp_path/'prepared')}
    path=tmp_path/'job.json';write(path,cfg)
    return path,cfg


def responses():
    ops=json.dumps([{'operation':'ADD','id':None,'content':'Prefer spoken action evidence when object cues conflict.'}])
    return [answer('Observed partial-credit trajectory','s1'),answer('Higher partial-credit trajectory','s2'),
            answer('<Experiences>1. Resolve action using explicit evidence.</Experiences>','a1'),
            answer(ops,'g1'),answer(ops,'b1')]


def test_full_reference_flow_and_replay_with_real_wire(tmp_path,monkeypatch):
    path,cfg=material(tmp_path);prepared=ref.prepare(path)
    root=Path(cfg['output_root']);caller=ref.load(root)[-1]
    calls=network(caller,monkeypatch,responses())
    assert prepared['plan']['stage_calls']=={'summary':2,'advantage':1,'group_update':1,'batch_update':1}
    assert not calls
    result=ref.run(root)
    assert len(calls)==5 and len(result['contacts'])==5
    assert result['actor_model_id']=='frozen-actor' and result['learner_model_id']=='external-model'
    assert result['accounting']['known_usage']['calls']==5
    assert not result['deployment_accepted'] and not result['native_learning_seal']
    assert ref.collect(root)==ref.run(root)==result and len(calls)==5
    stage_messages={}
    for p in (root/'learner/contacts').glob('*.intent.json'):
        request=ref.read(p);stage_messages.setdefault(request['stage'],[]).append(request['messages'])
    assert len(stage_messages['summary'])==2
    assert 'Privileged TRAIN reference' in json.dumps(stage_messages['summary'])
    assert 'Higher partial-credit trajectory' in json.dumps(stage_messages['advantage'])
    assert 'Higher partial-credit trajectory' not in json.dumps(stage_messages['batch_update'])
    # Completed learner JSON alone does not replace original provider wire.
    wire=next((root/'teacher').glob('*/response/response.bin'));wire.write_bytes(wire.read_bytes()+b' ')
    with pytest.raises(ValueError,match='wire changed'):
        ref.collect(root)
    assert len(calls)==5


def test_unknown_stops_without_reissue_and_collect_is_read_only(tmp_path,monkeypatch):
    path,cfg=material(tmp_path);ref.prepare(path);root=Path(cfg['output_root']);caller=ref.load(root)[-1]
    calls=network(caller,monkeypatch,[TimeoutError()])
    with pytest.raises(ValueError,match='no completed'):
        ref.collect(root)
    assert not calls
    for _ in range(2):
        with pytest.raises(Exception):ref.run(root)
    assert len(calls)==1


@pytest.mark.parametrize('fault',['actor','policy','group','projection','feedback','response','template','budget','cap','test'])
def test_prepare_rejects_changed_sources_and_impossible_budget(tmp_path,fault):
    path,cfg=material(tmp_path)
    if fault in ('policy','group','projection','feedback','response'):
        source=ref.read(cfg['source']['path']);batch=ref.read(cfg['source_batch']['path'])
        if fault=='policy':batch['groups'][0]['rollouts'][0]['policy_sha256']='b'*64
        elif fault=='group':source['document']['groups'][0]['group_sha256']='b'*64
        elif fault=='projection':source['document']['groups'][0]['evidence']['reference']='changed'
        elif fault=='feedback':
            row=batch['groups'][0]['rollouts'][0];c=json.loads(row['critique']);c['reward']=0.9;row['critique']=json.dumps(c)
        elif fault=='response':
            row=batch['groups'][0]['rollouts'][0];t=json.loads(row['trajectory']);t['scored_answer_after_fixed_normalization']='changed';row['trajectory']=json.dumps(t)
        # Rehash outer pins to prove internal provenance is checked too.
        cfg['source_batch']=write(Path(cfg['source_batch']['path']),batch)
        source['document']['source_batch_sha256']=sha(batch)
        if fault not in ('group','projection'):
            g=batch['groups'][0];source['document']['groups'][0]['group_sha256']=sha(g)
            if fault!='response':source['document']['groups'][0]['evidence']=project_group_evidence(g,evidence_view='full-query-v1')
        cfg['source']=write(Path(cfg['source']['path']),source)
    elif fault=='actor':cfg['actor_model_id']='foreign'
    elif fault=='template':cfg['template']['source_sha256']='b'*64
    elif fault=='budget':cfg['bounds']['max_completion_tokens_total']=131072
    elif fault=='cap':cfg['stage_completion_caps']['summary']=4096
    elif fault=='test':cfg['query_ids']=['slurp:test:one:audio-one']
    write(path,cfg)
    with pytest.raises(ValueError):ref.prepare(path)
    assert not Path(cfg['output_root']).exists()


def test_interrupted_after_observed_stage_resumes_without_duplicate_api(tmp_path, monkeypatch):
    path,cfg=material(tmp_path);ref.prepare(path);root=Path(cfg['output_root'])
    original_load=ref.load; caller=original_load(root)[-1]
    calls=network(caller,monkeypatch,responses())
    def interrupted_load(value):
        parts=list(original_load(value)); actual=parts[-1]; original=actual.preflight
        def preflight(request):
            if len(calls)==1:
                raise RuntimeError('controlled stop before next contact intent')
            return original(request)
        actual.preflight=preflight
        return tuple(parts)
    monkeypatch.setattr(ref,'load',interrupted_load)
    with pytest.raises(RuntimeError,match='controlled stop'):
        ref.run(root)
    assert len(calls)==1
    assert len(list((root/'learner/contacts').glob('*.intent.json')))==1
    monkeypatch.setattr(ref,'load',original_load)
    result=ref.run(root)
    assert len(calls)==5 and result['accounting']['known_usage']['calls']==5
    assert ref.collect(root)==result


def test_observed_truncation_retained_and_never_resampled(tmp_path,monkeypatch):
    path,cfg=material(tmp_path);ref.prepare(path);root=Path(cfg['output_root']);caller=ref.load(root)[-1]
    response=answer('Partial summary','length');response['choices'][0]['finish_reason']='length'
    calls=network(caller,monkeypatch,[response])
    for _ in range(2):
        with pytest.raises(ValueError,match='incomplete external reference response'):
            ref.run(root)
    assert len(calls)==1
    assert next((root/'teacher').glob('*/response/result.json')).exists()
    assert not (root/'learner/candidate.json').exists()


def test_audit_replays_translated_advantage_without_writes_or_dispatch(tmp_path, monkeypatch):
    path,cfg=material(tmp_path);ref.prepare(path);root=Path(cfg['output_root']);caller=ref.load(root)[-1]
    rs=responses();rs[2]=answer('```output\n<经验>1. Resolve action using explicit evidence.</经验>\n```','a1')
    calls=network(caller,monkeypatch,rs)
    ref.run(root)
    before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result=ref.audit(root)
    after={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert before==after and len(calls)==5
    assert result['known_contacts_replayable'] and result['known_usage']['calls']==5
    assert not result['new_model_calls'] and not result['source_modified']


def test_audit_distinguishes_unknown_from_unissued_without_retry(tmp_path,monkeypatch):
    path,cfg=material(tmp_path);ref.prepare(path);root=Path(cfg['output_root']);caller=ref.load(root)[-1]
    calls=network(caller,monkeypatch,[TimeoutError()])
    with pytest.raises(Exception):ref.run(root)
    result=ref.audit(root)
    assert len(calls)==1 and result['unknown_contacts']==1
    assert sum(result['unissued_stage_calls'].values())==4
    assert not result['known_contacts_replayable']
