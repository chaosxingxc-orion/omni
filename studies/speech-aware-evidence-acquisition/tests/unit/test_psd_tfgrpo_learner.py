import copy
import json
from pathlib import Path
import pytest
from speech_aware_evidence_acquisition.rails import tfgrpo_external_reference as reference, tfgrpo_external_compile as compiler
from speech_aware_evidence_acquisition.rails import external_exploration as external
from test_tfgrpo_reference_release import completed
from test_tfgrpo_external_reference import material
from test_tfgrpo_external_proposals import fixture
from test_external_exploration import job, write
from test_external_teacher import network

@pytest.mark.parametrize('target',['slot','decision'])
def test_fused_psd_prepare_collect_compile(tmp_path,monkeypatch,target):
    cfgpath,cfg=material(tmp_path);batch=json.loads(Path(cfg['source_batch']['path']).read_text());document=json.loads(Path(cfg['source']['path']).read_text())['document']
    document['offline_stale_policy']=False;cfg['source']=write(Path(cfg['source']['path']),{'document':document})
    groups,context,candidate=fixture();qid=batch['groups'][0]['query_id']; context.update(model_id=batch['model_id'],parent=batch['parent'],rollout_policy_sha256=batch['policy_sha256'],rollout_kind=batch['rollout_kind']);context['psd']={'target':target,'source_batch':cfg['source_batch']}
    candidate.update(target=target+'_experience_library', parent_sha256=context['parent']['sha256'],evidence_query_ids=[qid]);candidate['relative_feedback'][0].update(query_id=qid,higher_trajectory_id=batch['groups'][0]['rollouts'][1]['trajectory_id'],lower_trajectory_id=batch['groups'][0]['rollouts'][0]['trajectory_id'])
    # Fixture batch supplies genuine structured test trajectories, not production evidence.
    (tmp_path/'job').mkdir();j,root=job(tmp_path/'job');jc=external.read(j);jc.update(source=cfg['source'],query_ids=[qid],context=write(tmp_path/'context-psd.json',context),profile=external.pin(external.ROOT/f'configs/learners/tfrl/tfgrpo-psd-{target}.json'))
    write(j,jc);external.prepare(j)
    second=copy.deepcopy(candidate);second['id']='second';second['operations'][0]['content']='When evidence is uncertain, compare available cues.'
    response={'id':'fixture','model':'external-model','choices':[{'finish_reason':'stop','message':{'content':json.dumps({'algorithm':'tfgrpo','candidates':[candidate,second]})}}],'usage':{'prompt_tokens':10,'completion_tokens':10}}
    _,_,caller,req=external.load_prepared(root);network(caller,monkeypatch,[response]);caller(req);external.collect(root)
    spec={'schema':'saea-tfgrpo-external-compile-v1','prepared':external.pin(root/'prepared.json'),'collected':external.pin(root/'collected.json'),'candidate_index':0}
    preview=compiler.compile_preview(spec);assert preview['psd_target']==target and preview['items']
    batch['groups'][0]['rollouts'][0]['policy_sha256']='f'*64
    Path(cfg['source_batch']['path']).write_text(json.dumps(batch))
    with pytest.raises(ValueError):compiler.compile_preview(spec)

@pytest.mark.parametrize('target',['slot','decision'])
def test_reference_psd_fixed_policy_preparation(tmp_path,target):
    path,cfg=material(tmp_path);cfg['psd_target']=target;cfg['teacher']['research_prompt']=reference.pin(reference.ROOT/f'configs/prompts/learner/tfrl/tfgrpo-reference-psd-{target}.md');write(path,cfg)
    manifest=reference.prepare(path);assert manifest['plan']['planned_calls']>0 and manifest['new_model_calls']==0
    assert reference.load(Path(cfg['output_root']))[2]['psd_target']==target

@pytest.mark.parametrize('target',['slot','decision'])
def test_reference_psd_complete_fixture_replay(tmp_path,monkeypatch,target):
    from test_tfgrpo_external_reference import responses
    from speech_aware_evidence_acquisition.rails import tfgrpo_reference_release as release
    path,cfg=material(tmp_path);cfg['psd_target']=target;cfg['teacher']['research_prompt']=reference.pin(reference.ROOT/f'configs/prompts/learner/tfrl/tfgrpo-reference-psd-{target}.md');write(path,cfg)
    reference.prepare(path);root=Path(cfg['output_root']);network(reference.load(root)[-1],monkeypatch,responses());reference.run(root)
    spec={'schema':'saea-tfgrpo-reference-compile-v1','prepared':reference.pin(root/'prepared.json'),'collected':reference.pin(root/'collected.json')}
    result=release.compile_preview(spec);assert result['psd_target']==target and result['items'] and result['plan']['planned_calls']==5
