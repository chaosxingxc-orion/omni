"""Imported summaries advance the dependency graph without new generation spend."""
import pytest
from test_tfgrpo_learning import Model,group,MODEL,POLICY,LEARNING_CONFIG
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import ExperienceLearner,empty_experience,LearningError

@pytest.mark.parametrize('n',[1,2])
def test_imported_summaries_are_used_and_never_regenerated(tmp_path,n):
    model=Model();parent=empty_experience(MODEL);g=group(parent)
    imported={r['trajectory_id']:'Historical summary '+str(i) for i,r in enumerate(g['rollouts'][:n])}
    pin={'relpath':'historical/import.json','sha256':'1'*64}
    custody={'source':pin,'imported_summaries':n,'original_total_usage':{'calls':12}}
    learner=ExperienceLearner(tmp_path,model_id=MODEL,metered_call=model,learning_config_sha256=LEARNING_CONFIG,max_calls=5-n,concurrency=2)
    args=dict(agent_objective='Intent',learning_objective='Evidence',policy_sha256=POLICY,summary_import=pin,imported_summaries=imported,import_custody=custody)
    result=learner.learn([g],parent,**args)
    assert len(model.requests)==5-n
    assert result['learning_usage']['calls']==5-n
    assert result['planned_learning_calls']==5-n
    assert result['historical_summary_custody']==custody
    assert not any(r['stage']=='summary' and r['identity'] in imported for r in model.requests)
    advantage=next(r for r in model.requests if r['stage']=='advantage')
    assert all(text in advantage['messages'][1]['content'] for text in imported.values())
    assert learner.learn([g],parent,**args)==result
    assert len(model.requests)==5-n
    changed={**custody,'original_total_usage':{'calls':11}}
    with pytest.raises(LearningError,match='custody changed'):
        learner.learn([g],parent,**{**args,'import_custody':changed})

def test_unpinned_summary_is_rejected_before_generation(tmp_path):
    model=Model();learner=ExperienceLearner(tmp_path,model_id=MODEL,metered_call=model,learning_config_sha256=LEARNING_CONFIG,max_calls=5)
    with pytest.raises(LearningError,match='pinned custody'):
        learner.learn([group()],empty_experience(MODEL),agent_objective='Intent',learning_objective='Evidence',policy_sha256=POLICY,imported_summaries={'q1-t0':'old'})
    assert model.requests==[]
