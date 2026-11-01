import pytest
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import build_final_messages,render,LearningError
COMMON={'agent_objective':'Intent and slots','learning_objective':'Conditional evidence use','num_experiences':2}

def test_historical_prompt_bytes_are_preserved():
    material='{}'
    assert build_final_messages(COMMON,material,128,16000)==[
        {'role':'system','content':render('BATCH_EXPERIENCE_UPDATE_TEMPLATE_SP',**COMMON)},
        {'role':'user','content':render('BATCH_EXPERIENCE_UPDATE_TEMPLATE_UP',experiences_and_operations=material)}]

def test_new_synthesis_removes_one_per_input_constraint_and_budgets_output():
    common={**COMMON,'final_merge_policy':'bounded-synthesis-v1','final_merge_limits':{'max_items':128,'max_characters':16000}}
    messages=build_final_messages(common,'{"proposed_operations":[]}',128,16000)
    assert 'MUST exactly match the number' not in messages[0]['content']
    assert 'independent of the input count' in messages[0]['content']
    assert '10000 total Unicode' in messages[0]['content']
    assert '48 rules' in messages[0]['content']
    assert '128 rules and 16000' in messages[0]['content']
    assert 'applicability conditions and exceptions' in messages[0]['content']
    assert 'proposed_operations' in messages[1]['content']

@pytest.mark.parametrize('policy',['unknown','bounded-synthesis-v1'])
def test_synthesis_cannot_omit_exact_bounds(policy):
    with pytest.raises(LearningError):build_final_messages({**COMMON,'final_merge_policy':policy},'{}',128,16000)
