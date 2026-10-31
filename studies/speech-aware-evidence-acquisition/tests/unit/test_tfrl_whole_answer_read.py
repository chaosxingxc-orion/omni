import copy,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/analysis'))
import tfrl_whole_answer_read as m


def a(intent='x',values=()):
    return {'scenario':intent,'action':'set','entities':[{'type':'name','filler':v} for v in values]}


def test_whole_answer_does_not_mix_losing_intent_entities():
    rows=[a('x',['a']),a('x',['a','b']),a('x',['a']),a('y',['b'])];before=copy.deepcopy(rows)
    chosen,index=m.select(rows)
    assert index==0 and chosen==rows[0] and rows==before
    chosen['entities'].clear();assert rows==before


def test_exact_jaccard_medoid_can_select_later_complete_answer():
    rows=[a(values=['a']),a(values=['a','b']),a(values=['b']),a('y')]
    assert m.select(rows)==(rows[1],1)


def test_intent_and_medoid_ties_use_first_index():
    rows=[a('y',['b']),a('x',['a']),a('x',['a']),a('y',['c'])]
    assert m.select(rows)==(rows[0],0)
    assert m.select([a(),a(),a(),a()])[1]==0


def test_normalization_only_used_for_similarity():
    rows=[a(values=['  A ']),a(values=['a']),a(values=['B']),a('y')]
    assert m.select(rows)[0]['entities'][0]['filler']=='  A '


@pytest.mark.parametrize('fault',['count','field','entity','type'])
def test_malformed_candidate_refused(fault):
    rows=[a(),a(),a(),a()]
    if fault=='count':rows.pop()
    elif fault=='field':rows[0]['gold']='forbidden'
    elif fault=='entity':rows[0]['entities']=[{'type':'x'}]
    else:rows[0]['scenario']=None
    with pytest.raises(ValueError):m.select(rows)


@pytest.mark.parametrize('slu,intent,passed',[(.681,.8,True),(.679,.8,False),(.7,.79,False)])
def test_gate_cannot_trade_intent_for_slu_or_reopen_old_family(slu,intent,passed):
    result=m.gate({'slu_f1':slu,'intent_accuracy':intent},{'slu_f1':.67,'intent_accuracy':.8})
    assert result['screen_passed'] is passed
    assert result['old_family_stays_stopped'] and not result['system_accepted'] and not result['generation_authorized']
