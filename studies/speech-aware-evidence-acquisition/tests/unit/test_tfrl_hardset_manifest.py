from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/analysis'))
from tfrl_hardset_manifest import partition,coverage

def raw(n,text=None):
    return {'slurp_id':n,'sentence':text or str(n),'scenario':'s','action':'a',
            'entities':[{'type':'slot'}], 'recordings':[{'file':'x'}]}
def item(n):return {'sample_id':f'slurp:train:{n:06d}:audio-x','audio_seconds':1.0}
def ev(n,error=True):return {'row_id':item(n)['sample_id'],'learning_candidate':error,
                            'next_action':'diagnose_or_repeat' if error else 'keep_for_regression','contrast_candidates':[]}

def test_excludes_all_recordings_of_source_utterance_and_keeps_uncertainty():
    rows,info=partition([raw(1),raw(2),raw(3)],{'slurp:train:000001'},
                        {'slurp:train:000002'},[item(3)],[ev(3)])
    assert len(rows)==1 and rows[0]['sample_id']==item(3)['sample_id']
    assert rows[0]['observed_error_candidate'] and not rows[0]['confirmed_hard_example']
    assert info['role_eligible_equals_existing_roster']

def test_normalized_text_duplicate_of_excluded_source_is_blocked():
    rows,info=partition([raw(1,'Same  Text'),raw(2,' same text ')],
                        {'slurp:train:000001'},set(),[item(2)],[ev(2)])
    assert not rows
    assert info['excluded_by_primary_reason']['duplicate_text_of_excluded_source']==1

def test_same_text_candidates_count_once_and_easy_case_stays_regression():
    rows,_=partition([raw(1,'hello'),raw(2,'HELLO')],set(),set(),[item(1),item(2)],[ev(1,False),ev(2)])
    assert len(rows)==1 and not rows[0]['observed_error_candidate']

def test_missing_credit_or_duplicate_roster_refused():
    with pytest.raises(ValueError):partition([raw(1)],set(),set(),[item(1)],[])
    with pytest.raises(ValueError):partition([raw(1)],set(),set(),[item(1),item(1)],[ev(1)])

def test_coverage_counts_observed_relations_only():
    c=coverage([{'intent':'a','slot_types':['x','x']},{'intent':'b','slot_types':['y']}])
    assert c['observed_intent_slot_relations']==2
    assert c['slot_type_counts']=={'x':1,'y':1}
