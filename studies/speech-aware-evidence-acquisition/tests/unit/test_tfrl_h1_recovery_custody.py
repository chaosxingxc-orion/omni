from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts/analysis"))
import copy
import pytest
from tfrl_h1_recovery_custody import partition,context_integrity

def row(sid,compacted=False):
    contacts=[{'contact_id':str(i),'context_compaction':({'dropped_identifiers':['evidence']} if compacted and i==1 else None)} for i in range(7)]
    return {'sample_id':sid,'text':'must not affect selection','native_episode_receipt':{
        'context_fit':{'schema':'saea-native-context-fit-v1','contacts':contacts,'compacted_contacts':int(compacted)}}}

def test_partition_by_context_integrity_preserves_costly_but_invalid_source():
    rows=[row('a'),row('b',True)]
    expected={'reusable_ids':['a'],'affected_completed_ids':['b'],'remaining_ids':['b','c'],'uncompleted_ids':['c']}
    assert partition(['a','b','c'],rows)==expected
    for r in rows:r['text']='different answer';r['correct']=True
    assert partition(['a','b','c'],rows)==expected

@pytest.mark.parametrize('fault',['missing','count','contact','duplicate','outside'])
def test_unverifiable_context_never_silently_reused(fault):
    r=row('a');rows=[r]
    if fault=='missing':r['native_episode_receipt'].pop('context_fit')
    elif fault=='count':r['native_episode_receipt']['context_fit']['compacted_contacts']=1
    elif fault=='contact':r['native_episode_receipt']['context_fit']['contacts'].pop()
    elif fault=='duplicate':rows.append(copy.deepcopy(r))
    else:r['sample_id']='outside'
    with pytest.raises(ValueError):partition(['a','b'],rows)
