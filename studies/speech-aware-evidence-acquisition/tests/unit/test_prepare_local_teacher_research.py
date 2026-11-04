import importlib.util
from pathlib import Path
import pytest
SCRIPT=Path(__file__).resolve().parents[2]/'scripts/analysis/prepare_local_teacher_research.py'
s=importlib.util.spec_from_file_location('prep_local_research',SCRIPT);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_excludes_same_utterance_different_audio():
    ids=['slurp:train:000001:a'];excluded=['slurp:train:000001:b']
    data={'items':[{'sample_id':ids[0],'audio_sha256':'a'},{'sample_id':excluded[0],'audio_sha256':'b'}]}
    with pytest.raises(ValueError,match='overlaps'):m.audit_partition(ids,excluded,data)

def test_excludes_same_audio_different_utterance():
    ids=['slurp:train:000001:a'];excluded=['slurp:train:000002:b']
    data={'items':[{'sample_id':s,'audio_sha256':'same'} for s in ids+excluded]}
    with pytest.raises(ValueError,match='overlaps'):m.audit_partition(ids,excluded,data)

def test_no_test_tuning_and_no_missing_rows():
    with pytest.raises(ValueError,match='TRAIN'):m.audit_partition(['slurp:test:000001:a'],[],{'items':[]})
    with pytest.raises(ValueError,match='absent'):m.audit_partition(['slurp:train:000001:a'],[],{'items':[]})

def test_disjoint_roster_retains_explicit_limit():
    ids=['slurp:train:000001:a'];excluded=['slurp:train:000002:b']
    data={'items':[{'sample_id':s,'audio_sha256':s} for s in ids+excluded]}
    a=m.audit_partition(ids,excluded,data)
    assert a['overlap']=={'query':0,'utterance':0,'audio':0}
    assert a['excluded_audio_ids_resolved']==1
