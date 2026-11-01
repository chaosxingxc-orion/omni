import importlib.util
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[2]/'scripts/analysis/tfgrpo_population_audit.py'
s=importlib.util.spec_from_file_location('population_audit',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def fixture():
    labels={
        'a':dict(utterance_id='u1',scenario='s',intent='s_x',slots=['one','one'],sentence_sha256='t'),
        'b':dict(utterance_id='u1',scenario='s',intent='s_x',slots=['one'],sentence_sha256='t'),
        'c':dict(utterance_id='u3',scenario='s',intent='s_y',slots=['two'],sentence_sha256='v')}
    return {'h1':['a'],'h2':['b','c'],'learned':['a']},labels,{'a':'hash','b':'hash','c':'other'}


def test_detects_group_and_audio_overlap_without_inventing_label_pairs():
    result=m.summarize(*fixture())
    assert result['h1_h2_overlap']==dict(sample_ids=0,utterance_ids=1,audio_hashes=1,normalized_train_sentences=1)
    c=result['populations']['learned']['original_train_coverage']
    assert c['intent_slot']['total_observed_train']==2
    assert c['intent_slot']['missing']==[['s_y','two']]
    assert c['slot']['max_rows_per_present_label']==1


@pytest.mark.parametrize('fault',['duplicate','foreign','missing'])
def test_population_custody_refuses_invalid_inputs(fault):
    groups,labels,audio=fixture()
    if fault=='duplicate':groups['h1'].append('a')
    elif fault=='foreign':groups['learned']=['c']
    else:del audio['b']
    with pytest.raises(ValueError):m.summarize(groups,labels,audio)
