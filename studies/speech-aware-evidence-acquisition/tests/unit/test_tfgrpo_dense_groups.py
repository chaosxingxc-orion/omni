import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/analysis'))
from tfgrpo_dense_groups import select_ids


def test_allwrong_contrast_ties_and_correct_exclusion():
    def score(intent,tp,fp,fn):
        return {'scenario_action_exact':intent,'slot_f1_exact_tp':tp,'slot_f1_exact_fp':fp,'slot_f1_exact_fn':fn}
    bad=score(0,0,1,1);partial=score(1,0,1,1);good=score(1,1,0,0)
    tables={a:{'contrast':bad,'tie':partial,'correct':good,'mixed':bad} for a in ('s1','s2','s3','s4')}
    tables['s2']={**tables['s2'],'contrast':partial,'mixed':good}
    assert select_ids(tables,['contrast','tie','correct','mixed'])==(['contrast'],['tie'],['correct','mixed'])
    with pytest.raises(ValueError,match='same-roster'):
        select_ids(tables,['contrast'])
