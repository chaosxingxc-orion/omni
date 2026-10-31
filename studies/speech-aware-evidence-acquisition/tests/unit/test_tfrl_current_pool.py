import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/analysis'))
import tfrl_current_pool as m
from test_tfrl_current_half_result import result


def halves():
    out={}
    for half,n in [('R01C-H1',313),('R01C-H2',312)]:
        ids=[half+str(i) for i in range(n)]
        results={}
        for arm in (*m.ARMS,'vote'):
            r=result(True);row=r['per_sample'][0]
            r['per_sample']=[{**row,'sample_id':sid} for sid in ids];r['aggregate']['samples']=n;results[arm]=r
        out[half]=({'half':half,'iteration':1,'row_plan':[{'row_id':sid} for sid in ids]},results)
    return out


@pytest.mark.parametrize('fault',[None,'missing','overlap','iteration','scorer'])
def test_exact_full_construction_pool(fault):
    h=halves()
    if fault=='missing':h.pop('R01C-H2')
    elif fault=='overlap':h['R01C-H2'][0]['row_plan'][0]['row_id']=h['R01C-H1'][0]['row_plan'][0]['row_id']
    elif fault=='iteration':h['R01C-H2'][0]['iteration']=2
    elif fault=='scorer':
        for r in h['R01C-H2'][1].values():r['aggregate']['evaluator_commit']='other'
    if fault:
        with pytest.raises(ValueError):m.pool(h)
    else:
        d=m.pool(h);assert d['rows']==625 and d['vote_minus_lock_pp']==0
        assert not d['construction_gate_passed'] and not d['next_iteration_authorized']


def test_pool_uses_counts_instead_of_averaging_half_f1():
    h=halves()
    r=h['R01C-H2'][1]['vote']
    for row in r['per_sample']:
        for kind in ('word','char'):
            row[f'slu_f1_{kind}_tp']=0;row[f'slu_f1_{kind}_fp']=10;row[f'slu_f1_{kind}_fn']=10
    r['aggregate']['official']['slu_f1']=0
    d=m.pool(h);pooled=d['official_reaggregated']['vote']['slu_f1']
    assert pooled==pytest.approx(313/(313+3120))
    assert pooled!=pytest.approx(.5)
