from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
HOST=ROOT.parents[1]/'scripts/npu'
if not (HOST/'a0_full1080_result_queue.py').exists():pytest.skip('public umbrella host driver required',allow_module_level=True)
sys.path.insert(0,str(HOST))
import a0_full1080_result_queue as m


def test_waits_without_terminal_despite_complete_manifest():
    assert not m.check_terminal({},None,{'status':'COMPLETED'},None)


def test_complete_correct_source_allowed():
    assert m.check_terminal({}, {'exit':0},{'run_id':m.RUN,'status':'COMPLETED'},
        {'run_id':m.RUN,'attempt_id':m.ATTEMPT,'state':'FINALIZED'})


@pytest.mark.parametrize('fault',['failed-handoff','failed-exit','missing-manifest','wrong-run','open-attempt','wrong-attempt'])
def test_refuses_failed_or_unsealed_source(fault):
    h={};t={'exit':0};man={'run_id':m.RUN,'status':'COMPLETED'}
    a={'run_id':m.RUN,'attempt_id':m.ATTEMPT,'state':'FINALIZED'}
    if fault=='failed-handoff':h={'state':'failed'}
    elif fault=='failed-exit':t['exit']=1
    elif fault=='missing-manifest':man=None
    elif fault=='wrong-run':man['run_id']='other'
    elif fault=='open-attempt':a['state']='OPENED'
    else:a['attempt_id']='other'
    with pytest.raises(ValueError):m.check_terminal(h,t,man,a)
