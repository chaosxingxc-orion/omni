from pathlib import Path
import json
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
HOST=ROOT.parents[1]/'scripts/npu'
if not (HOST/'a0_resume_after_resident_refresh.py').exists():pytest.skip('umbrella required',allow_module_level=True)
sys.path.insert(0,str(HOST))
import a0_resume_after_resident_refresh as m


def put(path, doc):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(doc))


@pytest.mark.parametrize('fault',[None,'admitted','wrong_session','dirty','already_dispatched'])
def test_registered_stage_never_repeats_repin_or_resident_registration(tmp_path,monkeypatch,fault):
    monkeypatch.setattr(m,'D',tmp_path)
    out=tmp_path/'work/codex/recovery'; failed=tmp_path/'work/codex/failed'; tree=tmp_path/'tree'
    study=tree/'studies/speech-aware-evidence-acquisition';pkg=study/'docs/readiness'/m.NAME
    put(out/'status.json',{'state':'failed','error':'chore: register recovered A0 resident'})
    put(failed/'runtime/registration-complete.json',{'session':'resident','model_calls':0})
    put(pkg/'candidate.json',{'service_session':'resident','decision_relpath':'decision.json'})
    put(pkg/'resident-refresh.json',{'readiness':'held'})
    if fault=='admitted':put(tmp_path/'decision.json',{})
    elif fault=='wrong_session':put(failed/'runtime/registration-complete.json',{'session':'other','model_calls':0})
    elif fault=='already_dispatched':(out/'run').mkdir()
    monkeypatch.setattr(m,'git',lambda *_:'dirty' if fault=='dirty' else '')
    calls=[]
    class ReachedAdmission(Exception):pass
    def formal(tree,args,log,*extra):
        calls.append(args[0]);raise ReachedAdmission()
    monkeypatch.setattr(m,'formal',formal)
    if fault:
        with pytest.raises(ValueError):m._resume(tree,failed,out,True)
        assert calls==[]
    else:
        with pytest.raises(ReachedAdmission):m._resume(tree,failed,out,True)
        assert calls==['scripts/analysis/admit_real_engine_candidate.py']
        assert (out/'empty-commit-failure.json').exists()


@pytest.mark.parametrize('fault',[None,'wrong_failure','attempt','not_ready','missing_admission','dirty'])
def test_admitted_recovery_checks_without_reminting(tmp_path,monkeypatch,fault):
    monkeypatch.setattr(m,'D',tmp_path)
    tree=tmp_path/'tree';study=tree/'studies/speech-aware-evidence-acquisition'
    failed=tmp_path/'work/codex/previous';out=tmp_path/'work/codex/new'
    put(failed/'status.json',{'state':'failed','error':'flight_registration_check.py'})
    put(failed/'registration-check.log',{'passed':False,'blocking_cells':[m.NAME],
        'cells':[{'findings':[{'kind':'registered-carrier-binding'}]*2}]})
    put(failed/'admit.log',{'admission_minted':True,'new_model_calls':0})
    put(study/'docs/readiness'/m.NAME/'candidate.json',{'decision_relpath':'decision.json','run_id':'run'})
    put(study/'configs/experiment'/f'{m.NAME}.json',{'cells':[{'readiness':{'state':'ready'}}]})
    put(tmp_path/'decision.json',{})
    if fault=='wrong_failure':put(failed/'status.json',{'state':'failed','error':'other'})
    elif fault=='attempt':(tmp_path/'attempts/run').mkdir(parents=True)
    elif fault=='missing_admission':(tmp_path/'decision.json').unlink()
    elif fault=='not_ready':put(study/'configs/experiment'/f'{m.NAME}.json',{'cells':[{'readiness':{'state':'held'}}]})
    monkeypatch.setattr(m,'git',lambda *_:'dirty' if fault=='dirty' else '')
    calls=[]
    class ReachedCheck(Exception):pass
    def formal(tree,args,log,*extra):calls.append(args[0]);raise ReachedCheck()
    monkeypatch.setattr(m,'formal',formal)
    with pytest.raises(ValueError if fault else ReachedCheck):m.resume_admitted(tree,failed,out)
    assert calls==([] if fault else ['scripts/checks/flight_registration_check.py'])
