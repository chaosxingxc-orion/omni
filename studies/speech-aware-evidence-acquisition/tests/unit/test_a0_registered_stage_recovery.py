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
