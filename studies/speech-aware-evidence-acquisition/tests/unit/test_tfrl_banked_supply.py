import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/analysis'))
import tfrl_banked_supply as m


@pytest.mark.parametrize('fault',['incomplete','changed_source','changed_bank','producer'])
def test_failed_preparation_preserves_artifact_and_refuses_replay(tmp_path,monkeypatch,fault):
    import speech_aware_evidence_acquisition.contracts as contracts
    import speech_aware_evidence_acquisition.core.bundle.git_commit as git
    import speech_aware_evidence_acquisition.core.flight.factory as factory
    monkeypatch.setattr(contracts,'assert_execution_environment',lambda:None)
    monkeypatch.setattr(git,'require_clean_study_repo',lambda p:'a'*40)
    exp={'cells':[dict(run_id='run',attempt_id='a',model_fragment='m',dataset_fragment='d',baseline_fragment='b')],
         'populations':[{'ids':['one','two']}]}
    checks=[]
    def validate(*a):
        checks.append(1)
        if fault=='changed_bank' and len(checks)>1:raise ValueError('bank changed')
        pin='b' if fault=='changed_source' and len(checks)>1 else 'a'
        return exp,2,{'config':pin},{'relpath':'decision','sha256':'a'}
    monkeypatch.setattr(m,'validate_preparation_inputs',validate)
    calls=[]
    class Runtime:
        def prepare_phase_supply(self,path,**kw):
            calls.append(path);path.parent.mkdir(parents=True)
            output={'one':[]} if fault=='incomplete' else {'one':[],'two':[]}
            path.write_text(json.dumps({'output':{'values':{'first':output}}}))
            if fault=='producer':raise ValueError('producer failed')
    monkeypatch.setattr(factory,'build_flight_runtime',lambda *a,**kw:Runtime())
    out=tmp_path/'work/codex/prepare'
    def run():return m.prepare(tmp_path,tmp_path,'name',2,tmp_path/'decision',out)
    with pytest.raises(ValueError):run()
    assert (out/'failed.json').exists() and (out/'intent.json').exists()
    assert not (out/'complete.json').exists() and calls[0].exists()
    # Retry cannot invoke the producer again, even if a previous output exists.
    monkeypatch.setattr(m,'validate_preparation_inputs',lambda *a:(exp,2,{},{}))
    with pytest.raises(ValueError,match='fresh owned'):run()
    assert len(calls)==1
