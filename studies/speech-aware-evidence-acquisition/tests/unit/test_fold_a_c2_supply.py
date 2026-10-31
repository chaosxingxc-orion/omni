import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/flights/nx'))
import fold_a_c2_supply as m


@pytest.mark.parametrize('fault',[None,'row','phase','shape'])
def test_exact_first_phase_output(fault):
    x={k:{'perception':{'a':[],'b':[]}} for k in ('facets','values')}
    if fault=='row':del x['values']['perception']['b']
    elif fault=='phase':x['facets']['solver']={}
    elif fault=='shape':del x['facets']
    if fault:
        with pytest.raises(ValueError):m.verify_output(x,['a','b'])
    else:m.verify_output(x,['a','b'])


@pytest.mark.parametrize('fault',[None,'partial','changed_source','producer'])
def test_shared_producer_failure_preserved_and_never_replayed(tmp_path,monkeypatch,fault):
    import speech_aware_evidence_acquisition.contracts as contracts
    import speech_aware_evidence_acquisition.core.bundle.git_commit as git
    import speech_aware_evidence_acquisition.core.flight.factory as factory
    monkeypatch.setattr(contracts,'assert_execution_environment',lambda:None)
    calls=[];checks=[]
    monkeypatch.setattr(git,'require_clean_study_repo',lambda _:'head')
    e={'cells':[dict(model_fragment='m',dataset_fragment='d',baseline_fragment='b',attempt_id='a')]}
    def validate(*args):
        checks.append(1);return e,['a','b'],{'hash':'changed' if fault=='changed_source' and len(checks)>1 else 'same'},{}
    monkeypatch.setattr(m,'validate',validate)
    class Runtime:
        def prepare_phase_supply(self,path,**kwargs):
            calls.append(1);path.parent.mkdir(parents=True)
            ids=['a'] if fault=='partial' else ['a','b']
            path.write_text(json.dumps({'output':{k:{'perception':{i:[] for i in ids}} for k in ('facets','values')}}))
            if fault=='producer':raise ValueError('failed')
    monkeypatch.setattr(factory,'build_flight_runtime',lambda *args,**kwargs:Runtime())
    p=tmp_path/'derivation.json';p.write_text('{}');out=tmp_path/'work/codex/prep'
    def run():return m.prepare(tmp_path,tmp_path,p,out)
    if fault:
        with pytest.raises(ValueError):run()
        assert (out/'failed.json').exists() and not (out/'complete.json').exists()
    else:assert run()['model_calls']==0 and (out/'complete.json').exists()
    with pytest.raises(ValueError,match='fresh owned'):run()
    assert len(calls)==1


def test_real_c2_does_not_expose_facets():
    m.verify_output({'facets':{},'values':{'perception':{'a':[],'b':[]}}},['a','b'])
