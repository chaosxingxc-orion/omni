"""The current Fold A coordinator checks admission and never replays contacts."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/flights/nx'))
import fold_a_current_campaign as coordinator
from test_fold_a_current_registration import prepared,source
from test_fold_a_current_agreement import put
from fold_a_current_mint import admit


@pytest.fixture
def ready(prepared):
    repo,src,data,binding=prepared;r=coordinator.registration;v=r.evaluation
    r.register(repo,src,data,binding,data/'registered');admit(repo,data)
    ep=repo/f'configs/experiment/{v.CELL}.json';e=r.read(ep);e['cells'][0]['readiness']['state']='ready';put(ep,e)
    sp=data/f'prepared/{v.RUN}/phase-supply-v1.json';put(sp,{'output':{'fold':'a'}})
    bp=repo/f'configs/baseline/{v.CELL}.json';b=r.read(bp)
    b['prepared_phase_supply']={'relpath':str(sp.relative_to(data)),'sha256':r.sha(sp)};put(bp,b)
    lock=data/'lock.json';put(lock,{})
    return repo,data,lock


def test_real_read_registration_and_command_route(ready):
    repo,data,lock=ready
    plan=coordinator.validate(repo,data)
    commands=coordinator.commands(repo,data,lock,plan)
    assert [n for n,_ in commands]==['registration','preflight','campaign']
    assert '--read-only' in commands[1][1]
    assert 'speech_aware_evidence_acquisition.core.flight.cli' in commands[2][1]
    assert '--expected-session-id' in commands[2][1]


@pytest.mark.parametrize('fault',['held','roster','decision','spent','fit-spent','supply','read-pin'])
def test_precontact_validation_refusal(ready,fault):
    repo,data,lock=ready;r=coordinator.registration;v=r.evaluation
    ep=repo/f'configs/experiment/{v.CELL}.json';e=r.read(ep)
    if fault=='held':e['cells'][0]['readiness']['state']='held';put(ep,e)
    elif fault=='roster':e['populations'][0]['ids'].reverse();put(ep,e)
    elif fault=='decision':
        p=data/v.DECISION_PATH;d=r.read(p);d['eligible_attempt_ids']=['wrong'];put(p,d)
    elif fault=='spent':(data/'attempts'/v.RUN).mkdir(parents=True)
    elif fault=='fit-spent':(data/'prepared'/v.FIT).mkdir(parents=True)
    elif fault=='supply':(data/f'prepared/{v.RUN}/phase-supply-v1.json').write_text('{}')
    else:
        p=repo/r.PKG/'analysis-recipe-v1.json';p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError):coordinator.validate(repo,data)


@pytest.mark.parametrize('failure',[None,'preflight','campaign','fit','input-drift','source-drift','lease'])
def test_coordinator_stops_without_replay_and_preserves_terminal_evidence(ready,monkeypatch,failure):
    repo,data,lock=ready;r=coordinator.registration
    calls=[];head=['source-head']
    def run(args,**kwargs):
        name=Path(kwargs['stdout'].name).stem;calls.append(name)
        if name=='preflight':
            if failure=='input-drift':lock.write_bytes(lock.read_bytes()+b' ')
            elif failure=='source-drift':head[0]='wrong'
            elif failure=='lease':put(data/'locks/flight.lock',{})
        return SimpleNamespace(returncode=int(name==failure))
    monkeypatch.setattr(coordinator,'subprocess',SimpleNamespace(run=run,STDOUT=-2,
        check_output=lambda args,**kw:head[0] if 'rev-parse' in args else b''))
    def fit(*args):
        if failure=='fit':raise ValueError('incomplete FIT')
        return {'rows':1080,'status':'PASS'}
    monkeypatch.setattr(coordinator.fit,'build',fit)
    monkeypatch.setattr(coordinator.delivered,'build',lambda *args:{'rows':1080,'aggregate':{'synthetic':True}})
    out=data/'campaign'
    if failure:
        with pytest.raises((ValueError,RuntimeError)):coordinator.execute(repo,data,lock,out,'source-head')
        assert (out/'failed.json').exists() and not (out/'complete.json').exists()
    else:
        result=coordinator.execute(repo,data,lock,out,'source-head')
        assert result['rows']==1080 and result['claims_system_benefit'] is False
    if failure in ('preflight','input-drift','source-drift','lease'):assert calls==['registration','preflight']
    else:assert calls==['registration','preflight','campaign']
    before=list(calls)
    with pytest.raises(ValueError,match='fresh external'):coordinator.execute(repo,data,lock,out,'source-head')
    assert calls==before
