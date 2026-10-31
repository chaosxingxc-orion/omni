"""Real receipt mint and supply ownership guards for Fold A admission."""
import json
from pathlib import Path
import sys

import pytest

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/flights/nx'))
import fold_a_current_registration as registration
import fold_a_current_mint as mint
import fold_a_current_supply as supply
from test_fold_a_current_registration import prepared,source
from test_fold_a_current_agreement import put


def registered(prepared):
    repo,src,data,binding=prepared
    registration.register(repo,src,data,binding,data/'registration')
    return repo,data


def test_real_mint_and_no_overwrite(prepared):
    repo,data=registered(prepared);v=registration.evaluation
    dry=mint.admit(repo,data,True)
    assert dry['dry'] and not (data/v.DECISION_PATH).exists()
    result=mint.admit(repo,data)
    receipt=registration.read(data/v.DECISION_PATH)
    assert receipt['status']=='decided' and result['eligible_run_ids']==[v.RUN,v.FIT]
    with pytest.raises(SystemExit,match='overwrite'):mint.admit(repo,data)


@pytest.mark.parametrize('fault',['antecedent','config','attempt','proof'])
def test_mint_refuses_changed_registration(prepared,monkeypatch,fault):
    repo,data=registered(prepared);v=registration.evaluation
    if fault=='antecedent':
        p=repo/registration.PKG/'analysis-recipe-v1.json';p.write_bytes(p.read_bytes()+b' ')
    elif fault=='config':
        p=repo/f'configs/experiment/{v.CELL}.json';e=registration.read(p);e['failure_policy']='changed';put(p,e)
    elif fault=='attempt':
        p=repo/registration.PKG/'root-admission-draft-v1.json';d=registration.read(p);d['eligible_attempt_ids']=['wrong'];put(p,d)
    else:monkeypatch.setattr(registration,'accepted_proof',lambda *a:({}, {'changed':'b'*64}))
    with pytest.raises(ValueError):mint.admit(repo,data)
    assert not (data/v.DECISION_PATH).exists()


@pytest.mark.parametrize('fault',[None,'other-run','mutation','interruption'])
def test_supply_claims_own_fresh_path_and_preserves_failure(prepared,monkeypatch,fault):
    repo,data=registered(prepared);v=registration.evaluation
    mint.admit(repo,data)
    def fake_repin(**kwargs):
        assert kwargs['allow_output_change'] is False and kwargs['run_check_fn'] is None
        assert kwargs['experiments']==[(v.CELL,v.CELL)]
        name=f'prepared/{v.RUN if fault!="other-run" else "WRONG"}/phase-supply-v1.json'
        put(data/name,{'output':{'new_fold':'a'}})
        bp=repo/f'configs/baseline/{v.CELL}.json';b=registration.read(bp)
        b['prepared_phase_supply']={'relpath':name,'sha256':registration.sha(data/name)};put(bp,b)
        if fault=='interruption':raise RuntimeError('interrupted')
        if fault=='mutation':(data/v.DECISION_PATH).write_bytes((data/v.DECISION_PATH).read_bytes()+b' ')
        return {'checks':[]}
    monkeypatch.setattr(supply,'repin_package',fake_repin)
    monkeypatch.setattr(supply,'rewrite_dependent_hashes',lambda **kw:[])
    out=data/'supply-out'
    if fault:
        with pytest.raises((ValueError,RuntimeError)):supply.prepare(repo,data,data/'lock',out)
        assert (out/'failed.json').exists() and not (out/'complete.json').exists()
    else:
        result=supply.prepare(repo,data,data/'lock',out)
        assert result['launchable'] is False
        assert result['supply']['relpath'].startswith('prepared/'+v.RUN+'/')
        for kind in ('generation','fit'):
            assert registration.read(repo/registration.PKG/(kind+'-plan-registered-v1.json'))['prepared_phase_supply']==result['supply']
    with pytest.raises(ValueError):supply.prepare(repo,data,data/'lock',data/'retry')
