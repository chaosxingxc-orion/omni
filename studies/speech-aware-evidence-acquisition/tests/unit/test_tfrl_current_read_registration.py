import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/analysis'),str(ROOT/'scripts/flights/nx')]
import tfrl_current_read_registration as m


@pytest.mark.parametrize('iteration',[1,2,3])
@pytest.mark.parametrize('fault',[None,'source','gate','missing_binding'])
def test_whole_half_read_registration_preserves_iteration_and_source(tmp_path,monkeypatch,iteration,fault):
    import speech_aware_evidence_acquisition.contracts as contracts
    import speech_aware_evidence_acquisition.core.bundle.git_commit as git
    import tfrl_banked_supply as supply
    monkeypatch.setattr(contracts,'assert_execution_environment',lambda:None)
    monkeypatch.setattr(git,'require_clean_study_repo',lambda p:'a'*40)
    repo=tmp_path/'repo';data=tmp_path/'data'
    def put(path,doc):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(doc))
    pins={};prefix='seed' if iteration==1 else 'bank'
    for half,n in [('h1',313),('h2',312)]:
        name=f'slurp-tfrl-current-{prefix}-i{iteration}-{half}-v1'
        run=f'SAEA-SLURP-TFRL-CURRENT-{prefix.upper()}-I{iteration}-{half.upper()}-V1'
        rp=repo/f'docs/rosters/{half}.json';put(rp,{'items':[{'sample_id':str(i)} for i in range(n)]})
        exp={'cells':[dict(run_id=run,baseline_fragment=name,attempt_id='generation-v1',
            readiness={'state':'held'},allowed_execution_stages=['check'])],
            'population':dict(roster_receipt=str(rp.relative_to(repo)),count=n,split_identity_hash=half)}
        ep=repo/f'configs/experiment/{name}.json';bp=repo/f'configs/baseline/{name}.json'
        supply_path=data/f'prepared/{run}/phase-supply-v1.json';put(supply_path,{'output':{}})
        put(ep,exp);put(bp,{'prepared_phase_supply':{'relpath':str(supply_path.relative_to(data)),'sha256':m.sha(supply_path)}})
        pins.update({str(p.relative_to(repo)):m.sha(p) for p in (ep,bp)})
        put(repo/f'docs/readiness/{name}/sampling-plan.json',{'run_id':run,'iteration':iteration,
            'roster':{'rows':n,'file_sha256':m.sha(rp)}})
    ledger=repo/'docs/exposure-ledger.md';ledger.write_text('')
    if fault=='source':pins[next(iter(pins))]='b'*64
    if fault=='missing_binding':pins={}
    binding=data/'work/codex/binding.json';put(binding,{'schema':'saea-tfrl-current-supply-binding-v1',
        'rows':625,'new_model_calls':0,'readiness':'held','output_hashes':pins})
    decision=data/'work/codex/decision.json';put(decision,{'decision':'pass'})
    checks=[]
    def guard(*args,**kwargs):
        checks.append(args[2]);assert kwargs['allow_bound'] is True
        if fault=='gate':raise ValueError('gate failed')
    monkeypatch.setattr(supply,'validate_preparation_inputs',guard)
    out=data/'work/codex/registered'
    call=lambda:m.register(repo,data,binding,out,iteration,decision)
    should_fail=fault=='source' or (iteration>1 and fault in ('gate','missing_binding'))
    if should_fail:
        with pytest.raises(ValueError):call()
        assert ledger.read_text()=='' and not out.exists()
        return
    result=call();assert len(result['recipes'])==2 and result['new_scoring_jobs']==0
    assert len(checks)==(0 if iteration==1 else 2)
    for rel in result['recipes']:
        recipe=json.loads((repo/rel).read_bytes())
        assert recipe['official_scoring_jobs']==7 and recipe['claims_benefit'] is False
        if iteration>1:assert recipe['iteration']==iteration and 'cross-fit' in recipe['request_gate']
    with pytest.raises(ValueError,match='fresh owned'):call()
