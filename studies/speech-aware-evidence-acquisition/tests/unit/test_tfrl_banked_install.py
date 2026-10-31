import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/analysis')]
import tfrl_banked_install as m


@pytest.mark.parametrize('iteration',[2,3])
@pytest.mark.parametrize('fault',[None,'stop','source','hash','label','guard','spent','existing'])
def test_install_derives_current_source_and_checks_both_halves_before_write(tmp_path,monkeypatch,iteration,fault):
    import speech_aware_evidence_acquisition.contracts as contracts
    import speech_aware_evidence_acquisition.core.bundle.git_commit as git
    import tfrl_current_next_plan as planner
    import tfrl_banked_supply as supply
    monkeypatch.setattr(contracts,'assert_execution_environment',lambda:None)
    monkeypatch.setattr(git,'require_clean_study_repo',lambda p:'a'*40)
    repo=tmp_path/'repo';repo.mkdir();(repo/'configs').mkdir();(repo/'docs').mkdir()
    data=tmp_path/'data';out=data/'work/codex/install';calls=[];validated=[]
    def write(path,doc):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(doc))
    def decide(repo,data,path):
        calls.append('decision');doc={'decision':'stop-family' if fault=='stop' else 'prepare-held-successor',
                                     'construction_gain_pp':.4 if fault=='stop' else 1.2}
        write(path,doc);return doc
    monkeypatch.setattr(planner,'decide',decide)
    def stage(repo,data,path,iteration):
        calls.append('stage');hashes={};plans={}
        for h,n in [('h1',313),('h2',312)]:
            name=f'slurp-tfrl-current-bank-i{iteration}-{h}-v1'
            for kind in ('baseline','experiment'):
                rel=f'configs/{kind}/{name}.json'
                doc={} if kind=='baseline' else {'stage3_algorithm':{'id':'shared-rollout' if fault=='label' else 'tfrl-v2',
                    'operators_used':['rollout','scoring','context_application']},
                    'cells':[{'readiness':{'state':'held'},'allowed_execution_stages':['check']}]}
                write(path/rel,doc);hashes[rel]=m.sha(path/rel)
            write(path/f'{h}-sampling-plan.json',{'rows':n});plans[h]=m.sha(path/f'{h}-sampling-plan.json')
        if fault=='hash':hashes[next(iter(hashes))]='b'*64
        result={'schema':'saea-tfrl-current-next-held-plans-v1','source_commit':'b'*40 if fault=='source' else 'a'*40,
            'iteration':iteration,'rows':625,'readiness':'held','new_model_calls':0,
            'generation_authorized_by_this_file':False,'construction_gate':{'construction_gain_pp':1.2},
            'held_cell_files':hashes,'plans':plans}
        write(path/'complete.json',result);return result
    monkeypatch.setattr(planner,'run',stage)
    def guard(overlay,data,name,it,decision):
        assert overlay!=repo and (overlay/f'configs/baseline/{name}.json').exists()
        assert (overlay/f'docs/readiness/{name}/sampling-plan.json').exists()
        assert not (repo/f'configs/baseline/{name}.json').exists()
        validated.append(name)
        if fault=='guard' and name.endswith('h2-v1'):raise ValueError('bad second half')
    monkeypatch.setattr(supply,'validate_preparation_inputs',guard)
    if fault=='spent':(data/f'prepared/SAEA-SLURP-TFRL-CURRENT-BANK-I{iteration}-H1-V1').mkdir(parents=True)
    if fault=='existing':write(repo/f'configs/baseline/slurp-tfrl-current-bank-i{iteration}-h1-v1.json',{})
    before={str(p.relative_to(repo)):p.read_bytes() for p in repo.rglob('*') if p.is_file()}
    call=lambda:m.install(repo,data,out,iteration)
    if fault not in (None,'stop'):
        with pytest.raises(ValueError):call()
        after={str(p.relative_to(repo)):p.read_bytes() for p in repo.rglob('*') if p.is_file()}
        assert after==before
        if fault in ('spent','existing'):assert calls==[]
        else:assert (out/'failed.json').exists()
        return
    result=call()
    if fault=='stop':
        assert calls==['decision'] and result['installed_files']==0
        assert not list((repo/'configs').rglob('*.json'))
    else:
        assert calls==['decision','stage'] and len(validated)==2
        assert result['rows']==625 and len(result['output_hashes'])==6
        for rel,digest in result['output_hashes'].items():assert m.sha(repo/rel)==digest
        assert result['readiness']=='held' and result['registered'] is False
    with pytest.raises(ValueError,match='fresh owned'):call()
