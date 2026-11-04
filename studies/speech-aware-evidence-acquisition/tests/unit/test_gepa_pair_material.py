"""Source routing seams; native provenance gates are deliberately not bypassed."""
import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parents[2]/'scripts/analysis'))
import gepa_pair_material as m
from speech_aware_evidence_acquisition.rails import gepa_mutation_source as source
from speech_aware_evidence_acquisition.rails import gepa_mutation_runtime as runtime
from speech_aware_evidence_acquisition.core import bundle


@pytest.fixture
def setup(tmp_path, monkeypatch):
    live, seed, mutation = [tmp_path/n for n in ('live', 'seed', 'mutation')]
    for root in (live, seed, mutation):
        (root/'configs').mkdir(parents=True)
        (root/'configs/solver.txt').write_text('original')
    sp, mp = {'ref':'seed'}, {'ref':'mutation'}
    seen = []
    def resolve(spec, *, repo, data):
        pin=spec.get('historical_repository')
        return {None:live, 'seed':seed, 'mutation':mutation}[pin['ref'] if pin else None]
    monkeypatch.setattr(source, 'source_repository', resolve)
    checked={'document':{'fit_estimate':{},'max_supplied_items':1,'source_pins':{}},
        'contract':{'contract_sha256':'contract'},'parent':{'candidate_sha256':'parent'},
        'runtime_receipt':{'model':{'model_dir':'tokenizer'}},'tokenizer_identity':{'t':1},
        'source_baseline':{'phase_material':{'solver':{'instruction_template':'solver.txt'}}},
        'model':{},'partition':{},'source_pins':{}}
    monkeypatch.setattr(source, 'seed_contract', lambda *a,**k: checked if k['repo']==seed else pytest.fail('wrong seed tree'))
    def read_pin(repo,data,pin):
        seen.append(('manifest',repo));return mutation/'manifest',{}
    monkeypatch.setattr(m.seed,'read_pin',read_pin)
    monkeypatch.setattr(bundle,'verify_bundle',lambda path:seen.append(('bundle',path)) or 'verified')
    def verify(b,*,repo,data):
        assert b=='verified';seen.append(('mutation',repo))
        return {'contract_sha256':'contract','parent_sha256':'parent','mutation_text':'text','response_sha256':'response'}
    monkeypatch.setattr(runtime,'verify_mutation_bundle',verify)
    monkeypatch.setattr(m,'load_text_token_counter',lambda _: (len,{'t':1}))
    monkeypatch.setattr(m.prompt,'derive_candidate',lambda *a,**k:{'candidate_sha256':'child'})
    monkeypatch.setattr(m,'compile_template_pair',lambda repo,*a,**k:seen.append(('compile',repo)) or {})
    args=dict(seed_material={'root':'repo'},mutation_manifest={},historical_repository=sp,mutation_repository=mp)
    return live,tmp_path,args,seen,checked


def test_distinct_seed_mutation_and_live_tree(setup):
    live,data,args,seen,_=setup
    result=m.derive(live,data,**args)
    assert ('manifest',data/'mutation') in seen and ('mutation',data/'mutation') in seen
    assert ('bundle',data/'mutation/manifest') in seen and ('compile',live) in seen
    assert result['parent']['candidate_sha256']=='parent'


def test_legacy_mutation_repository_defaults_to_live(setup):
    live,data,args,seen,_=setup;args.pop('mutation_repository')
    m.derive(live,data,**args)
    assert ('mutation',live) in seen and ('manifest',live) in seen


@pytest.mark.parametrize('fault',['contract','parent','template','native_rejected'])
def test_source_separation_preserves_rejections(setup,monkeypatch,fault):
    live,data,args,seen,checked=setup
    if fault=='contract':checked['contract']['contract_sha256']='changed'
    elif fault=='parent':checked['parent']['candidate_sha256']='changed'
    elif fault=='template':(live/'configs/solver.txt').write_text('changed')
    else:
        def reject(*a,**k):raise ValueError('native source custody rejected')
        monkeypatch.setattr(runtime,'verify_mutation_bundle',reject)
    with pytest.raises(ValueError):m.derive(live,data,**args)
    assert not any(k=='compile' for k,_ in seen)
