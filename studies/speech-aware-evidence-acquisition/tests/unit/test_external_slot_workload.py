"""Single-slot supply comparison keeps the legacy prompt and delivering lock."""
import copy,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/analysis'))
from speech_aware_evidence_acquisition.rails import consumption_module as cm
from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks


def block():
    base=json.loads((ROOT/'configs/baseline/slurp-tfrl-hardset-a0-observation-r1-v1.json').read_text())
    b=copy.deepcopy(base['native_episode']['consumption_module'])
    b['columns']=[{'column_id':'learned','kind':'supply','node':'solver','max_supplied_items':56,'release_swap':{'store_id':'slot-contracts','release':'knowledge/fixture/release.json'}}]
    b['supply_frame']={'delivering_max_supplied_items':56,'twin':False,'fit_estimate':{'base_prompt_tokens':9600,'completion_ceiling':4096,'slot_tokens':65536,'tokens_per_supplied_item':120},'alternates':{'slot-contracts':[{'release':{'relpath':'knowledge/fixture/release.json','sha256':'1'*64},'snapshot':{'relpath':'knowledge/fixture/snapshot.json','sha256':'2'*64}}]}}
    return b


def test_knowledge_pair_retains_locked_prompt_and_has_exact_three_contacts():
    b=block();compiled=cm.compile_module(b,carrier_module_hooks(repo_root=ROOT),repo_root=ROOT)
    assert compiled.program_ceiling==3
    assert compiled.values()['native_episode']['prompt_contact_limits']==b['prompt_contact_limits']
    arms=compiled.values()['native_episode']['roles']['solver']['supply_variants']['arms']
    assert [a['arm_id'] for a in arms]==['learned','lock']
    assert arms[-1]['deliver'] is True and 'release_swap' not in arms[-1]


@pytest.mark.parametrize('change',['twin','override','bound','wrong-arm'])
def test_no_extra_fanout_or_routing_variation_admitted(change):
    b=block()
    if change=='twin':b['supply_frame']['twin']=True
    elif change=='override':b['columns'][0]['overrides']={'knowledge_stores.train-exemplars.phase_supply.solver.k':40}
    elif change=='bound':b['columns'][0]['max_supplied_items']=55
    else:b['columns'][0]['column_id']='unexpected'
    with pytest.raises(cm.ConsumptionModuleConfigError):cm.compile_module(b,carrier_module_hooks(repo_root=ROOT),repo_root=ROOT)


def test_slot_pair_enables_only_exact_declared_operational_custody():
    from speech_aware_evidence_acquisition.core.flight.native_recovery import slot_pair_layout_matches
    b=block();compiled=cm.compile_module(b,carrier_module_hooks(repo_root=ROOT),repo_root=ROOT)
    v=compiled.values();v['native_episode']['consumption_module']=b
    assert slot_pair_layout_matches(v)
    from speech_aware_evidence_acquisition.core.flight.native_recovery import native_contact_custody_enabled
    v.update(transport='vllm',carrier_lock_key='slurp',arm='raw-speech-agent-native-tools')
    assert native_contact_custody_enabled(v)
    v['native_episode']['roles']['solver']['supply_variants']['arms'][0]['release_swap']['release']='knowledge/changed.json'
    assert not slot_pair_layout_matches(v)
