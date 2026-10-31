import copy
import hashlib
import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/flights/nx'))
import tfrl_v2_sampler as sampler
from speech_aware_evidence_acquisition.rails import trajectory_seeds as seeds
from speech_aware_evidence_acquisition.rails.native_episode_receipt import decoding_override_error
from speech_aware_evidence_acquisition.rails.supply_variants import parse_decoding_variants, SupplyVariantError
from test_decoding_variants import _block


def plan():
    return sampler.build_plan({'roster_id':sampler.cm.HALF_ROSTER_ID['R01C-H1'], 'split_identity_hash':'a'*64,
        'items':[{'sample_id':f'fixture:{i}','audio_seconds':2.} for i in range(2)]}, 'R01C-H1',1)


def group():
    p=plan(); columns=sampler.execution_columns(p); b=_block()
    b['arms']=[{'arm_id':c['column_id'],'decoding':c['decoding']} for c in columns]+[
        {'arm_id':'aa','replicate_of':'lock'},{'arm_id':'lock','deliver':True}]
    return p,b,parse_decoding_variants(b,role_kind='solver',phase_id='solver')


def test_runtime_seeds_match_existing_plan_hash_formula():
    p,b,g=group()
    for row in p['row_plan']:
        actual=[seeds.resolve(g.arm(f's{k+1}').decoding,row['row_id']) for k in range(4)]
        assert [r['seed'] for r in actual]==[c['seed'] for c in row['contacts'][1:]]
        assert len({r['seed'] for r in actual})==4
        assert len({json.dumps({k:v for k,v in r.items() if k!='seed'},sort_keys=True) for r in actual})==1
        for k,r in enumerate(actual):
            text=f"tfrl-v2-seed|{p['iteration_seed']}|{row['row_id']}|{k}"
            assert r['seed']==int(hashlib.sha256(text.encode()).hexdigest()[:8],16)%(2**31-2)+1
            assert not (set(r)&seeds.RECIPE_KEYS)
    assert g.arm('aa').decoding is None and g.arm('lock').decoding is None


@pytest.mark.parametrize('fault',['missing','duplicate','top_p','base','greedy','bool','seed','partial_group'])
def test_malformed_or_non_seed_only_groups_refused(fault):
    p,b,g=group(); d=b['arms'][0]['decoding']
    if fault=='missing':d.pop('trajectory_index')
    elif fault=='duplicate':d['trajectory_index']=1
    elif fault=='top_p':d['top_p']=0.8
    elif fault=='base':d['trajectory_seed_base']+=1
    elif fault=='greedy':d['temperature']=0
    elif fault=='bool':d['trajectory_index']=True
    elif fault=='seed':d['seed']=123
    else:b['arms'].pop(0)
    with pytest.raises(SupplyVariantError):parse_decoding_variants(b,role_kind='solver',phase_id='solver')


def test_plan_tamper_refused_before_compilation():
    p=plan();p['row_plan'][0]['contacts'][1]['seed']+=1
    with pytest.raises(sampler.SamplerError):sampler.execution_columns(p)


def test_legacy_seed_override_is_still_refused():
    assert decoding_override_error({'seed':1}) is not None
    assert seeds.resolve({'temperature':0.7},'row')=={'temperature':0.7}


def test_plan_columns_compile_load_and_reject_changed_method():
    from test_decoding_variants import _decoding_block
    from speech_aware_evidence_acquisition.rails import consumption_module as cm
    from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks
    from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
    from speech_aware_evidence_acquisition.rails.native_episode_config import load_native_episode_config
    block=_decoding_block();block['columns']=sampler.execution_columns(plan())
    block['prompt_contact_limits']='legacy-c17-lock-v1'
    compiled=cm.compile_module(block,carrier_module_hooks())
    values=compiled.values();values['carrier_lock_key']='slurp'
    values['native_episode']['consumption_module']=block
    parsed=load_native_episode_config(values,default_phase_program(values['policy']))
    assert compiled.program_ceiling==parsed.max_contacts==7
    assert parsed.supply_variant_group.arm_ids==('s1','s2','s3','s4','aa','lock')
    cm.verify_declared_module(block,values)
    values['native_episode']['roles']['solver']['decoding_variants']['arms'][0]['decoding']['trajectory_index']=1
    with pytest.raises(cm.ConsumptionModuleConfigError):cm.verify_declared_module(block,values)
