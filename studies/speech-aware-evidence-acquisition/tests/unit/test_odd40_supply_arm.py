import pytest
from test_consumption_module_supply_columns import _supply_block,_compose,EX_K
from speech_aware_evidence_acquisition.rails.supply_variants import (
    ODD40_OVERRIDE_PATH as PATH,ODD40_OVERRIDE_VALUE as VALUE,
    parse_supply_variant_knob,supply_variant_arm_record,SupplyVariantArm,SupplyVariantError)


def test_explicit_b_compiles_while_aa_and_lock_remain_original():
    prefix='knowledge_stores.train-exemplars.phase_supply.solver.'
    overrides={PATH:VALUE,EX_K:40,**{prefix+'fuse_coverage.'+k:40 for k in ('m','c','top_l')}}
    block=_supply_block(alternates=False,columns=[{'column_id':'b','kind':'supply','node':'solver',
        'max_supplied_items':55,'overrides':overrides}])
    compiled,values=_compose(block)
    arms=values['native_episode']['roles']['solver']['supply_variants']['arms']
    assert compiled.program_ceiling==4
    assert arms[0]['overrides']==overrides
    assert [a['arm_id'] for a in arms]==['b','aa','lock']
    assert all('overrides' not in a for a in arms[1:])


@pytest.mark.parametrize('path,value',[(PATH,'unknown'),(PATH,40),(PATH.replace('train-exemplars','intent-boundaries'),VALUE),
    (PATH.replace('solver','perception'),VALUE),(PATH.replace('fuse_policy','rank_policy'),VALUE)])
def test_no_arbitrary_policy_override(path,value):
    with pytest.raises(SupplyVariantError):
        parse_supply_variant_knob(path,value,phase_id='solver',first_phase_id='perception',declared_stores=['train-exemplars','intent-boundaries'])


def test_receipt_retains_exact_named_policy():
    knob=parse_supply_variant_knob(PATH,VALUE,phase_id='solver',first_phase_id='perception',declared_stores=['train-exemplars'])
    arm=SupplyVariantArm(arm_id='b',deliver=False,knobs=(knob,),max_supplied_items=20)
    record=supply_variant_arm_record(arm=arm,arm_index=0,contact_id='b',supplied_value_ids=['value:1'])
    assert record['resolved_knobs']==[{'path':PATH,'value':VALUE}]


def test_complete_receipt_accepts_exact_policy_but_rejects_unknown_string():
    import copy
    from test_supply_variant_multistore_swaps import _parse,_records,_validate,FIT
    from speech_aware_evidence_acquisition.rails.supply_variants import supply_variant_receipt
    group=_parse({'arm_visits':'first','fit_estimate':FIT,'arms':[
        {'arm_id':'b','max_supplied_items':40,'overrides':{PATH:VALUE,EX_K:40}},
        {'arm_id':'lock','deliver':True,'max_supplied_items':40}]})
    receipt=supply_variant_receipt(group=group,phase_id='solver',records=_records(group))
    _validate(receipt,group)
    bad=copy.deepcopy(receipt)
    for entry in bad['arms'][0]['resolved_knobs']:
        if entry['path']==PATH:entry['value']='unknown-policy'
    with pytest.raises(ValueError):_validate(bad,group)


@pytest.mark.parametrize('fault',[None,'k','weights','coverage'])
def test_odd40_requires_complete_matching_recall(fault):
    from test_phase_supply import _fragment,_program,_source,_SHA_A
    from speech_aware_evidence_acquisition.rails.phase_supply import load_phase_supply_specs,PhaseSupplyError
    kwargs=dict(fuse_policy=VALUE,k=40,tau=0,rank_weights={'w_cos':1.,'w_prior':0.,'w_agree':0.},
                fuse_coverage={'m':40,'c':40,'top_l':40})
    if fault=='k':kwargs['k']=20
    elif fault=='weights':kwargs['rank_weights']['w_prior']=1.
    elif fault=='coverage':kwargs['fuse_coverage']['m']=20
    fragment=_fragment(_source('v/q.jsonl',_SHA_A),**kwargs)
    if fault:
        with pytest.raises(PhaseSupplyError):load_phase_supply_specs(fragment,_program())
    else:assert load_phase_supply_specs(fragment,_program())['evidence'].k==40
