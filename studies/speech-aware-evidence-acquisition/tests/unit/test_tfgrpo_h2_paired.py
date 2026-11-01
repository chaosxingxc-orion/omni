import copy
import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/analysis'))
import tfgrpo_h2_paired as m
from speech_aware_evidence_acquisition.rails import consumption_module as cm
from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks
from speech_aware_evidence_acquisition.core.flight.native_recovery import native_contact_custody_enabled
from test_tfgrpo_consumption import release,candidate,TARGET
from test_llama_server_transport import _spawn_live_matching_process
FIT={'slot_tokens':65536,'completion_ceiling':4096,'base_prompt_tokens':9600,'tokens_per_supplied_item':120}


def paired_block(params):
    base=json.loads((ROOT/'configs/baseline/slurp-tfrl-hardset-a0-observation-r1-v1.json').read_bytes())
    block=copy.deepcopy(base['native_episode']['consumption_module'])
    params=copy.deepcopy(params)
    params['target_arm']='learned'
    params['base_template']['relpath']=base['phase_material']['solver']['instruction_template']
    block['components'].append({'component_id':m.consumption.COMPONENT,'state':'on',
        'admitted_by':'development-evaluation','record':m.AUTH,'params':params})
    block['columns']=[{'column_id':'learned','kind':'template','node':'solver',
        'template':{'relpath':'configs/'+params['template']['relpath'],'sha256':params['template']['sha256']},'template_tokens':500}]
    block['template_frame']={'max_supplied_items':20,'delivering_template_tokens':400,'twin':False,'fit_estimate':FIT}
    return base,block


def test_pair_preserves_locked_solver_and_knowledge(release):
    _,_,params=release;base,block=paired_block(params)
    compiled=cm.compile_module(block,carrier_module_hooks(repo_root=ROOT),repo_root=ROOT)
    output=compiled.apply_to(base);output['native_episode']['consumption_module']=block
    assert compiled.program_ceiling==3
    assert output['phase_material']==base['phase_material']
    assert output['knowledge_stores']==base['knowledge_stores']
    arms=output['native_episode']['roles']['solver']['template_variants']['arms']
    assert [a['arm_id'] for a in arms]==['learned','lock']
    assert arms[1]['deliver'] is True and 'instruction_template' not in arms[1]
    assert native_contact_custody_enabled({**output,'transport':'vllm','carrier_lock_key':'slurp'})


@pytest.mark.parametrize('fault',['wrong_arm','twin','template','global_change','undeclared','other_variant'])
def test_paired_custody_is_narrow(release,fault):
    _,_,params=release;base,block=paired_block(params)
    values=cm.compile_module(block,carrier_module_hooks(repo_root=ROOT),repo_root=ROOT).apply_to(base)
    values.update(transport='vllm',carrier_lock_key='slurp')
    values['native_episode']['consumption_module']=block
    native=values['native_episode'];arms=native['roles']['solver']['template_variants']['arms']
    if fault=='wrong_arm':arms[0]['arm_id']='another'
    elif fault=='twin':arms.insert(1,{'arm_id':'aa','replicate_of':'lock'})
    elif fault=='template':arms[0]['instruction_template']['sha256']='0'*64
    elif fault=='global_change':values['phase_material']['solver']['instruction_template']='probes/changed.txt'
    elif fault=='undeclared':native['consumption_module']['components']=[]
    else:native['roles']['perception']['template_variants']={}
    assert not native_contact_custody_enabled(values)


def test_full_h2_derivation_is_held_and_disjoint(release,monkeypatch):
    _,data,params=release
    roster=json.loads((ROOT/m.ROSTER).read_bytes())
    params['consumer_ids']=[r['sample_id'] for r in roster['items']]
    base=json.loads((ROOT/'configs/baseline/slurp-tfrl-hardset-a0-observation-r1-v1.json').read_bytes())
    params['base_template']['relpath']=base['phase_material']['solver']['instruction_template']
    monkeypatch.setattr(m.consumption,'verify',lambda *a,**kw:None)
    result=m.derive(ROOT,data,params,fit_estimate=FIT,base_template_tokens=400,learned_template_tokens=500)
    assert result['evaluation']['normal_generation_calls']==936
    assert result['evaluation']['fallback_generation_cap']==1872
    assert result['experiment']['cells'][0]['readiness']['state']=='held'
    assert len(result['experiment']['populations'][0]['ids'])==312
    assert result['baseline']['native_contact_fit']['contact_cap']==936
    assert result['baseline']['phase_material']==base['phase_material']


@pytest.mark.parametrize('boundary',[1,2])
def test_actual_paired_graph_replays_known_contacts_then_continues(tmp_path,boundary):
    from dataclasses import replace
    from types import MappingProxyType
    from test_a0_native_recovery import harness
    from test_template_variants_live import _block,_builder,_arm_output,LOCKED
    from test_stage3_supply_variant_interleaving import _live_config
    from speech_aware_evidence_acquisition.rails.supply_variants import parse_template_variants
    from speech_aware_evidence_acquisition.core.contact_journal import NativeContactReplay
    from test_template_variants_delivery import _runtime,_raw_native
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
    from speech_aware_evidence_acquisition.core.bundle.native_replay import verify_native_replay
    declaration=_block(twin=False,candidates=1)
    declaration['arms'][0]['arm_id']='learned'
    group=parse_template_variants(declaration,role_kind='solver',phase_id='solver',locked_template=LOCKED)
    def build(path,**kwargs):
        if 'plan' not in kwargs:
            from test_rails_frozen_core_client import _plan
            kwargs['plan']=_plan(run_id='SAEA-E-native-role-runner-proof',attempt_id='attempt-native-role-runner-proof',planned_model_calls=3)
        h,fit,calls=harness(path,**kwargs)
        # The fixture program is a PhaseProgram with a contact_ceiling.
        from test_native_episode_runner import _program
        h.program=_program(max_contacts=3)
        fit.runtime._composed.values['native_contact_fit']['contact_cap']=3
        h.config=replace(_live_config(group=group,max_contacts=3),prompt_contact_limits='legacy-c17-lock-v1')
        h.template_variant_materials=MappingProxyType({'learned':_builder(h,'t1')})
        h._output=_arm_output
        h.slot_count=1
        return h,fit,calls
    old,oldfit,oldcalls=build(tmp_path/'old')
    emit=old.client._adapter._sink.emit;count=0
    def fail(channel,rid,payload):
        nonlocal count
        result=emit(channel,rid,payload)
        if channel=='MODEL_RESPONSE':
            count+=1
            if count==boundary:raise RuntimeError('known persisted contact boundary')
        return result
    old.client._adapter._sink.emit=fail
    try:
        with pytest.raises(Exception) as caught:old.run()
        assert count>=boundary,str(caught.value)
    finally:old.close()
    old.client._adapter._attempt_handle.mark_aborted('known stopped paired graph')
    plan=replace(old.plan,attempt_id='paired-resume-v1')
    replay=NativeContactReplay(old.world.data_root,old.plan,plan,old.client.arm)
    new,fit,calls=build(tmp_path/'new',plan=plan,data=old.world.data_root,replay=replay)
    try:
        summary=new.run()
        outputs=_runtime(new,summary,_raw_native(new,solver_role={'template_variants':declaration}))._canonical_output_rows(new.client)
    finally:cost=new.close()['cost']
    records={rid:p for channel,rid,p in new.trace.records}
    _verify_outputs_trace_binding(tuple(outputs),records)
    assert verify_native_replay({'arm':new.client.arm,'attempt':{'attempt_id':plan.attempt_id,'state_path':str(new.client._adapter._attempt_handle.path)},'cost':cost},records,plan)
    assert cost['calls_used']+cost['replayed_predecessor_cost']['calls']==3
    assert cost['replayed_predecessor_cost']['calls']>=boundary
    assert len(new.dispatched)==3-cost['replayed_predecessor_cost']['calls']
    assert len(fit.receipts)==3


@pytest.mark.parametrize('fault',[None,'base','learned','extra'])
def test_runtime_checks_paired_template_before_consumption(release,monkeypatch,fault):
    repo,data,params=release
    paired={**params,'target_arm':'learned'}
    values={'phase_material':{'solver':{'instruction_template':params['base_template']['relpath']}},
        'native_episode':{'consumption_module':{'components':[{'component_id':m.consumption.COMPONENT,'state':'on','params':paired}]},
            'roles':{'solver':{'template_variants':{'node':'solver','arms':[
                {'arm_id':'learned','instruction_template':{'relpath':'configs/'+params['template']['relpath'],'sha256':params['template']['sha256']}},
                {'arm_id':'lock','deliver':True}]}}}}}
    if fault=='base':values['phase_material']['solver']['instruction_template']=params['template']['relpath']
    elif fault=='learned':values['native_episode']['roles']['solver']['template_variants']['arms'][0]['instruction_template']['sha256']='0'*64
    elif fault=='extra':values['native_episode']['roles']['solver']['template_variants']['arms'].append({'arm_id':'extra'})
    if fault:
        with pytest.raises(ValueError):m.consumption.verify_for_flight(values,repo=repo,data=data,model_id='model',consumer_ids=[TARGET])
    else:assert m.consumption.verify_for_flight(values,repo=repo,data=data,model_id='model',consumer_ids=[TARGET])==candidate()['sha256']
