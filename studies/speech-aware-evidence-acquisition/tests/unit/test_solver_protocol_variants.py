import copy,json
from pathlib import Path
import pytest
from speech_aware_evidence_acquisition.rails import consumption_module as cm
from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks
from speech_aware_evidence_acquisition.rails.slurp_solver_protocol import FORMAT_ID,response_format
from speech_aware_evidence_acquisition.rails.supply_variants import parse_protocol_variants,supply_variant_arm_record,supply_variant_receipt
from speech_aware_evidence_acquisition.rails.native_episode_receipt import validate_supply_variant_receipt
from test_llama_server_transport import _spawn_live_matching_process
ROOT=Path(__file__).parents[2]


def block():
    b=copy.deepcopy(json.loads((ROOT/'configs/baseline/slurp-tfrl-hardset-a0-observation-r1-v1.json').read_text())['native_episode']['consumption_module'])
    b['columns']=[{'column_id':'constrained','kind':'protocol','node':'solver','response_format_id':FORMAT_ID}]
    b['protocol_frame']={'max_supplied_items':56,'template_tokens':500,'twin':False,'fit_estimate':{'base_prompt_tokens':9600,'completion_ceiling':4096,'slot_tokens':65536,'tokens_per_supplied_item':120}}
    return b


def compiled():
    b=block();c=cm.compile_module(b,carrier_module_hooks(repo_root=ROOT),repo_root=ROOT)
    v=c.values();v['native_episode']['consumption_module']=b
    return c,v


def test_configured_protocol_pair_has_three_contacts_and_custody():
    from speech_aware_evidence_acquisition.core.flight.native_recovery import protocol_pair_layout_matches,native_contact_custody_enabled
    c,v=compiled();assert c.program_ceiling==3
    assert protocol_pair_layout_matches(v)
    v.update(transport='vllm',carrier_lock_key='slurp',arm='raw-speech-agent-native-tools')
    assert native_contact_custody_enabled(v)


@pytest.mark.parametrize('fault',['sampling','template','twin','format'])
def test_protocol_does_not_admit_other_axes(fault):
    b=block()
    if fault=='sampling':b['columns'][0]['decoding']={'temperature':1}
    elif fault=='template':b['columns'][0]['instruction_template']='bad'
    elif fault=='twin':b['protocol_frame']['twin']=True
    else:b['columns'][0]['response_format_id']='unknown'
    with pytest.raises(cm.ConsumptionModuleConfigError):cm.compile_module(b,carrier_module_hooks(repo_root=ROOT),repo_root=ROOT)


def test_protocol_receipt_binds_format_and_identical_evidence():
    c,v=compiled();g=parse_protocol_variants(v['native_episode']['roles']['solver']['protocol_variants'],role_kind='solver',phase_id='solver')
    records=[supply_variant_arm_record(arm=a,arm_index=i,contact_id='c'+str(i),supplied_value_ids=['x'],protocol_group=True) for i,a in enumerate(g.arms)]
    receipt=supply_variant_receipt(group=g,phase_id='solver',records=records)
    validate_supply_variant_receipt(receipt,arm_ids=g.arm_ids,delivered_arm_id='lock',phase_id='solver',arm_visits='first')
    bad=copy.deepcopy(receipt);bad['arms'][0]['response_format_id']='other'
    with pytest.raises(ValueError):validate_supply_variant_receipt(bad,arm_ids=g.arm_ids,delivered_arm_id='lock',phase_id='solver')


def test_real_graph_requests_differ_only_by_protocol(tmp_path,monkeypatch):
    pytest.importorskip('openjiuwen')
    import test_native_episode_runner as nr
    from test_decoding_variants_live import _decoding_harness,_run,_requests
    c,v=compiled();g=parse_protocol_variants(v['native_episode']['roles']['solver']['protocol_variants'],role_kind='solver',phase_id='solver')
    fmt=response_format({'audio':['volume_up']},['person'])
    original=nr.NativeEpisodeRunBinding
    monkeypatch.setattr(nr,'NativeEpisodeRunBinding',lambda **kw:original(**kw,protocol_response_formats={FORMAT_ID:fmt}))
    h=_decoding_harness(tmp_path,g);summary,close=_run(h)
    assert close['request_count']==3
    p,a,b=_requests(h)
    assert 'response_format' not in p['decoding_params']
    assert a['decoding_params']=={**b['decoding_params'],'response_format':fmt}
    from speech_aware_evidence_acquisition.rails.solver_protocol_audit import verify_pair
    assert verify_pair(a,b,expected_format=fmt)['same_input_except_protocol']
    for key in ('task_instruction','supplied_evidence','speech_feature_parts','speech_packet_sha256','tool_schemas','evidence_provenance'):
        assert a[key]==b[key]
    assert json.loads(a['history'][-1]['content'])['messages']==json.loads(b['history'][-1]['content'])['messages']
    assert summary.results[0].supply_variants['schema']=='saea-supply-variant-protocol-v1'
    from test_template_variants_delivery import _runtime,_raw_native
    runtime=_runtime(h,summary,_raw_native(h,solver_role={'protocol_variants':v['native_episode']['roles']['solver']['protocol_variants']}))
    outputs=runtime._canonical_output_rows(h.client)
    assert outputs[0]['supply_variants']['schema']=='saea-supply-variant-protocol-v1'
    h.config=__import__('dataclasses').replace(h.config,prompt_contact_limits='legacy-c17-lock-v1')
    from speech_aware_evidence_acquisition.rails.native_episode_config import validate_prompt_contact_limits
    validate_prompt_contact_limits(h.config)



def test_answer_only_reask_preserves_protocol_and_sampling():
    from speech_aware_evidence_acquisition.core.reasoning_knobs import answer_only_decoding
    f=response_format({'audio':['volume_up']},['person'])
    d={'max_tokens':4096,'thinking_budget_tokens':3072,'chat_template_kwargs':{'enable_thinking':True},'response_format':f,'temperature':0,'seed':7}
    after=answer_only_decoding(d)
    assert after['response_format']==f and after['temperature']==0 and after['seed']==7
    assert after['max_tokens']==1024 and after['chat_template_kwargs']=={'enable_thinking':False}


def test_pair_audit_rejects_hidden_prompt_or_sampling_change():
    from speech_aware_evidence_acquisition.rails.solver_protocol_audit import verify_pair
    b={'request_id':'b','history':[],'decoding_params':{'seed':7},'task_instruction':'same'}
    fmt=response_format({'audio':['volume_up']},['person'])
    a=copy.deepcopy(b);a['request_id']='a';a['decoding_params']['response_format']=fmt
    assert verify_pair(a,b,expected_format=fmt)
    a['task_instruction']='different'
    with pytest.raises(ValueError):verify_pair(a,b,expected_format=fmt)


def test_official_reader_dispatches_protocol_and_verifies_wire(tmp_path,monkeypatch):
    import test_decoding_variants_score as old
    from speech_aware_evidence_acquisition.rails.supply_variants import supply_variant_arm_record as actual_record
    from unittest.mock import patch
    c,v=compiled();b=v['native_episode']['roles']['solver']['protocol_variants']
    g=old.mod.parse_solver_arm_group(v,v['native_episode']['roles']['solver'],{'phase_id':'perception'}, {})
    assert g.is_protocol
    def record(**kw):
        kw.pop('decoding_group',None);return actual_record(**kw,protocol_group=True)
    with patch.object(old,'supply_variant_arm_record',record):
        bundle=old._bundle(tmp_path,g)
    from speech_aware_evidence_acquisition.rails import native_episode_receipt as nr
    from types import SimpleNamespace
    monkeypatch.setattr(nr,'validate_native_episode_delivery_receipt',lambda episode,**kw:SimpleNamespace(kind=episode['kind']))
    # The old fixture has no actual request payload: fail closed rather than
    # silently treating receipt labels as proof of equal model inputs.
    with pytest.raises(old.mod.Stage3ScoreError,match='wire equality'):
        old._render(bundle,g,tmp_path)

    fmt=response_format({'audio':['volume_up']},['person'])
    for arm in g.arms:
        key=f'{old.IDS[0]}-0-{arm.arm_id}/request'
        payload={'request_id':key,'history':[],'decoding_params':{'seed':7},'task_instruction':'same'}
        if arm.response_format_id:payload['decoding_params']['response_format']=fmt
        bundle.trace_records[key]['payload']=payload
    manifest=old.mod.render_supply_variant_first_visit(bundle,g,{old.IDS[0]:'alarm'},tmp_path/'analysis',
        run_id='RUN',attempt_id='ATT',roster_sha256='1'*64,config_hash='c'*64,expected_protocol_format=fmt)
    assert set(manifest['copies'])=={'constrained','lock'}


def test_real_development_config_reuses_exact_frozen_eighty():
    from solver_protocol_workload import ProtocolWorkload
    cfg=json.loads((ROOT/'configs/probes/solver-protocol/development80.json').read_text())
    w=ProtocolWorkload(cfg);material,checked=w.material_checked(ROOT,ROOT,cfg['material'])
    assert len(checked['roster']['items'])==80
    assert [x['sample_id'] for x in checked['roster']['items']]==cfg['sample_ids']
    assert material['frame']['template_tokens']==208


def test_actual_adapter_fallback_keeps_label_protocol(tmp_path):
    import test_thinking_transport_and_native_knobs as t
    from speech_aware_evidence_acquisition.rails.solver_protocol_audit import verify_contact_pair
    fmt=response_format({'audio':['volume_up']},['person'])
    a,transport,sunk=t._native(tmp_path,[('', 'length',4096),('answer','stop',20)],**{**t.KNOBS,'response_format':fmt})
    buffer=t.TraceBuffer('SAEA-E-TEST-native-tools-client');completion=t._dispatch(a,buffer)
    assert len(transport.calls)==2 and all(r['decoding_params']['response_format']==fmt for r in transport.calls)
    receipt,=t._receipts(buffer)
    records={rid:payload for channel,rid,payload in buffer.iter_records()}
    # The other arm uses the same original context and succeeds without re-ask.
    original=copy.deepcopy(records[receipt['first_request_id']+'/request']['payload'])
    original['request_id']='lock';original['decoding_params'].pop('response_format')
    records['lock/request']={'payload':original}
    contacts=[{'request_id':receipt['reask_request_id'],'agent_contact_id':'c-solver'}, {'request_id':'lock','agent_contact_id':'c-lock'}]
    assert verify_contact_pair(records,contacts,expected_format=fmt)['same_input_except_protocol']


def test_real_flight_schema_uses_governed_train_loader(monkeypatch):
    from speech_aware_evidence_acquisition.core.flight.runtime import FlightRuntime
    from speech_aware_evidence_acquisition.rails import slurp_solver_protocol as sp
    c,v=compiled()
    from speech_aware_evidence_acquisition.rails.native_episode_config import load_native_episode_config
    from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
    v['carrier_lock_key']='slurp'
    cfg=load_native_episode_config(v,default_phase_program(v['policy']))
    runtime=object.__new__(FlightRuntime);runtime._lock=object();runtime._data_root=Path('/owned-data')
    seen=[]
    monkeypatch.setattr(sp,'registered_response_format',lambda lock,root:seen.append((lock,root)) or {'verified':True})
    assert runtime._protocol_response_formats(cfg)=={FORMAT_ID:{'verified':True}}
    assert seen==[(runtime._lock,runtime._data_root)]


@pytest.mark.parametrize('fault',['drop_pair','rename_action','drop_type','missing_authority'])
def test_protocol_audit_rejects_valid_shape_with_wrong_inventory(fault):
    from speech_aware_evidence_acquisition.rails.solver_protocol_audit import verify_pair
    expected=response_format({'audio':['volume_up','volume_down'],'play':['music']},['person','song_name'])
    actions={'audio':['volume_up','volume_down'],'play':['music']};types=['person','song_name']
    if fault=='drop_pair':actions['audio'].pop()
    elif fault=='rename_action':actions['audio'][0]='not_a_real_action'
    elif fault=='drop_type':types.pop()
    fmt=response_format(actions,types)
    lock={'history':[],'decoding_params':{'seed':7},'task_instruction':'same'}
    constrained={**lock,'decoding_params':{'seed':7,'response_format':fmt}}
    with pytest.raises(ValueError,match='governed TRAIN'):
        verify_pair(constrained,lock,expected_format=None if fault=='missing_authority' else expected)


def test_protocol_compiler_requires_carrier_admission():
    from dataclasses import replace
    hooks = carrier_module_hooks(repo_root=ROOT)
    with pytest.raises(cm.ConsumptionModuleConfigError, match="solver_protocol_formats"):
        cm.compile_module(block(), replace(hooks, solver_protocol_formats=None), repo_root=ROOT)
    with pytest.raises(cm.ConsumptionModuleConfigError, match="not admitted"):
        cm.compile_module(block(), replace(hooks, solver_protocol_formats=frozenset()), repo_root=ROOT)
    alternative = block()
    alternative["columns"][0]["response_format_id"] = "other-carrier-output-v1"
    compiled = cm.compile_module(alternative, replace(hooks, solver_protocol_formats=frozenset({"other-carrier-output-v1"})), repo_root=ROOT)
    protocol = compiled.values()["native_episode"]["roles"]["solver"]["protocol_variants"]
    assert protocol["arms"][0]["response_format_id"] == "other-carrier-output-v1"
    assert parse_protocol_variants(protocol, role_kind="solver", phase_id="solver").arms[0].response_format_id == "other-carrier-output-v1"
