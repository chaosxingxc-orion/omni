"""A repaired implementation cannot silently reuse stale CPU selections."""
from copy import deepcopy
import pytest
from speech_aware_evidence_acquisition.core.preparation import prepare_artifact, PreparationError
from speech_aware_evidence_acquisition.core.prepared_supply_recovery import verified_runtime_repair


def fixture(tmp_path):
    old={'config_sha256':'c'*64,'split_identity_hash':'d'*64,
         'sources':{'implementation':{'runtime':'a'*64,'rails':'e'*64},'packets':{'row':'pin'}}}
    output={'facets':{'p':{'row':['f1']}},'values':{}}
    path=tmp_path/'supply.json'
    receipt=prepare_artifact(path,kind='phase-supply',inputs=old,producer=lambda:output)
    current=deepcopy(old);current['sources']['implementation']['runtime']='b'*64
    return path,receipt['sha256'],current,output


def test_runtime_only_repair_recomputes_and_retains_exact_output(tmp_path):
    path,pin,current,output=fixture(tmp_path);calls=[]
    got,proof=verified_runtime_repair(path,expected_sha256=pin,inputs=current,
        produce=lambda:(calls.append(1) or deepcopy(output)))
    assert got==output and calls==[1]
    assert proof['current_runtime_sha256']=='b'*64 and proof['previous_runtime_sha256']=='a'*64
    assert proof['recomputed_with_current_implementation'] and proof['new_model_calls']==0


@pytest.mark.parametrize('fault',['config','split','rails','packet','same-runtime','bad-pin','output'])
def test_only_equal_output_and_runtime_only_identity_accepted(tmp_path,fault):
    path,pin,current,output=fixture(tmp_path)
    if fault=='config':current['config_sha256']='f'*64
    elif fault=='split':current['split_identity_hash']='f'*64
    elif fault=='rails':current['sources']['implementation']['rails']='f'*64
    elif fault=='packet':current['sources']['packets']['row']='other'
    elif fault=='same-runtime':current['sources']['implementation']['runtime']='a'*64
    elif fault=='bad-pin':pin='f'*64
    elif fault=='output':output['facets']['p']['row']=['f2']
    with pytest.raises(PreparationError):
        verified_runtime_repair(path,expected_sha256=pin,inputs=current,produce=lambda:output)


@pytest.mark.parametrize('recover',[False,True])
def test_actual_flight_requires_native_recovery_and_writes_rebuild_proof(tmp_path,monkeypatch,recover):
    from types import SimpleNamespace
    import json
    from test_prepared_phase_supply import setup_runtime
    from speech_aware_evidence_acquisition.rails import phase_supply
    from speech_aware_evidence_acquisition.core.flight import runtime as runtime_module
    runtime,binding,program=setup_runtime(tmp_path)
    monkeypatch.setattr(phase_supply,'build_bootstrap_selection',lambda *a,**k:({}, {'s1':('v1',)}))
    receipt=runtime.prepare_phase_supply(tmp_path/'seed.json')
    runtime._composed.values['prepared_phase_supply']={'relpath':'seed.json','sha256':receipt['sha256']}
    original=runtime._phase_seed_inputs
    def changed(*args):
        value=original(*args);value['sources']['implementation']['runtime']='f'*64;return value
    runtime._phase_seed_inputs=changed
    runtime._run_dir=lambda:tmp_path/'run'
    monkeypatch.setattr(runtime_module,'default_phase_program',lambda _:program)
    if recover:
        runtime._native_contact_replay=SimpleNamespace(source_plan=SimpleNamespace(attempt_id='prior'))
        runtime._native_contact_recovery={'path':'pinned','sha256':'a'*64}
        result=runtime._phase_supply_seeds(program,binding,binding['supply_specs'],SimpleNamespace(split_identity_hash='d'*64))
        assert result==({}, {'slots':{'s1':('v1',)}})
        proof=json.loads((tmp_path/'run/prepared-supply-runtime-recovery.json').read_bytes())
        assert proof['recomputed_with_current_implementation'] and proof['source_attempt_id']=='prior'
        runtime._prepared_artifact_reports=None
        assert runtime.verify_prepared_artifacts()[0]['identity_ok']
    else:
        with pytest.raises(PreparationError):
            runtime._phase_supply_seeds(program,binding,binding['supply_specs'],SimpleNamespace(split_identity_hash='d'*64))
        assert not (tmp_path/'run').exists()
