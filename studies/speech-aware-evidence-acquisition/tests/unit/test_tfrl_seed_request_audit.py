import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/flights/nx'))
import tfrl_seed_request_audit as m


@pytest.mark.parametrize('fault',[None,'run','attempt','unfinished','unfinalized'])
def test_file_audit_requires_verified_finalized_plan_identity(tmp_path,monkeypatch,fault):
    from speech_aware_evidence_acquisition import contracts
    from speech_aware_evidence_acquisition.core import bundle
    monkeypatch.setattr(contracts,'assert_execution_environment',lambda:None)
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    plan=tmp_path/'plan.json';plan.write_text(json.dumps({'run_id':'run','attempt_id':'attempt'}))
    doc={'status':'COMPLETED','run_id':'run','attempt':{'attempt_id':'attempt'},'raw_trace':{'sha256':'a'*64}}
    if fault=='run':doc['run_id']='other'
    elif fault=='attempt':doc['attempt']['attempt_id']='other'
    elif fault=='unfinished':doc['status']='FAILED'
    verified=SimpleNamespace(manifest=doc,attempt_state='OPEN' if fault=='unfinalized' else 'FINALIZED',outputs_rows=[],trace_records={})
    calls=[]
    monkeypatch.setattr(bundle,'verify_bundle',lambda p:(calls.append(p),verified)[1])
    monkeypatch.setattr(m,'verify',lambda *a:{'rows':313})
    out=tmp_path/'check.json'
    if fault:
        with pytest.raises(ValueError):m.audit(manifest,plan,out)
        assert not out.exists()
    else:
        got=m.audit(manifest,plan,out)
        assert got['rows']==313 and len(got['plan_sha256'])==64
        with pytest.raises(ValueError):m.audit(manifest,plan,out)
    assert calls==[manifest]


def test_shared_bundle_verifier_invokes_seed_request_check(monkeypatch):
    from test_bundle_supply_variants import fixture
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_supply_variant_trace_binding
    from speech_aware_evidence_acquisition.core.bundle import BundleError
    from speech_aware_evidence_acquisition.rails import trajectory_seeds
    row,trace=fixture()
    def refuse(*args):raise ValueError('actual seed differs')
    monkeypatch.setattr(trajectory_seeds,'verify_recorded_group',refuse)
    with pytest.raises(BundleError,match='seed-only request binding failed'):
        _verify_supply_variant_trace_binding(row,trace)
