"""Environment to FlightRuntime to rails to real OJW scheduling and checkpoint."""
import functools
import json
import threading
from unittest.mock import patch
import pytest
from speech_aware_evidence_acquisition.core.admission_control import AdmissionDrained, ENVIRONMENT_KEY, request_stop
from speech_aware_evidence_acquisition.core.row_checkpoint import read_checkpoint, checkpoint_path
from speech_aware_evidence_acquisition.rails.phase_runner import run_continuous_program
from test_flight_row_checkpoint_resume import _ckpt_skeleton, _sequential
from test_flight_resume_recovery import _harness
from test_rails_frozen_core_client import _World
from test_llama_server_transport import _spawn_live_matching_process


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_environment_to_real_scheduler_and_checkpoint(tmp_path, monkeypatch, cleanup_failure):
    monkeypatch.setattr(_World, 'SAMPLE_SPECS', tuple((f's{i}', 1.5 + i * .5) for i in range(1, 5)))
    ids = ('s1','s2','s3','s4')
    harness = _harness(tmp_path, 'world', ids)
    _sequential(harness)
    runtime = _ckpt_skeleton(harness, monkeypatch, tmp_path/'attempt', ids)
    runtime._continuous_phase_runner = functools.partial(run_continuous_program, slot_count=1, queue_depth=4)
    directory=tmp_path/'control'; directory.mkdir()
    config=directory/'config.json'
    config.write_text(json.dumps(dict(schema='saea-admission-control-v1', run_id=harness.plan.run_id,
        attempt_id=harness.plan.attempt_id, request_path=str(directory/'stop.json'),
        receipt_path=str(directory/'receipt.json'), stop_policy='cancel_unstarted')))
    monkeypatch.setenv(ENVIRONMENT_KEY, str(config))
    post=harness.transport._post
    first=threading.Event()
    def stopping_post(url, body):
        response=post(url, body)
        if not first.is_set():
            first.set()
            request_stop(config, request_id='after-p')
        return response
    harness.transport._post=stopping_post
    import speech_aware_evidence_acquisition.rails.runner as rails_runner
    original_invoke = rails_runner._invoke_native_role_episode
    async def invoke_with_cleanup(*args, **kwargs):
        try:
            return await original_invoke(*args, **kwargs)
        except Exception:
            if cleanup_failure:
                raise RuntimeError('real episode cleanup failed')
            raise
    try:
        with patch.object(rails_runner, '_invoke_native_role_episode', invoke_with_cleanup):
            with pytest.raises(RuntimeError if cleanup_failure else AdmissionDrained) as caught:
                runtime._dispatch_phase_program(harness.client, harness.plan, harness.program.program_id)
        if cleanup_failure:
            assert str(caught.value) == 'real episode cleanup failed'
            assert not (directory/'receipt.json').exists()
            return
    finally:
        harness.close()
    receipt=json.loads((directory/'receipt.json').read_text())
    observed=json.loads((directory/'receipt.json.observed').read_text())
    doc=read_checkpoint(checkpoint_path(tmp_path/'attempt', harness.plan.run_id))
    assert observed['admitted_count'] == 4
    assert receipt['stop_summary'] == dict(admitted_sample_ids=list(ids), cancelled_sample_ids=list(ids[1:]), started_sample_ids=['s1'], completed_sample_ids=['s1'])
    assert receipt['admitted_sample_ids'] == ['s1'] and doc.complete_rows == 1
    import hashlib
    trace_file=directory/'receipt.json.cancelled-traces.json'
    trace_doc=json.loads(trace_file.read_text())
    assert set(trace_doc['samples']) == set(ids[1:])
    assert hashlib.sha256(trace_file.read_bytes()).hexdigest() == receipt['cancelled_trace_custody']['sha256']
    assert {key[0] for key in harness.dispatched} == {'s1'}
    assert len(harness.dispatched) == 4
