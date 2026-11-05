import threading
from concurrent.futures import ThreadPoolExecutor
from speech_aware_evidence_acquisition.core.model.adapter import FrozenCoreAdapter
from speech_aware_evidence_acquisition.core.tracing import TraceSink
from test_session import _adapter_world, _RecordingSessionTransport, _OpenGate


def test_response_completion_does_not_wait_on_other_admission(tmp_path):
    binding,plan,payload=_adapter_world(tmp_path)
    adapter=FrozenCoreAdapter(gate=_OpenGate(),plan=plan,arm='bare-core',transport=_RecordingSessionTransport(),sink=TraceSink(tmp_path/'run',plan.run_id),binding=binding)
    adapter._metering_lock.acquire()
    pool=ThreadPoolExecutor(max_workers=1)
    try:
        f=pool.submit(adapter._consume_response,payload,b'{"text":"ok","usage":{"prompt_tokens":1,"completion_tokens":1}}',0.01,'a'*64)
        assert f.result(timeout=3).text=='ok'
    finally:
        adapter._metering_lock.release()
        pool.shutdown(wait=True)
    assert adapter.cost_summary()['prompt_tokens_total']==1


def test_concurrent_live_checks_not_serialized_and_budget_exact(tmp_path):
    from speech_aware_evidence_acquisition.core.model import FrozenCoreError
    binding,plan,payload=_adapter_world(tmp_path)
    barrier=threading.Barrier(2,timeout=3)
    class Transport(_RecordingSessionTransport):
        def assert_session_alive(self):
            self.session_checks+=1
            if self.session_checks>1:barrier.wait()
    transport=Transport();adapter=FrozenCoreAdapter(gate=_OpenGate(),plan=plan,arm='bare-core',transport=transport,sink=TraceSink(tmp_path/'parallel',plan.run_id),binding=binding)
    def request(i):return adapter.request({**payload,'request_id':f'req-{i}'})
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(request,range(2)))
    assert len(results)==2 and adapter.cost_summary()['calls_used']==2
    assert adapter.cost_summary()['prompt_tokens_total']==2


def test_post_reservation_session_failure_retains_charge(tmp_path):
    import pytest
    binding,plan,payload=_adapter_world(tmp_path)
    class Transport(_RecordingSessionTransport):
        def __call__(self,raw):raise RuntimeError('session changed before send')
    t=Transport();adapter=FrozenCoreAdapter(gate=_OpenGate(),plan=plan,arm='bare-core',transport=t,sink=TraceSink(tmp_path/'drift',plan.run_id),binding=binding)
    with pytest.raises(RuntimeError,match='session changed'):adapter.request(payload)
    assert adapter.cost_summary()['calls_used']==1


def test_failed_live_check_records_no_cost(tmp_path):
    import pytest
    from speech_aware_evidence_acquisition.core.session import SessionError
    binding,plan,payload=_adapter_world(tmp_path);t=_RecordingSessionTransport()
    adapter=FrozenCoreAdapter(gate=_OpenGate(),plan=plan,arm='bare-core',transport=t,sink=TraceSink(tmp_path/'pre-fail',plan.run_id),binding=binding)
    t.should_fail=True
    with pytest.raises(SessionError):adapter.request(payload)
    assert adapter.cost_summary()['calls_used']==0 and t.calls==0


def test_concurrent_admission_does_not_overspend(tmp_path):
    from dataclasses import replace
    from speech_aware_evidence_acquisition.core.model import FrozenCoreError
    binding,plan,payload=_adapter_world(tmp_path);plan=replace(plan,planned_model_calls=1)
    t=_RecordingSessionTransport();adapter=FrozenCoreAdapter(gate=_OpenGate(),plan=plan,arm='bare-core',transport=t,sink=TraceSink(tmp_path/'budget',plan.run_id),binding=binding)
    barrier=threading.Barrier(2)
    def request(i):
        barrier.wait()
        try:adapter.request({**payload,'request_id':f'id-{i}'});return 'sent'
        except FrozenCoreError:return 'refused'
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(request,range(2)))
    assert sorted(results)==['refused','sent'] and adapter.cost_summary()['calls_used']==1
