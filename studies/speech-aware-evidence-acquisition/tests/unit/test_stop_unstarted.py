import asyncio
import json
import threading
import pytest
from speech_aware_evidence_acquisition.core.admission_control import AdmissionControl, request_stop
from speech_aware_evidence_acquisition.reproduction.ojw.runner import execute_speech_agent_workflows
from speech_aware_evidence_acquisition.core.row_checkpoint import RowCheckpointWriter


def test_queue_stop_preserves_started_and_cancels_unsent(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(dict(schema='saea-admission-control-v1', run_id='r', attempt_id='a', request_path=str(tmp_path/'stop.json'), receipt_path=str(tmp_path/'receipt.json'), stop_policy='cancel_unstarted')))
    control = AdmissionControl(path, run_id='r', attempt_id='a')
    seen = []
    async def episode(state, contact, session):
        if state == 0:
            await contact(lambda slot: seen.append((state, 'p')))
            request_stop(path, request_id='stop')
            await contact(lambda slot: seen.append((state, 's')))
        else:
            await contact(lambda slot: seen.append((state, 'p')))
    events = []
    writer = RowCheckpointWriter(tmp_path/'attempt', 'r', 'a', identity={})
    def checkpoint(state, timing):
        writer.append_row(sample_id=str(state), complete=True, output_rows=[{'sample_id':str(state)}], trace_lines=[])
    def drain(states):
        events.append(states)
        csummary = {key.replace('_indices', '_sample_ids'): [str(i) for i in values] for key, values in control.stop_summary.items()}
        control.record_stop_summary(csummary)
        control.publish_cancelled_traces({str(s): [] for s in (1,2,3)})
        control.finish(writer.path, [str(s) for s in states])
    result = execute_speech_agent_workflows(list(range(5)), episode_invoker=episode, branch_count=1, slot_count=1, queue_depth=4, workflow_timeout_seconds=20, admission_should_stop=control.should_stop, admission_drained=drain, sample_timing_sink=checkpoint, admission_stop_policy=control.stop_policy, admission_stop_observed=control.observe_stop, admission_stop_summary=control.record_stop_summary)
    assert seen == [(0, 'p'), (0, 's')]
    assert result == [0] and events == [[0]]
    receipt = json.loads(control.receipt_path.read_text())
    assert receipt['admitted_sample_ids'] == ['0']
    assert receipt['stop_summary'] == {'admitted_sample_ids': ['0','1','2','3'], 'cancelled_sample_ids': ['1','2','3'], 'started_sample_ids': ['0'], 'completed_sample_ids': ['0']}
    assert receipt['checkpoint']['complete_rows'] == 1 and not receipt['generation_lease_released']
    writer.close()
    assert json.loads((tmp_path/'receipt.json.observed').read_text())['stop_request_sha256'] == control.request_digest


def test_tool_does_not_mark_model_started():
    checks = iter([False, True])
    calls = []
    async def episode(state, contact, session):
        await contact(lambda slot: calls.append('tool'), contact_kind='tool')
        await contact(lambda slot: calls.append('model'))
    summary=[]
    execute_speech_agent_workflows([0], episode_invoker=episode, branch_count=1, slot_count=1, queue_depth=1, workflow_timeout_seconds=20, admission_should_stop=lambda: next(checks,True), admission_drained=lambda states:None, admission_stop_policy='cancel_unstarted', admission_stop_observed=lambda info:None, admission_stop_summary=summary.append)
    assert calls == ['tool'] and summary[0]['cancelled_indices'] == [0]


@pytest.mark.parametrize('mode', ['node_batch', 'node_pipeline'])
def test_unsupported_modes_are_explicit(mode):
    options = {'node_batch_order':['p'], 'node_batch_size':1} if mode == 'node_batch' else {'node_pipeline_order':['p'], 'node_pipeline_group_size':1, 'node_pipeline_prepare_workers':1, 'node_pipeline_max_jobs':1}
    async def episode(*args, **kwargs): pass
    with pytest.raises(ValueError, match='cancel_unstarted'):
        execute_speech_agent_workflows([0], episode_invoker=episode, branch_count=1, slot_count=1, queue_depth=1, workflow_timeout_seconds=20, admission_should_stop=lambda:False, admission_drained=lambda states:None, admission_stop_policy='cancel_unstarted', admission_stop_observed=lambda info:None, admission_stop_summary=lambda info:None, **options)


def test_opt_in_without_stop_completes_all():
    calls = []
    async def episode(state, contact, session):
        await contact(lambda slot: calls.append(state))
    result = execute_speech_agent_workflows(list(range(5)), episode_invoker=episode, branch_count=1,
        slot_count=1, queue_depth=4, workflow_timeout_seconds=20, admission_should_stop=lambda:False,
        admission_drained=lambda _:pytest.fail('unexpected drain'), admission_stop_policy='cancel_unstarted',
        admission_stop_observed=lambda _:pytest.fail('unexpected stop'), admission_stop_summary=lambda _:pytest.fail('unexpected summary'))
    assert result == list(range(5)) and sorted(calls) == list(range(5))


def test_unknown_model_failure_never_creates_stop_receipt(tmp_path):
    path = tmp_path/'config.json'
    path.write_text(json.dumps(dict(schema='saea-admission-control-v1', run_id='r', attempt_id='a', request_path=str(tmp_path/'stop.json'), receipt_path=str(tmp_path/'receipt.json'), stop_policy='cancel_unstarted')))
    control = AdmissionControl(path, run_id='r', attempt_id='a')
    async def episode(state, contact, session):
        def failure(slot):
            request_stop(path, request_id='stop')
            raise TimeoutError('unknown response')
        await contact(failure)
    with pytest.raises(TimeoutError):
        execute_speech_agent_workflows([0,1], episode_invoker=episode, branch_count=1, slot_count=1,
            queue_depth=2, workflow_timeout_seconds=20, admission_should_stop=control.should_stop,
            admission_drained=lambda _:pytest.fail('must not drain unknown'), admission_stop_policy=control.stop_policy,
            admission_stop_observed=control.observe_stop, admission_stop_summary=control.record_stop_summary)
    assert not control.receipt_path.exists()


def test_observed_attempt_refused_and_new_attempt_isolated(tmp_path, monkeypatch):
    from speech_aware_evidence_acquisition.core.admission_control import from_environment, ENVIRONMENT_KEY
    def make(directory, attempt):
        directory.mkdir()
        path = directory/'config.json'
        path.write_text(json.dumps(dict(schema='saea-admission-control-v1', run_id='r', attempt_id=attempt,
            request_path=str(directory/'stop.json'), receipt_path=str(directory/'receipt.json'), stop_policy='cancel_unstarted')))
        return path
    old = make(tmp_path/'old', 'a')
    first = AdmissionControl(old, run_id='r', attempt_id='a')
    request_stop(old, request_id='stop')
    first.observe_stop(dict(observed_unix_ns=1, observed_monotonic_ns=2, admitted_count=0, started_count=0, completed_count=0))
    original = (tmp_path/'old'/'receipt.json.observed').read_bytes()
    with pytest.raises(FileExistsError):
        first.observe_stop(dict(observed_unix_ns=9, observed_monotonic_ns=9, admitted_count=0, started_count=0, completed_count=0))
    assert (tmp_path/'old'/'receipt.json.observed').read_bytes() == original
    monkeypatch.setenv(ENVIRONMENT_KEY, str(old))
    with pytest.raises(ValueError, match='new attempt'):
        from_environment(run_id='r', attempt_id='a')
    new = make(tmp_path/'new', 'b')
    monkeypatch.setenv(ENVIRONMENT_KEY, str(new))
    assert not from_environment(run_id='r', attempt_id='b').should_stop()
    with pytest.raises(ValueError, match='bind'):
        from_environment(run_id='r', attempt_id='a')


def test_natural_completion_and_tool_cost_custody(tmp_path):
    checks = iter([False, False, True])
    events=[]; summary=[]; completed=[]
    async def episode(state, contact, session):
        if state == 0:
            return
        await contact(lambda slot: events.append(('tool', state)), contact_kind='tool')
        await contact(lambda slot: events.append(('model', state)))
    execute_speech_agent_workflows([0,1], episode_invoker=episode, branch_count=1, slot_count=1,
        queue_depth=2, workflow_timeout_seconds=20, admission_should_stop=lambda:next(checks, True),
        admission_drained=completed.extend, admission_stop_policy='cancel_unstarted',
        admission_stop_observed=lambda _:None, admission_stop_summary=summary.append)
    assert events == [('tool',1)]
    assert completed == [0]
    assert summary == [dict(admitted_indices=[0,1], cancelled_indices=[1], started_indices=[], completed_indices=[0])]


def test_cancelled_tool_trace_is_fsynced_and_bound_to_receipt(tmp_path):
    import hashlib
    from speech_aware_evidence_acquisition.core.row_checkpoint import serialize_trace_records
    path=tmp_path/'config.json'
    path.write_text(json.dumps(dict(schema='saea-admission-control-v1',run_id='r',attempt_id='a',
        request_path=str(tmp_path/'stop.json'),receipt_path=str(tmp_path/'receipt.json'),stop_policy='cancel_unstarted')))
    control=AdmissionControl(path,run_id='r',attempt_id='a')
    request_stop(path,request_id='tool-cost')
    control.record_stop_summary(dict(admitted_sample_ids=['s1'],cancelled_sample_ids=['s1'],started_sample_ids=[],completed_sample_ids=[]))
    with pytest.raises(ValueError, match='custody'):
        control.finish(tmp_path/'absent', [])
    assert not control.receipt_path.exists()
    lines=serialize_trace_records([('USE','s1/tool',{'use_cost':{'cpu_seconds':1.25}})])
    control.publish_cancelled_traces({'s1':lines})
    receipt=control.finish(tmp_path/'absent',[])
    raw=control.cancelled_trace_path.read_bytes()
    proof=receipt['cancelled_trace_custody']
    assert hashlib.sha256(raw).hexdigest() == proof['sha256'] and proof['trace_record_count'] == 1
    assert json.loads(raw)['samples']['s1'] == lines
    assert json.loads(lines[0])['payload']['use_cost']['cpu_seconds'] == 1.25
    with pytest.raises(FileExistsError):
        control.publish_cancelled_traces({'s1':lines})


def test_foreign_request_and_conflicting_observed_do_not_get_success(tmp_path, monkeypatch):
    from speech_aware_evidence_acquisition.core.admission_control import from_environment, ENVIRONMENT_KEY
    path=tmp_path/'config.json'
    path.write_text(json.dumps(dict(schema='saea-admission-control-v1',run_id='r',attempt_id='a',
        request_path=str(tmp_path/'stop.json'),receipt_path=str(tmp_path/'receipt.json'),stop_policy='cancel_unstarted')))
    control=AdmissionControl(path,run_id='r',attempt_id='a')
    request_stop(path,request_id='foreign')
    request=json.loads(control.request_path.read_text());request['config_sha256']='foreign';control.request_path.write_text(json.dumps(request))
    fresh=AdmissionControl(path,run_id='r',attempt_id='a')
    assert not fresh.should_stop()
    with pytest.raises(ValueError,match='requires stop'):
        fresh.observe_stop({})
    assert not fresh.observed_path.exists() and not fresh.receipt_path.exists()
    fresh.observed_path.write_text('{"foreign":true}')
    monkeypatch.setenv(ENVIRONMENT_KEY,str(path))
    with pytest.raises(ValueError,match='new attempt'):
        from_environment(run_id='r',attempt_id='a')
    assert json.loads(fresh.observed_path.read_text()) == {'foreign':True}


def test_default_tool_failure_prevents_successful_peer_drain():
    calls=[]
    async def episode(state, contact, session):
        def tool(slot):
            calls.append(state)
            raise TimeoutError('tool failure')
        await contact(tool, contact_kind='tool')
    with pytest.raises(TimeoutError):
        execute_speech_agent_workflows([0,1], episode_invoker=episode, branch_count=1, slot_count=1,
            queue_depth=2, workflow_timeout_seconds=20, admission_should_stop=lambda:False,
            admission_drained=lambda _:pytest.fail('failed tool must not drain'))
    assert calls == [0]



def test_foreign_derived_evidence_paths_are_rejected(tmp_path):
    path=tmp_path/'config.json'
    path.write_text(json.dumps(dict(schema='saea-admission-control-v1',run_id='r',attempt_id='a',
        request_path=str(tmp_path/'receipt.json.observed'),receipt_path=str(tmp_path/'receipt.json'),stop_policy='cancel_unstarted')))
    with pytest.raises(ValueError,match='distinct'):
        AdmissionControl(path,run_id='r',attempt_id='a')


@pytest.mark.parametrize('error_kind', ['runtime', 'framework'])
def test_cleanup_exception_after_stop_is_not_swallowed(error_kind):
    from openjiuwen.core.common.exception.codes import StatusCode
    from openjiuwen.core.common.exception.errors import BaseError
    checks = iter([False, True])
    drained = []
    async def episode(state, contact, session):
        try:
            await contact(lambda slot: pytest.fail('model must not dispatch'))
        finally:
            if error_kind == 'runtime':
                raise RuntimeError('cleanup failed')
            raise BaseError(StatusCode.WORKFLOW_EXECUTION_TIMEOUT)
    expected = RuntimeError if error_kind == 'runtime' else Exception
    with pytest.raises(expected) as caught:
        execute_speech_agent_workflows([0], episode_invoker=episode, branch_count=1,
            slot_count=1, queue_depth=1, workflow_timeout_seconds=20,
            admission_should_stop=lambda:next(checks, True), admission_drained=drained.append,
            admission_stop_policy='cancel_unstarted', admission_stop_observed=lambda _:None,
            admission_stop_summary=lambda _:pytest.fail('must not report successful stop'))
    if error_kind == 'runtime':
        assert str(caught.value) == 'cleanup failed'
    else:
        from speech_aware_evidence_acquisition.reproduction.ojw.runner import OpenJiuwenExecutorError
        assert isinstance(caught.value, OpenJiuwenExecutorError)
    assert drained == []


@pytest.mark.parametrize('error_kind', ['explicit_runtime', 'explicit_framework', 'noted_sentinel'])
def test_explicit_cleanup_errors_remain_failures(error_kind):
    from openjiuwen.core.common.exception.codes import StatusCode
    from openjiuwen.core.common.exception.errors import BaseError
    checks=iter([False, True]); drained=[]
    async def episode(state, contact, session):
        try:
            await contact(lambda slot:pytest.fail('unexpected model'))
        except Exception as stop:
            if error_kind == 'explicit_runtime':
                raise RuntimeError('cleanup failed') from stop
            if error_kind == 'explicit_framework':
                raise BaseError(StatusCode.WORKFLOW_EXECUTION_TIMEOUT) from stop
            stop.add_note('cleanup failed')
            raise
    with pytest.raises(Exception):
        execute_speech_agent_workflows([0], episode_invoker=episode, branch_count=1, slot_count=1,
            queue_depth=1, workflow_timeout_seconds=20, admission_should_stop=lambda:next(checks, True),
            admission_drained=drained.append, admission_stop_policy='cancel_unstarted',
            admission_stop_observed=lambda _:None, admission_stop_summary=lambda _:pytest.fail('successful stop'))
    assert not drained
