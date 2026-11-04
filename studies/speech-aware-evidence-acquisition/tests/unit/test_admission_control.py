"""Real concurrent admission, durable checkpoints, and restart-bound stop custody."""
import json
import threading
from pathlib import Path
import pytest
from test_llama_server_transport import _fake_inference_session, _spawn_live_matching_process
from speech_aware_evidence_acquisition.core.admission_control import AdmissionControl, request_stop
from speech_aware_evidence_acquisition.core.row_checkpoint import RowCheckpointWriter
from speech_aware_evidence_acquisition.reproduction.ojw.runner import execute_speech_agent_workflows


def control(tmp_path, run='RUN', attempt='a1'):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path=tmp_path/'config.json'
    path.write_text(json.dumps({'schema':'saea-admission-control-v1','run_id':run,'attempt_id':attempt,
        'request_path':str(tmp_path/'stop.json'),'receipt_path':str(tmp_path/'drained.json')}))
    return AdmissionControl(path,run_id=run,attempt_id=attempt)


def execute(states, episode, stop, drain, **kwargs):
    return execute_speech_agent_workflows(states,episode_invoker=episode,branch_count=1,
        slot_count=2,queue_depth=2,workflow_timeout_seconds=20,
        admission_should_stop=stop,admission_drained=drain,**kwargs)


def test_real_concurrent_workers_finish_checkpoint_without_new_admission(tmp_path):
    c=control(tmp_path/'control');writer=RowCheckpointWriter(tmp_path/'attempt','RUN','a1',identity={})
    barrier=threading.Barrier(2);active=set();completed=[];calls=[];lock=threading.Lock()
    async def episode(state,run_contact,session):
        def contact(slot):
            with lock: active.add(state);calls.append(state)
            barrier.wait(timeout=10)
            if state==0:request_stop(c.path,request_id='stop-1')
            barrier.wait(timeout=10)
            with lock: active.remove(state)
        await run_contact(contact)
    def checkpoint(state,timing):
        assert timing['status']=='completed'
        writer.append_row(sample_id=str(state),complete=True,output_rows=[{'sample_id':str(state)}],trace_lines=[])
        completed.append(state)
    def drain(states):
        assert not active and sorted(completed)==[0,1]
        c.finish(writer.path,[str(s) for s in states])
    assert execute(list(range(8)),episode,c.should_stop,drain,sample_timing_sink=checkpoint)==[0,1]
    assert sorted(calls)==[0,1]
    receipt=json.loads(c.receipt_path.read_text());assert receipt['inflight_samples']==0
    assert receipt['checkpoint']['complete_rows']==2 and not receipt['generation_lease_released']
    before=c.receipt_path.read_bytes()
    restarted=AdmissionControl(c.path,run_id='RUN',attempt_id='a1')
    assert execute(list(range(8)),episode,restarted.should_stop,
        lambda states: restarted.finish(writer.path,[]))==[]
    assert c.receipt_path.read_bytes()==before and len(calls)==2
    writer.close()


def test_failed_contact_never_publishes_drained(tmp_path):
    c=control(tmp_path);request_stop(c.path,request_id='x')
    # Stop arrives during admission, after both workers entered the invoker.
    checks=iter([False,False,True]);barrier=threading.Barrier(2);seen=[]
    async def episode(state,run_contact,session):
        def contact(slot):
            seen.append(state);barrier.wait(timeout=10)
            if state==0:raise TimeoutError('unknown provider response')
        await run_contact(contact)
    with pytest.raises(TimeoutError):
        execute([0,1,2],episode,lambda:next(checks,True),lambda _:pytest.fail('must not declare drained'))
    assert sorted(seen)==[0,1] and not c.receipt_path.exists()


@pytest.mark.parametrize('fault',['request','symlink'])
def test_foreign_or_invalid_stop_never_gets_a_drain_receipt(tmp_path,fault):
    c=control(tmp_path);request_stop(c.path,request_id='x');p=c.request_path
    if fault=='symlink':
        actual=tmp_path/'foreign.json';p.rename(actual);p.symlink_to(actual)
    else:
        d=json.loads(p.read_text());key={'run':'run_id','attempt':'attempt_id','config':'config_sha256','request':'request_id'}[fault]
        d[key]='wrong' if fault!='request' else '';p.write_text(json.dumps(d))
    async def episode(*args):pytest.fail('must not admit')
    with pytest.raises(ValueError):execute([0],episode,c.should_stop,lambda _:pytest.fail('must not drain'))
    assert not c.receipt_path.exists()


def test_no_checkpoint_or_unkeepable_rows_cannot_claim_drain(tmp_path):
    c=control(tmp_path/'control');request_stop(c.path,request_id='x')
    with pytest.raises(ValueError,match='checkpoints'):c.finish(tmp_path/'missing',['x'])
    writer=RowCheckpointWriter(tmp_path/'attempt','RUN','a1',identity={})
    writer.append_row(sample_id='x',complete=False,output_rows=[],trace_lines=[],reason='failed')
    with pytest.raises(ValueError,match='checkpoints'):c.finish(writer.path,['x'])
    assert not c.receipt_path.exists();writer.close()


def test_stop_request_idempotent_latched_and_attempt_bound(tmp_path):
    c=control(tmp_path);a=request_stop(c.path,request_id='x')
    assert request_stop(c.path,request_id='x')==a
    with pytest.raises(ValueError):request_stop(c.path,request_id='y')
    with pytest.raises(ValueError):AdmissionControl(c.path,run_id='RUN',attempt_id='a2')
    assert c.should_stop();c.request_path.unlink();assert c.should_stop()


def test_preexisting_stop_prevents_entire_node_panel(tmp_path):
    c=control(tmp_path);request_stop(c.path,request_id='x');seen=[]
    async def episode(*args,**kwargs):pytest.fail('panel must not start')
    assert execute([0,1],episode,c.should_stop,lambda states:seen.extend(states),
        node_pipeline_order=['p','s'],node_pipeline_group_size=2,node_pipeline_prepare_workers=1,
        node_pipeline_max_jobs=2)==[]
    assert seen==[]


def test_stop_between_node_cohorts_drains_all_roles(tmp_path):
    from test_native_node_batch_scheduler import run
    c=control(tmp_path);seen=[];drained=[]
    async def episode(state,contact,_session):
        await contact(lambda slot:seen.append((state,'p')),prefix_family='perception')
        if state==0:request_stop(c.path,request_id='cohort')
        await contact(lambda slot:seen.append((state,'s')),prefix_family='solver')
    assert run(list(range(6)),episode,size=2,admission_should_stop=c.should_stop,
               admission_drained=lambda states:drained.extend(states))==[0,1]
    assert drained==[0,1] and sorted(seen)==[(0,'p'),(0,'s'),(1,'p'),(1,'s')]


def test_actual_native_runner_flushes_checkpoint_and_refuses_partial_success(tmp_path,monkeypatch):
    from test_native_episode_runner import _ActualEpisodeHarness
    from speech_aware_evidence_acquisition.core.admission_control import AdmissionDrained, ENVIRONMENT_KEY
    from speech_aware_evidence_acquisition.core.row_checkpoint import serialize_trace_records
    h=_ActualEpisodeHarness(tmp_path/'world',sample_count=3,max_revisions=0,max_contacts=4,planner_actions=('ACCEPT',))
    h.slot_count=1
    c=control(tmp_path/'control',h.plan.run_id,h.plan.attempt_id)
    writer=RowCheckpointWriter(tmp_path/'actual-checkpoint',h.plan.run_id,h.plan.attempt_id,identity={})
    done=[]
    def checkpoint(*,outcome,trace_records):
        sid=outcome.prepared.sample.sample_id
        writer.append_row(sample_id=sid,complete=True,output_rows=[{'sample_id':sid}],
                          trace_lines=serialize_trace_records(trace_records))
        done.append(sid);request_stop(c.path,request_id='first-durable-row')
    checkpoint.checkpoint_path=writer.path
    h.trace.row_checkpoint=checkpoint
    import test_native_episode_runner as fixture
    from dataclasses import replace
    original_binding = fixture.NativeEpisodeRunBinding
    monkeypatch.setattr(fixture, 'NativeEpisodeRunBinding',
        lambda **kwargs: replace(original_binding(**kwargs), admission_control=c))
    try:
        with pytest.raises(AdmissionDrained):h.run()
    finally:
        summary=h.close();writer.close()
    assert done==['s1'] and summary['request_count']==4
    receipt=json.loads(c.receipt_path.read_text())
    assert receipt['admitted_sample_ids']==['s1'] and receipt['checkpoint']['complete_rows']==1
    assert not receipt['generation_complete']


def test_flight_injects_operational_control_and_keeps_complete_rows(tmp_path,monkeypatch):
    from test_native_episode_runner import _ActualEpisodeHarness
    from test_flight_row_checkpoint_resume import _ckpt_skeleton, _sequential
    from speech_aware_evidence_acquisition.core.admission_control import AdmissionDrained, ENVIRONMENT_KEY
    from speech_aware_evidence_acquisition.core.row_checkpoint import read_checkpoint
    h=_ActualEpisodeHarness(tmp_path/'world',sample_count=2,max_revisions=0,max_contacts=4,planner_actions=('ACCEPT',))
    _sequential(h)
    runtime=_ckpt_skeleton(h,monkeypatch,tmp_path/'explicit-run-dir',['s1','s2'])
    c=control(tmp_path/'control',h.plan.run_id,h.plan.attempt_id)
    monkeypatch.setenv(ENVIRONMENT_KEY,str(c.path))
    original=runtime._checkpoint_row
    def checkpoint(**kwargs):
        original(**kwargs)
        request_stop(c.path,request_id='after-first-checkpoint')
    monkeypatch.setattr(runtime,'_checkpoint_row',checkpoint)
    try:
        with pytest.raises(AdmissionDrained):
            runtime._dispatch_phase_program(h.client,h.plan,h.program.program_id)
    finally:
        summary=h.close()
    receipt=json.loads(c.receipt_path.read_text())
    assert receipt['admitted_sample_ids']==['s1'] and summary['request_count']==4
    assert Path(receipt['checkpoint']['path']).parent==tmp_path/'explicit-run-dir'
    assert read_checkpoint(Path(receipt['checkpoint']['path'])).complete_rows==1


def test_concurrent_identical_stop_requests_publish_once(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    c=control(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        replies=list(pool.map(lambda _:request_stop(c.path,request_id='same'),range(8)))
    assert all(r==replies[0] for r in replies) and c.should_stop()


def test_restart_rejects_forged_drain_receipt(tmp_path):
    c=control(tmp_path);request_stop(c.path,request_id='x');c.finish(tmp_path/'missing',[])
    receipt=json.loads(c.receipt_path.read_text());receipt['generation_complete']=True
    c.receipt_path.write_text(json.dumps(receipt))
    restarted=AdmissionControl(c.path,run_id='RUN',attempt_id='a1')
    with pytest.raises(ValueError,match='receipt differs'):restarted.finish(tmp_path/'missing',[])


@pytest.mark.parametrize('field',['run_id','attempt_id','config_sha256'])
def test_foreign_request_cannot_stop_another_run(tmp_path,field):
    c=control(tmp_path);request_stop(c.path,request_id='foreign')
    request=json.loads(c.request_path.read_text());request[field]='different'
    c.request_path.write_text(json.dumps(request));seen=[]
    async def episode(state,run_contact,session):
        await run_contact(lambda slot:seen.append(state))
    assert execute([0,1,2],episode,c.should_stop,lambda _:pytest.fail('foreign stop'))==[0,1,2]
    assert sorted(seen)==[0,1,2] and not c.receipt_path.exists()
