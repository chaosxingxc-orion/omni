"""Real blocking callbacks must reach admitted concurrency above Python's default."""
import threading
import pytest
from speech_aware_evidence_acquisition.reproduction.ojw.runner import execute_speech_agent_workflows


@pytest.mark.parametrize('slots,rows', [(4,12),(40,40),(96,96)])
def test_blocking_contacts_fill_admitted_slots_without_duplicate_slots(slots, rows):
    barrier=threading.Barrier(slots)
    lock=threading.Lock()
    active=set()
    seen=[]
    peaks=[]
    async def episode(state, run_contact, session):
        def contact(slot):
            with lock:
                assert slot not in active
                active.add(slot)
                peaks.append(len(active))
                seen.append(state)
            try:
                barrier.wait(timeout=10)
            finally:
                with lock: active.remove(slot)
        await run_contact(contact)
    states=list(range(rows))
    assert execute_speech_agent_workflows(states,episode_invoker=episode,branch_count=1,
        slot_count=slots,queue_depth=rows,workflow_timeout_seconds=20)==states
    assert sorted(seen)==states
    assert max(peaks)==slots
    assert not active
