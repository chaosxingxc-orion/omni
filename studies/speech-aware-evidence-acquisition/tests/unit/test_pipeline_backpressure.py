"""Exercise bounded direct scheduling without services or temporary Git repos."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading

import pytest

from speech_aware_evidence_acquisition.core.flight.runtime import FlightRuntime


def harness(count, prepare, complete, validate):
    runtime = object.__new__(FlightRuntime)
    runtime._samples = tuple(range(count))
    runtime.uses_speech_feature_packets = lambda: False
    runtime._decoding_params = lambda: {}
    runtime._sample_decoding_params = lambda sample, base: base
    runtime._adapter = SimpleNamespace(assemble_prompt=lambda sample, _: sample)
    runtime._row_for = validate
    active = set()
    flushed = []

    def build(sample, **kwargs):
        prepare(sample)
        request = (sample, object())
        active.add(id(request))
        return request

    def send(request, **kwargs):
        response = complete(request[0])
        active.remove(id(request))
        return response

    client = SimpleNamespace(
        prepare_request=build, complete_prepared=send,
        discard_prepared=lambda request: active.discard(id(request)),
        new_trace_buffer=lambda: object(), flush_trace_buffer=flushed.append,
    )
    return runtime, client, active, flushed


def run(runtime, client, *, depth=1):
    return runtime._dispatch_prepared_pipeline(client, workers=1, options={
        'prepare_workers': 1, 'validation_workers': 1, 'queue_depth': depth,
    })


def test_validation_failure_does_not_send_late_preparation():
    """A prepared item arriving after a known failure must be discarded."""
    preparing = threading.Event()
    release = threading.Event()
    sent = []

    def prepare(sample):
        if sample == 1:
            preparing.set()
            assert release.wait(5), 'test failed to release preparation'

    def validate(sample, messages, response):
        if sample == 0:
            assert preparing.wait(5), 'preparation did not overlap validation'
            raise ValueError('invalid response')
        return sample

    runtime, client, active, flushed = harness(8, prepare, lambda s: sent.append(s) or 'ok', validate)
    # Observe cancellation of a RUNNING preparation future: cancel returns
    # false, then the builder must finish and its prepared request be discarded.
    import concurrent.futures
    original = concurrent.futures.Future.cancel
    cancelled = threading.Event()

    def observed_cancel(future):
        result = original(future)
        if not result and future.running():
            cancelled.set()
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(concurrent.futures.Future, 'cancel', observed_cancel)
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(run, runtime, client)
            try:
                assert cancelled.wait(5), 'validation failure did not wake the dispatcher'
            finally:
                release.set()
            with pytest.raises(ValueError, match='invalid response'):
                result.result(timeout=5)
    assert sent == [0]
    assert not active
    assert len(flushed) == len(sent)


def test_slow_validation_bounds_responses_and_preserves_order():
    release = threading.Event()
    blocked = threading.Event()
    sent = []

    def validate(sample, messages, response):
        if sample == 0:
            blocked.set()
            assert release.wait(5)
        return {'sample_id': sample, 'response': response}

    runtime, client, active, flushed = harness(20, lambda s: None, lambda s: sent.append(s) or str(s), validate)
    # Backpressure is witnessed directly at wait(), avoiding timing sleeps.
    import concurrent.futures
    original_wait = concurrent.futures.wait
    pressure = threading.Event()

    def observed_wait(futures, **kwargs):
        if blocked.is_set() and len(futures) == 1 and len(sent) >= 2:
            pressure.set()
        return original_wait(futures, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(concurrent.futures, 'wait', observed_wait)
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(run, runtime, client)
            try:
                assert pressure.wait(5), 'no backpressure on a blocked validator'
                assert len(sent) == 2  # one validator + one completed model
            finally:
                release.set()
            rows, receipt = result.result(timeout=5)
    assert [row['sample_id'] for row in rows] == list(range(20))
    assert receipt['max_validation_pending'] == 1
    assert receipt['validation_backpressure_events'] >= 1
    assert not active
    assert len(flushed) == 20
