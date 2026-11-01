"""Opt-in operational stage timings outside scientific payloads and traces.

Nested intervals overlap; do not sum inclusive durations. The output path must be
fresh and owned by the launcher. No prompt, audio, answer, or exception text is
recorded. A timing write failure propagates instead of silently losing evidence.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps, lru_cache
import json
import math
import hashlib
import re
import uuid
import os
from pathlib import Path
import threading
import time

_context = ContextVar('saea_stage_timing_context', default={})
_lock = threading.Lock()
_fd = None
_path = None
_configuration = None
_active_span = ContextVar('saea_debug_span', default=None)


PROFILES = {
    'normal': {'mode':'normal','client_detail':False,'server_metrics_interval_seconds':600},
    'debug': {'mode':'debug','client_detail':True,'server_metrics_interval_seconds':600},
}
_METADATA = frozenset({'request_bytes','response_bytes','audio_bytes','encoded_bytes',
    'http_status','endpoint_sha256','server_request_id_sha256','timeout_seconds'})


def configuration():
    """Operational settings only; no scientific config or request mutation."""
    return dict(_load_configuration(os.environ.get('SAEA_OBSERVABILITY_MODE','normal'),
        os.environ.get('SAEA_OBSERVABILITY_CONFIG'),
        os.environ.get('SAEA_OBSERVABILITY_RUN_ID'),os.environ.get('SAEA_OBSERVABILITY_ATTEMPT_ID')))


@lru_cache(maxsize=8)
def _load_configuration(mode, config_path, run_id, attempt_id):
    # A launched process pins its profile once. Editing a profile on disk does
    # not alter an active run; its effective bytes are in the durable snapshot.
    if mode not in PROFILES:raise ValueError('observability mode must be normal or debug')
    result=dict(PROFILES[mode])
    if config_path:
        supplied=json.loads(Path(config_path).read_bytes())
        if not isinstance(supplied,dict) or set(supplied)!=set(result):
            raise ValueError('exact observability profile fields required')
        if supplied['mode']!=mode or supplied['client_detail'] is not (mode=='debug'):
            raise ValueError('observability mode/profile mismatch')
        interval=supplied['server_metrics_interval_seconds']
        if type(interval) is not int or interval<10:
            raise ValueError('metrics interval must be an integer >=10 seconds')
        if mode=='normal' and interval!=600:
            raise ValueError('normal observation interval is 600 seconds')
        result=supplied
    for key,value in (('run_id',run_id),('attempt_id',attempt_id)):
        if value:
            if not re.fullmatch(r'[A-Za-z0-9_.:-]{1,200}',value):raise ValueError('unsafe observability identity')
            result[key]=value
    return result


def debug_enabled():
    return bool(os.environ.get('SAEA_STAGE_TIMING_PATH')) and configuration()['client_detail']


def _metadata(values):
    if set(values)-_METADATA:raise ValueError('unapproved debug metadata field')
    if any(not isinstance(v,(int,float,str,type(None))) or isinstance(v,bool) for v in values.values()):
        raise ValueError('debug metadata must contain bounded scalar measurements')
    for k,v in values.items():
        if k.endswith('_sha256') and v is not None and (not isinstance(v,str) or not re.fullmatch('[a-f0-9]{64}',v)):
            raise ValueError('debug identifier must be hashed')
        if not k.endswith('_sha256') and (v is not None and (not isinstance(v,(int,float)) or not math.isfinite(v) or v<0)):
            raise ValueError('debug measurement must be nonnegative')
    return values


@contextmanager
def span(stage, metadata=None):
    """Debug-only fine interval. Caller may add only whitelisted scalar metrics.

    Client header latency is NOT server queue, prefill, decode or TTFT. Nested
    intervals must not be summed. No raw payload or endpoint is accepted.
    """
    values={} if metadata is None else dict(metadata)
    if not debug_enabled():
        yield values
        return
    _metadata(values)
    parent=_active_span.get();span_id=uuid.uuid4().hex;token=_active_span.set(span_id)
    started=time.perf_counter();wall=time.time_ns();status='ok'
    try:
        yield values
    except BaseException:
        status='error';raise
    finally:
        elapsed=time.perf_counter()-started;_active_span.reset(token)
        _metadata(values)
        _record(os.environ['SAEA_STAGE_TIMING_PATH'],{'schema':'saea-stage-timing-v1','stage':stage,
            'started_monotonic':started,'started_unix_ns':wall,'elapsed_seconds':elapsed,
            'pid':os.getpid(),'thread':threading.get_ident(),'status':status,
            **_context.get(),'detail':'debug','span_id':span_id,'parent_span_id':parent,'measurements':values})


def _record(path, event):
    global _fd, _path, _configuration
    effective=configuration()
    with _lock:
        if _fd is None:
            target = Path(path)
            if not target.is_absolute():
                raise ValueError('stage timing requires an absolute output path')
            _fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            _path = path
            _configuration=effective
            # Separate file keeps the existing event schema and readers stable.
            snapshot={'schema':'saea-observability-config-v1','effective':effective,
                'configuration_sha256':hashlib.sha256(json.dumps(effective,sort_keys=True).encode()).hexdigest(),
                'capabilities':{'client_header_and_body_intervals':effective['client_detail'],
                    'per_request_server_queue':False,'per_request_server_prefill':False,
                    'per_request_server_decode':False,'ttft':False},
                'scientific_checkpoint_policy':'unchanged; independent fsync',
                'timing_durability':'kernel write; not per-event fsync',
                'server_metrics_policy':'separate collector; interval configuration does not start polling'}
            with target.with_name(target.name+'.config.json').open('x') as stream:
                json.dump(snapshot,stream,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        if path != _path:
            raise ValueError('stage timing path changed inside one process')
        if effective!=_configuration:raise ValueError('observability profile changed inside one process')
        event={**event,'observability_mode':effective['mode'],**{k:effective[k] for k in ('run_id','attempt_id') if k in effective}}
        raw = (json.dumps(event, sort_keys=True) + '\n').encode()
        offset = 0
        while offset < len(raw):
            written = os.write(_fd, raw[offset:])
            if written <= 0:
                raise OSError('short stage timing write')
            offset += written


def measure(stage, identity=None):
    """Measure a synchronous stage without altering its inputs or return value."""
    def decorate(fn):
        @wraps(fn)
        def invoke(*args, **kwargs):
            path = os.environ.get('SAEA_STAGE_TIMING_PATH')
            if not path:
                return fn(*args, **kwargs)
            context = dict(_context.get())
            if identity is not None:
                context.update(identity(*args, **kwargs))
            token = _context.set(context)
            started = time.perf_counter()
            status = 'ok'
            try:
                return fn(*args, **kwargs)
            except BaseException:
                status = 'error'
                raise
            finally:
                elapsed = time.perf_counter() - started
                _context.reset(token)
                _record(path, {'schema': 'saea-stage-timing-v1', 'stage': stage,
                    'started_monotonic': started, 'elapsed_seconds': elapsed,
                    'pid': os.getpid(), 'thread': threading.get_ident(),
                    'status': status, **context})
        return invoke
    return decorate


@contextmanager
def measured_lock(lock, stage, identity=None):
    """Record lock acquisition wait only; always release even if timing I/O fails.

    The protected body is deliberately excluded. Disabled timing does not inspect
    identity or touch the timing sink. This keeps existing locking semantics.
    """
    if not os.environ.get('SAEA_STAGE_TIMING_PATH'):
        with lock:
            yield
        return
    acquired = False

    @measure(stage, identity)
    def acquire():
        nonlocal acquired
        lock.acquire()
        acquired = True

    try:
        acquire()
        yield
    finally:
        if acquired:
            lock.release()
