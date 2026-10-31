"""Opt-in operational stage timings outside scientific payloads and traces.

Nested intervals overlap; do not sum inclusive durations. The output path must be
fresh and owned by the launcher. No prompt, audio, answer, or exception text is
recorded. A timing write failure propagates instead of silently losing evidence.
"""
from contextvars import ContextVar
from functools import wraps
import json
import os
from pathlib import Path
import threading
import time

_context = ContextVar('saea_stage_timing_context', default={})
_lock = threading.Lock()
_fd = None
_path = None


def _record(path, event):
    global _fd, _path
    with _lock:
        if _fd is None:
            target = Path(path)
            if not target.is_absolute():
                raise ValueError('stage timing requires an absolute output path')
            _fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            _path = path
        if path != _path:
            raise ValueError('stage timing path changed inside one process')
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
