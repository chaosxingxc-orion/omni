"""Local text token counting in one explicitly selected Python environment.

Run this file directly as the worker so the tokenizer interpreter never imports
the business package or borrows its site-packages. The JSON pipe is CPU-only;
it neither authorizes nor performs a model/server contact.
"""
from __future__ import annotations

import atexit
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import threading
import time
import weakref

TOKENIZER_PYTHON_ENV = 'SAEA_TOKENIZER_PYTHON'
MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_RESPONSE_BYTES = 64 * 1024
DEFAULT_TIMEOUT_SECONDS = 120.0
_live_counters = weakref.WeakSet()


class TokenizerRuntimeError(ValueError):
    """A configured local tokenizer failed; never fall back or retry."""


def load_local_token_counter(tokenizer_path, *, mode='chat'):
    """The original local tokenizer/template, hashes, versions and count rule."""
    if mode not in ('chat', 'text'):
        raise TokenizerRuntimeError('unsupported tokenizer count mode')
    from transformers import AutoTokenizer
    tokenizer_path = Path(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True, local_files_only=True)
    template_path = tokenizer_path / 'chat_template.json'
    tokenizer.chat_template = json.loads(template_path.read_bytes())['chat_template']
    names = ('tokenizer.json', 'vocab.json', 'merges.txt', 'tokenizer.model',
             'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.json', 'config.json')
    identity = {name: hashlib.sha256((tokenizer_path / name).read_bytes()).hexdigest()
                for name in names if (tokenizer_path / name).is_file()}
    if not ('tokenizer.json' in identity or {'vocab.json', 'merges.txt'} <= identity.keys()
            or 'tokenizer.model' in identity):
        raise TokenizerRuntimeError('tokenizer vocabulary identity missing')
    identity['runtime_versions'] = {name: importlib.metadata.version(name)
                                    for name in ('transformers', 'tokenizers', 'jinja2')}
    identity['tokenizer_class'] = type(tokenizer).__name__
    identity['rendering'] = {'add_generation_prompt': True, 'add_special_tokens': False,
                             'truncation': False}
    if mode == 'text':
        identity['rendering'] = {'mode': 'text', 'add_generation_prompt': False,
                                 'add_special_tokens': False, 'truncation': False}
    def count(messages):
        if mode == 'text':
            if not isinstance(messages, str):
                raise TokenizerRuntimeError('text count requires a string')
            rendered = messages
        else:
            if not isinstance(messages, list):
                raise TokenizerRuntimeError('chat count requires a message list')
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return len(tokenizer.encode(rendered, add_special_tokens=False, truncation=False))
    return count, identity


def load_text_token_counter(tokenizer_path):
    """Count prompt-block text, without adding a chat wrapper or truncating.

    The returned subprocess callable owns its worker and supports close/context
    management. This count is for the instruction contract, never native FIT.
    """
    python = os.environ.get(TOKENIZER_PYTHON_ENV)
    if python is None:
        return load_local_token_counter(tokenizer_path, mode='text')
    count = SubprocessTokenCounter(tokenizer_path, python, mode='text')
    return count, count.identity


def _json_line(value, limit):
    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8') + b'\n'
    if len(raw) > limit:
        raise TokenizerRuntimeError('tokenizer protocol frame exceeds byte limit')
    return raw


def _decode(raw):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise TokenizerRuntimeError('duplicate tokenizer protocol key')
            result[key] = value
        return result
    def invalid_constant(_value):
        raise TokenizerRuntimeError('nonfinite tokenizer protocol value')
    return json.loads(raw.decode('utf-8'), object_pairs_hook=object_pairs, parse_constant=invalid_constant)


class SubprocessTokenCounter:
    """One loaded tokenizer, serialized calls, bounded frames and deadlines.

    The callable owns its child. Explicit close/context exit, garbage collection
    and interpreter shutdown all release it. A failed counter stays closed.
    """

    def __init__(self, tokenizer_path, python, *, timeout_seconds=DEFAULT_TIMEOUT_SECONDS, mode='chat'):
        self._process = None
        self._owner_pid = os.getpid()
        self._lock = threading.Lock()
        self._sequence = 0
        self._closed = False
        if mode not in ('chat', 'text'):
            raise TokenizerRuntimeError('unsupported tokenizer count mode')
        self._mode = mode
        if (not isinstance(python, str) or not Path(python).is_absolute()
                or not Path(python).is_file() or not os.access(python, os.X_OK)):
            raise TokenizerRuntimeError('SAEA_TOKENIZER_PYTHON must name an executable absolute interpreter path')
        if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
                or timeout_seconds <= 0):
            raise TokenizerRuntimeError('tokenizer timeout must be finite and positive')
        self._timeout = float(timeout_seconds)
        path = Path(tokenizer_path).resolve()
        if not path.is_dir():
            raise TokenizerRuntimeError('tokenizer bridge requires an existing local directory')
        environment = dict(os.environ)
        environment.pop(TOKENIZER_PYTHON_ENV, None)
        environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        try:
            argv = [python, '-I', '-u', str(Path(__file__).resolve()), '--worker', str(path)]
            if mode == 'text':
                argv.append('text')
            self._process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                env=environment, bufsize=0, close_fds=True)
            os.set_blocking(self._process.stdin.fileno(), False)
            os.set_blocking(self._process.stdout.fileno(), False)
            _live_counters.add(self)
            reply = self._exchange('identity')
            if set(reply) != {'id', 'ok', 'identity'} or not isinstance(reply['identity'], dict) or not reply['identity']:
                raise TokenizerRuntimeError('tokenizer worker identity response differs')
            self.identity = reply['identity']
            if mode == 'text' and self.identity.get('rendering') != {
                    'mode': 'text', 'add_generation_prompt': False,
                    'add_special_tokens': False, 'truncation': False}:
                raise TokenizerRuntimeError('tokenizer worker text mode identity differs')
        except BaseException:
            self._shutdown()
            raise

    def _exchange(self, operation, **fields):
        if os.getpid() != self._owner_pid:
            self._shutdown()
            raise TokenizerRuntimeError('tokenizer worker cannot be reused after fork')
        with self._lock:
            if self._closed or self._process is None:
                raise TokenizerRuntimeError('tokenizer worker is closed')
            self._sequence += 1
            sequence = self._sequence
            try:
                raw = _json_line({'id': sequence, 'op': operation, **fields}, MAX_REQUEST_BYTES)
                deadline = time.monotonic() + self._timeout
                with selectors.DefaultSelector() as selector:
                    writer, reader = self._process.stdin.fileno(), self._process.stdout.fileno()
                    selector.register(writer, selectors.EVENT_WRITE)
                    sent = 0
                    incoming = bytearray()
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0 or not selector.select(remaining):
                            raise TokenizerRuntimeError('tokenizer worker timed out')
                        if sent < len(raw):
                            try:
                                written = os.write(writer, raw[sent:])
                            except BlockingIOError:
                                continue
                            if written <= 0:
                                raise TokenizerRuntimeError('tokenizer worker pipe closed')
                            sent += written
                            if sent == len(raw):
                                selector.unregister(writer)
                                selector.register(reader, selectors.EVENT_READ)
                            continue
                        try:
                            chunk = os.read(reader, min(65536, MAX_RESPONSE_BYTES + 1 - len(incoming)))
                        except BlockingIOError:
                            continue
                        if not chunk:
                            raise TokenizerRuntimeError('tokenizer worker exited before response')
                        incoming.extend(chunk)
                        if len(incoming) > MAX_RESPONSE_BYTES:
                            raise TokenizerRuntimeError('tokenizer response exceeds byte limit')
                        if b'\n' in incoming:
                            if incoming.count(b'\n') != 1 or not incoming.endswith(b'\n'):
                                raise TokenizerRuntimeError('tokenizer response framing differs')
                            reply = _decode(incoming)
                            break
                if (not isinstance(reply, dict) or type(reply.get('id')) is not int
                        or reply['id'] != sequence or reply.get('ok') is not True):
                    raise TokenizerRuntimeError('tokenizer worker refused request or response identity differs')
                return reply
            except BaseException as error:
                self._shutdown()
                if isinstance(error, (KeyboardInterrupt, SystemExit, TokenizerRuntimeError)):
                    raise
                raise TokenizerRuntimeError('tokenizer worker protocol failed') from error

    def __call__(self, messages):
        if self._mode == 'text':
            reply = self._exchange('count_text', text=messages)
        else:
            reply = self._exchange('count', messages=messages)
        if set(reply) != {'id', 'ok', 'count'} or type(reply['count']) is not int or reply['count'] < 0:
            self.close()
            raise TokenizerRuntimeError('tokenizer worker count response differs')
        return reply['count']

    def _shutdown(self):
        self._closed = True
        process, self._process = self._process, None
        if process is None:
            return
        for pipe in (process.stdin, process.stdout):
            if pipe is not None:
                pipe.close()
        if os.getpid() == self._owner_pid:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
        _live_counters.discard(self)

    def close(self):
        if os.getpid() != self._owner_pid:
            self._shutdown()
            return
        with self._lock:
            self._shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        self.close()

    def __del__(self):
        try:
            self._shutdown()
        except Exception:
            pass


def _close_live_counters():
    for counter in list(_live_counters):
        counter.close()


atexit.register(_close_live_counters)


def worker_main(tokenizer_path, *, mode='chat'):
    """Load once, then serve exact identity/count frames until stdin closes."""
    incoming, outgoing = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # tokenizer diagnostics cannot contaminate the pipe
    os.environ.pop(TOKENIZER_PYTHON_ENV, None)
    try:
        count, identity = load_local_token_counter(tokenizer_path, mode=mode)
        previous = 0
        while True:
            raw = incoming.readline(MAX_REQUEST_BYTES + 1)
            if not raw:
                return 0
            if len(raw) > MAX_REQUEST_BYTES or not raw.endswith(b'\n'):
                raise TokenizerRuntimeError('invalid tokenizer request frame')
            request = _decode(raw)
            if (not isinstance(request, dict) or type(request.get('id')) is not int
                    or request['id'] != previous + 1):
                raise TokenizerRuntimeError('invalid tokenizer request sequence')
            previous = request['id']
            if set(request) == {'id', 'op'} and request['op'] == 'identity':
                reply = {'id': previous, 'ok': True, 'identity': identity}
            elif mode == 'chat' and set(request) == {'id', 'op', 'messages'} and request['op'] == 'count':
                reply = {'id': previous, 'ok': True, 'count': count(request['messages'])}
            elif mode == 'text' and set(request) == {'id', 'op', 'text'} and request['op'] == 'count_text':
                reply = {'id': previous, 'ok': True, 'count': count(request['text'])}
            else:
                raise TokenizerRuntimeError('invalid tokenizer operation')
            outgoing.write(_json_line(reply, MAX_RESPONSE_BYTES))
            outgoing.flush()
    except Exception as error:
        # Do not expose prompts, model paths, environment or exception payloads.
        outgoing.write(_json_line({'id': locals().get('previous', 0), 'ok': False,
                                   'error': type(error).__name__}, MAX_RESPONSE_BYTES))
        outgoing.flush()
        return 1


if __name__ == '__main__':
    if (len(sys.argv) not in (3, 4) or sys.argv[1] != '--worker'
            or (len(sys.argv) == 4 and sys.argv[3] != 'text')):
        raise SystemExit('exact tokenizer worker arguments required')
    raise SystemExit(worker_main(sys.argv[2], mode='text' if len(sys.argv) == 4 else 'chat'))
