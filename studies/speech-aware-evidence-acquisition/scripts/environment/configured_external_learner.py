"""One durable external teacher request; never promotes a candidate or retries spend."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import time
import urllib.error
import urllib.parse
import urllib.request


def persist(path, data):
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2).encode() + b'\n'
    with path.open('xb') as handle:
        os.chmod(path, 0o600)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class TotalDeadlineExceeded(TimeoutError):
    pass


def deadline_handler(signum, frame):
    raise TotalDeadlineExceeded('external learner total wall deadline exceeded')


def read_stream(response, root):
    """Journal every SSE line before interpreting it; incomplete streams stay unknown."""
    content, reasoning, usage, model, response_id = [], [], None, None, None
    finish = None
    pending = []
    done = False
    with (root/'response.sse').open('xb') as journal:
        os.chmod(root/'response.sse', 0o600)
        for line in response:
            journal.write(line)
            journal.flush()
            os.fsync(journal.fileno())
            value = line.decode('utf-8').rstrip('\r\n')
            if value.startswith('data:'):
                pending.append(value[5:].lstrip(' '))
                continue
            if value or not pending:
                continue
            payload = '\n'.join(pending)
            pending.clear()
            if payload == '[DONE]':
                done = True
                break
            event = json.loads(payload)
            if 'error' in event:
                raise ValueError('stream provider error; see private journal')
            if event.get('id'):
                if response_id is not None and response_id != event['id']:
                    raise ValueError('stream response identity changed')
                response_id = event['id']
            if event.get('model'):
                if model is not None and model != event['model']:
                    raise ValueError('stream model changed')
                model = event['model']
            if event.get('usage') is not None:
                usage = event['usage']
            for choice in event.get('choices', []):
                if choice.get('index') != 0:
                    raise ValueError('only one teacher completion supported')
                delta = choice.get('delta', {})
                if delta.get('content') is not None:
                    content.append(delta['content'])
                if delta.get('reasoning_content') is not None:
                    reasoning.append(delta['reasoning_content'])
                if choice.get('finish_reason') is not None:
                    finish = choice['finish_reason']
    if not done or finish is None:
        raise ValueError('incomplete stream; no retry')
    result = {'id': response_id, 'model': model, 'usage': usage, 'choices': [
        {'index': 0, 'finish_reason': finish, 'message': {'role': 'assistant',
         'content': ''.join(content), 'reasoning_content': ''.join(reasoning)}}]}
    persist(root/'response.json', result)
    return result, hashlib.sha256((root/'response.sse').read_bytes()).hexdigest()


def execute(config_path):
    cfg = json.loads(Path(config_path).read_bytes())
    keys = {'schema', 'endpoint', 'model', 'api_key_env', 'messages', 'max_tokens',
            'timeout_seconds', 'output_root'}
    if not keys <= set(cfg) <= keys | {'stream', 'total_timeout_seconds', 'thinking'} or cfg['schema'] != 'saea-external-learner-v1':
        raise ValueError('exact external learner configuration required')
    if type(cfg.get('stream', False)) is not bool:
        raise ValueError('boolean stream setting required')
    if 'total_timeout_seconds' in cfg and (type(cfg['total_timeout_seconds']) is not int
            or cfg['total_timeout_seconds'] <= 0):
        raise ValueError('positive total wall deadline required')
    if 'thinking' in cfg and cfg['thinking'] not in ({'type': 'enabled'}, {'type': 'disabled'}):
        raise ValueError('explicit provider thinking mode required')
    url = urllib.parse.urlsplit(cfg['endpoint'])
    if (url.scheme != 'https' or not url.hostname or url.username or url.password
            or url.query or url.fragment or not url.path.endswith('/chat/completions')):
        raise ValueError('HTTPS chat endpoint without credentials required')
    if (not re.fullmatch(r'[A-Z][A-Z0-9_]*', cfg['api_key_env'])
            or not isinstance(cfg['model'], str) or not cfg['model']):
        raise ValueError('model and credential environment reference required')
    for key in ('max_tokens', 'timeout_seconds'):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError('positive finite request limits required')
    pin = cfg['messages']
    if set(pin) != {'path', 'sha256'}:
        raise ValueError('pinned messages required')
    raw = Path(pin['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin['sha256']:
        raise ValueError('messages changed')
    messages = json.loads(raw)
    if not isinstance(messages, list) or not messages or any(
            not isinstance(m, dict) or set(m) != {'role', 'content'}
            or m['role'] not in ('system', 'user', 'assistant')
            or not isinstance(m['content'], str) for m in messages):
        raise ValueError('explicit text-only teacher messages required')
    secret = os.environ.get(cfg['api_key_env'])
    if not secret:
        raise ValueError('credential environment variable is unset')
    root = Path(cfg['output_root'])
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink() or root.stat().st_mode & 0o077:
        raise ValueError('private nonsymlink output directory required')
    with (root/'request.guard').open('a+b') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        intent = {'config': cfg, 'messages_sha256': pin['sha256']}
        if (root/'intent.json').exists():
            if json.loads((root/'intent.json').read_bytes()) != intent:
                raise ValueError('existing request identity changed')
            if (root/'result.json').exists():
                return json.loads((root/'result.json').read_bytes())
            raise ValueError('previous dispatch may have spent; reconcile without retry')
        body = {'model': cfg['model'], 'messages': messages, 'max_tokens': cfg['max_tokens'],
                'stream': cfg.get('stream', False)}
        if body['stream']:
            body['stream_options'] = {'include_usage': True}
        if 'thinking' in cfg:
            body['thinking'] = cfg['thinking']
        persist(root/'request.json', body)
        persist(root/'intent.json', intent)
        req = urllib.request.Request(cfg['endpoint'], data=json.dumps(body).encode(),
            headers={'Authorization': 'Bearer ' + secret, 'Content-Type': 'application/json'}, method='POST')
        started = time.monotonic()
        previous_handler = None
        if 'total_timeout_seconds' in cfg:
            previous_handler = signal.signal(signal.SIGALRM, deadline_handler)
            signal.setitimer(signal.ITIMER_REAL, cfg['total_timeout_seconds'])
        try:
            with urllib.request.build_opener(NoRedirect()).open(req, timeout=cfg['timeout_seconds']) as response:
                persist(root/'headers.json', {'http_status': response.status,
                    'request_id': response.headers.get('x-request-id'),
                    'content_type': response.headers.get('content-type')})
                if body['stream']:
                    result, response_sha = read_stream(response, root)
                else:
                    wire = response.read()
                    # Preserve complete wire bytes before interpretation, even for malformed replies.
                    with (root/'response.bin').open('xb') as handle:
                        os.chmod(root/'response.bin', 0o600)
                        handle.write(wire); handle.flush(); os.fsync(handle.fileno())
                    result = json.loads(wire)
                    response_sha = hashlib.sha256(wire).hexdigest()
                summary = {'state': 'response_received', 'http_status': response.status,
                    'request_id': response.headers.get('x-request-id'), 'model': result.get('model'),
                    'usage': result.get('usage'), 'response_sha256': response_sha,
                    'finish_reasons': [c.get('finish_reason') for c in result.get('choices', [])],
                    'candidate_accepted': False, 'semantic_validation': 'pending'}
        except urllib.error.HTTPError as error:
            summary = {'state': 'http_error', 'http_status': error.code, 'usage': None, 'automatic_retry': False}
        except Exception as error:
            summary = {'state': 'unknown', 'error_type': type(error).__name__, 'usage': None, 'automatic_retry': False}
        finally:
            if previous_handler is not None:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_handler)
        summary['elapsed_seconds'] = time.monotonic() - started
        persist(root/'result.json', summary)
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.config)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result['state'] == 'response_received' else 1)
