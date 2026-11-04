"""One durable external teacher request; never promotes a candidate or retries spend."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
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


def execute(config_path):
    cfg = json.loads(Path(config_path).read_bytes())
    keys = {'schema', 'endpoint', 'model', 'api_key_env', 'messages', 'max_tokens',
            'timeout_seconds', 'output_root'}
    if set(cfg) != keys or cfg['schema'] != 'saea-external-learner-v1':
        raise ValueError('exact external learner configuration required')
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
        body = {'model': cfg['model'], 'messages': messages, 'max_tokens': cfg['max_tokens'], 'stream': False}
        persist(root/'request.json', body)
        persist(root/'intent.json', intent)
        req = urllib.request.Request(cfg['endpoint'], data=json.dumps(body).encode(),
            headers={'Authorization': 'Bearer ' + secret, 'Content-Type': 'application/json'}, method='POST')
        started = time.monotonic()
        try:
            with urllib.request.build_opener(NoRedirect()).open(req, timeout=cfg['timeout_seconds']) as response:
                wire = response.read()
                # Persist complete response before JSON interpretation, including malformed replies.
                with (root/'response.bin').open('xb') as handle:
                    os.chmod(root/'response.bin', 0o600)
                    handle.write(wire); handle.flush(); os.fsync(handle.fileno())
                result = json.loads(wire)
                summary = {'state': 'response_received', 'http_status': response.status,
                    'request_id': response.headers.get('x-request-id'), 'model': result.get('model'),
                    'usage': result.get('usage'), 'response_sha256': hashlib.sha256(wire).hexdigest(),
                    'finish_reasons': [c.get('finish_reason') for c in result.get('choices', [])],
                    'candidate_accepted': False, 'semantic_validation': 'pending'}
        except urllib.error.HTTPError as error:
            summary = {'state': 'http_error', 'http_status': error.code, 'usage': None, 'automatic_retry': False}
        except Exception as error:
            summary = {'state': 'unknown', 'error_type': type(error).__name__, 'usage': None, 'automatic_retry': False}
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
