"""One durable external teacher request; never promotes a candidate or retries spend."""
import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import http.client
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


class ResponseDiagnostics:
    """Local timings only: iterator wait includes buffering, not pure network time."""
    def __init__(self, started):
        self.started = started
        self.data = {
            'schema': 'saea-external-response-diagnostics-v1',
            'headers_elapsed_seconds': None,
            'first_body_data_elapsed_seconds': None,
            'last_body_data_elapsed_seconds': None,
            'body_read_wait_seconds': 0.0,
            'body_write_flush_seconds': 0.0,
            'body_fsync_seconds': 0.0,
            'body_fsync_calls': 0,
            'received_body_bytes': 0,
            'durable_body_bytes': 0,
            'sse_lines': 0,
            'sse_events': 0,
            'reasoning_characters': 0,
            'content_characters': 0,
            'done_received': False,
            'finish_received': False,
            'usage_received': False,
            'partial_exception_bytes': 0,
            'failure_phase': None,
            'timing_scope': 'client-visible body reads; includes buffering and scheduling, not pure network or model time',
        }
        self.phase = 'request-open'

    def read(self, operation):
        self.phase = 'body-read'
        before = time.monotonic()
        try:
            return operation()
        finally:
            self.data['body_read_wait_seconds'] += time.monotonic() - before

    def received(self, raw):
        if raw:
            elapsed = time.monotonic() - self.started
            if self.data['first_body_data_elapsed_seconds'] is None:
                self.data['first_body_data_elapsed_seconds'] = elapsed
            self.data['last_body_data_elapsed_seconds'] = elapsed
            self.data['received_body_bytes'] += len(raw)

    def write(self, handle, raw):
        self.phase = 'body-write-flush'
        before = time.monotonic()
        try:
            handle.write(raw)
            handle.flush()
        finally:
            self.data['body_write_flush_seconds'] += time.monotonic() - before
        self.phase = 'body-fsync'
        before = time.monotonic()
        try:
            os.fsync(handle.fileno())
        finally:
            self.data['body_fsync_seconds'] += time.monotonic() - before
            self.data['body_fsync_calls'] += 1
        self.data['durable_body_bytes'] += len(raw)

    def save_bytes(self, path, raw):
        with path.open('xb') as handle:
            os.chmod(path, 0o600)
            self.write(handle, raw)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def partial(self, error, root):
        if isinstance(error, http.client.IncompleteRead) and error.partial:
            # Separate file: these bytes were not yielded by the failed read.
            raw = error.partial
            self.received(raw)
            self.data['partial_exception_bytes'] += len(raw)
            self.save_bytes(root/'response.partial.bin', raw)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class TotalDeadlineExceeded(TimeoutError):
    pass


def deadline_handler(signum, frame):
    raise TotalDeadlineExceeded('external learner total wall deadline exceeded')


def durable_stream_lines(response, journal, diagnostics, batch_bytes, batch_seconds):
    """Bound pending bytes; persist a batch before exposing its lines to parsing.

    The time threshold is checked on received lines, not a background flush SLA.
    On read error/EOF flush the pending tail; crashes leave the request unknown.
    """
    pending, size = [], 0
    last = time.monotonic()
    iterator = iter(response)
    while True:
        terminal = None
        try:
            line = diagnostics.read(lambda: next(iterator))
        except Exception as error:
            terminal = error
        if terminal is None:
            diagnostics.received(line)
            diagnostics.data['sse_lines'] += 1
            pending.append(line); size += len(line)
        complete_marker = (len(pending) >= 2 and not pending[-1].strip()
                           and pending[-2].strip() == b'data: [DONE]')
        if pending and (terminal is not None or complete_marker or size >= batch_bytes
                        or time.monotonic() - last >= batch_seconds):
            if journal is not None:
                diagnostics.write(journal, b''.join(pending))
            for value in pending:
                yield value
            pending, size = [], 0
            last = time.monotonic()
        if terminal is not None:
            if isinstance(terminal, StopIteration):
                return
            diagnostics.phase = 'body-read'
            raise terminal


def read_stream(response, root, diagnostics, *, persist_response=True,
                journal_batch_bytes=65536, journal_batch_seconds=1):
    """Persist bounded SSE batches before interpretation; incomplete stays unknown."""
    content, reasoning, usage, model, response_id = [], [], None, None, None
    finish = None
    pending = []
    done = False
    wire_hash = hashlib.sha256()
    with ((root/'response.sse').open('xb') if persist_response else nullcontext()) as journal:
        if persist_response:
            os.chmod(root/'response.sse', 0o600)
        for line in durable_stream_lines(response, journal, diagnostics,
                                         journal_batch_bytes, journal_batch_seconds):
            wire_hash.update(line)
            diagnostics.phase = 'sse-parse'
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
                diagnostics.data['done_received'] = True
                break
            event = json.loads(payload)
            diagnostics.data['sse_events'] += 1
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
                diagnostics.data['usage_received'] = True
            for choice in event.get('choices', []):
                if choice.get('index') != 0:
                    raise ValueError('only one teacher completion supported')
                delta = choice.get('delta', {})
                if delta.get('content') is not None:
                    content.append(delta['content'])
                    diagnostics.data['content_characters'] += len(delta['content'])
                if delta.get('reasoning_content') is not None:
                    reasoning.append(delta['reasoning_content'])
                    diagnostics.data['reasoning_characters'] += len(delta['reasoning_content'])
                if choice.get('finish_reason') is not None:
                    finish = choice['finish_reason']
                    diagnostics.data['finish_received'] = True
    diagnostics.phase = 'stream-terminal-validation'
    if not done or finish is None:
        raise ValueError('incomplete stream; no retry')
    result = {'id': response_id, 'model': model, 'usage': usage, 'choices': [
        {'index': 0, 'finish_reason': finish, 'message': {'role': 'assistant',
         'content': ''.join(content), 'reasoning_content': ''.join(reasoning)}}]}
    if persist_response:
        persist(root/'response.json', result)
    return result, (hashlib.sha256((root/'response.sse').read_bytes()).hexdigest()
                    if persist_response else wire_hash.hexdigest())


def analysis_budget_prompt(cfg):
    """Render advisory limits; never claim provider enforcement or trim evidence."""
    if 'analysis_budget' not in cfg:
        return ''
    budget = cfg['analysis_budget']
    required = {'input_context_target_tokens', 'max_reasoning_tokens_target',
                'max_mechanism_dimensions', 'max_hypotheses',
                'representative_evidence_per_hypothesis',
                'counterexamples_per_hypothesis', 'representative_case_target',
                'stop_when_evidence_sufficient'}
    if not isinstance(budget, dict) or set(budget) != required:
        raise ValueError('complete analysis budget required')
    if 'research_prompt' not in cfg:
        raise ValueError('analysis budget requires pinned research prompt')
    for name in required - {'stop_when_evidence_sufficient'}:
        if type(budget[name]) is not int or budget[name] <= 0:
            raise ValueError('positive integer analysis budget required')
    if budget['stop_when_evidence_sufficient'] is not True:
        raise ValueError('analysis budget requires evidence stopping policy')
    if (type(cfg.get('max_tokens')) is not int or cfg['max_tokens'] < 131072
            or budget['max_reasoning_tokens_target'] >= cfg['max_tokens']):
        raise ValueError('analysis reasoning target exceeds output budget or leaves no answer space')
    return ('\n\n## 本次配置的研究预算（提示目标，不是服务端硬限制）\n'
        f"输入材料容量规划目标 {budget['input_context_target_tokens']} tokens；"
        '这不是服务商窗口证明，也不是裁剪阈值。保留完整任务材料，不补无关材料凑数。\n'
        f"思考目标不超过约 {budget['max_reasoning_tokens_target']} tokens，"
        '为最终报告保留充足空间；不要求写满。\n'
        f"主要机制维度至多 {budget['max_mechanism_dimensions']} 个，"
        f"主要假设至多 {budget['max_hypotheses']} 个；"
        f"每个假设优先选 {budget['representative_evidence_per_hypothesis']} 个代表证据和"
        f" {budget['counterexamples_per_hypothesis']} 个最强反例。\n"
        f"深入核查的代表案例目标为 {budget['representative_case_target']} 个，"
        '不是证据访问上限；关键冲突可增查并说明原因，禁止逐case顺序复述。'
        '按机制聚合其余证据，覆盖少数关键反例，不以哈希代替证据内容。\n'
        '证据足以支持、拒绝或暂缓假设即收束；缺证据时列出补证办法，不反复猜测。')


def execute(config_path):
    cfg = json.loads(Path(config_path).read_bytes())
    keys = {'schema', 'endpoint', 'model', 'api_key_env', 'messages', 'max_tokens',
            'timeout_seconds', 'output_root'}
    optional = {'stream', 'total_timeout_seconds', 'thinking', 'research_prompt',
                'analysis_budget', 'journal_batch_bytes', 'journal_batch_seconds'}
    if not keys <= set(cfg) <= keys | optional or cfg['schema'] != 'saea-external-learner-v1':
        raise ValueError('exact external learner configuration required')
    if type(cfg.get('stream', False)) is not bool:
        raise ValueError('boolean stream setting required')
    for name, default, maximum in (('journal_batch_bytes', 65536, 1048576),
                                    ('journal_batch_seconds', 1, 10)):
        if type(cfg.get(name, default)) is not int or not 1 <= cfg.get(name, default) <= maximum:
            raise ValueError('bounded journal batch configuration required')
    if 'total_timeout_seconds' in cfg and (type(cfg['total_timeout_seconds']) is not int
            or cfg['total_timeout_seconds'] <= 0):
        raise ValueError('positive total wall deadline required')
    if 'thinking' in cfg and cfg['thinking'] not in ({'type': 'enabled'}, {'type': 'disabled'}):
        raise ValueError('explicit provider thinking mode required')
    budget_prompt = analysis_budget_prompt(cfg)
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
    if 'research_prompt' in cfg:
        policy = cfg['research_prompt']
        if (not isinstance(policy, dict) or set(policy) != {'path', 'sha256'}
                or not isinstance(policy['path'], str) or not policy['path']):
            raise ValueError('pinned research prompt required')
        policy_path = Path(policy['path'])
        if not policy_path.is_absolute():
            policy_path = Path(config_path).resolve().parent / policy_path
        policy_raw = policy_path.read_bytes()
        if hashlib.sha256(policy_raw).hexdigest() != policy['sha256']:
            raise ValueError('research prompt changed')
        policy_text = policy_raw.decode('utf-8')
        if not policy_text.strip():
            raise ValueError('nonempty research prompt required')
        if cfg['max_tokens'] < 131072:
            raise ValueError('research output budget must be at least 131072')
        # Keep all task evidence intact; persist the effective messages in request.json.
        messages = [{'role': 'system', 'content': policy_text + budget_prompt}, *messages]
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
        diagnostics = ResponseDiagnostics(started)
        previous_handler = None
        if 'total_timeout_seconds' in cfg:
            previous_handler = signal.signal(signal.SIGALRM, deadline_handler)
            signal.setitimer(signal.ITIMER_REAL, cfg['total_timeout_seconds'])
        try:
            with urllib.request.build_opener(NoRedirect()).open(req, timeout=cfg['timeout_seconds']) as response:
                diagnostics.data['headers_elapsed_seconds'] = time.monotonic() - started
                persist(root/'headers.json', {'http_status': response.status,
                    'request_id': response.headers.get('x-request-id'),
                    'content_type': response.headers.get('content-type')})
                if body['stream']:
                    result, response_sha = read_stream(response, root, diagnostics,
                        journal_batch_bytes=cfg.get('journal_batch_bytes', 65536),
                        journal_batch_seconds=cfg.get('journal_batch_seconds', 1))
                else:
                    wire = diagnostics.read(response.read)
                    diagnostics.received(wire)
                    # Preserve complete wire bytes before interpretation, even for malformed replies.
                    diagnostics.save_bytes(root/'response.bin', wire)
                    diagnostics.phase = 'response-parse'
                    result = json.loads(wire)
                    diagnostics.data['usage_received'] = result.get('usage') is not None
                    diagnostics.data['finish_received'] = any(
                        c.get('finish_reason') is not None for c in result.get('choices', []))
                    response_sha = hashlib.sha256(wire).hexdigest()
                summary = {'state': 'response_received', 'http_status': response.status,
                    'request_id': response.headers.get('x-request-id'), 'model': result.get('model'),
                    'usage': result.get('usage'), 'response_sha256': response_sha,
                    'finish_reasons': [c.get('finish_reason') for c in result.get('choices', [])],
                    'candidate_accepted': False, 'semantic_validation': 'pending'}
        except urllib.error.HTTPError as error:
            diagnostics.data['failure_phase'] = diagnostics.phase
            diagnostics.data['headers_elapsed_seconds'] = time.monotonic() - started
            # Provider bodies may contain echoed task data; never copy to stdout/summary.
            try:
                wire = diagnostics.read(error.read)
                diagnostics.received(wire)
                diagnostics.save_bytes(root/'http-error-body.bin', wire)
            except Exception as read_error:
                diagnostics.data['http_error_body_read_error_type'] = type(read_error).__name__
                diagnostics.partial(read_error, root)
            finally:
                error.close()
            summary = {'state': 'http_error', 'http_status': error.code, 'usage': None, 'automatic_retry': False}
        except Exception as error:
            diagnostics.data['failure_phase'] = diagnostics.phase
            diagnostics.partial(error, root)
            summary = {'state': 'unknown', 'error_type': type(error).__name__, 'usage': None, 'automatic_retry': False}
        finally:
            if previous_handler is not None:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_handler)
        summary['elapsed_seconds'] = time.monotonic() - started
        diagnostics.data['elapsed_seconds'] = summary['elapsed_seconds']
        diagnostics.data['terminal_state'] = summary['state']
        persist(root/'diagnostics.json', diagnostics.data)
        persist(root/'result.json', summary)
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.config)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result['state'] == 'response_received' else 1)
