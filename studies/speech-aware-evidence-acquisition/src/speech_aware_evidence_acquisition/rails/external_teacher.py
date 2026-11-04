"""External chat receipts for ExperienceLearner, without native-model impersonation."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

from .tfgrpo_learning import LearningError, UnknownLearningContact, canonical, sha


def require(condition, message):
    if not condition:
        raise LearningError(message)


def read(path):
    return json.loads(path.read_bytes())


class ExternalTeacher:
    """Pinned external executor and configuration, one private directory per request.

    The executor is the existing configured_external_learner.py, whose durable
    intent prevents duplicate sends. Reconciliation only reads verified bytes.
    This adapter does not claim tokenizer preflight or native FIT admission.
    """
    def __init__(self, root, *, config, executor):
        self.root = Path(root).resolve()
        require(isinstance(executor, dict) and set(executor) == {'path', 'sha256'},
                'pinned external executor required')
        path = Path(executor['path']).resolve()
        require(hashlib.sha256(path.read_bytes()).hexdigest() == executor['sha256'],
                'external executor changed')
        spec = importlib.util.spec_from_file_location('saea_external_executor', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.executor = module
        self.config = copy.deepcopy(config)
        require(not {'messages', 'output_root'} & self.config.keys(),
                'adapter supplies request-specific messages and output root')
        required = {'schema', 'endpoint', 'model', 'api_key_env', 'max_tokens',
                    'timeout_seconds', 'stream', 'research_prompt'}
        require(required <= self.config.keys() <= required | {'thinking', 'total_timeout_seconds',
                    'journal_batch_bytes', 'journal_batch_seconds'},
                'explicit external research configuration required')
        require(self.config['schema'] == 'saea-external-learner-v1'
                and type(self.config['max_tokens']) is int and self.config['max_tokens'] >= 131072,
                '128K external research budget required')
        policy = self.config['research_prompt']
        require(isinstance(policy, dict) and set(policy) == {'path', 'sha256'}
                and Path(policy['path']).is_absolute(), 'absolute pinned research prompt required')
        require(hashlib.sha256(Path(policy['path']).read_bytes()).hexdigest() == policy['sha256'],
                'research prompt changed')
        self.policy_text = Path(policy['path']).read_text()
        self.model_id = self.config['model']
        self.completion_cap = self.config['max_tokens']
        require(isinstance(self.model_id, str) and self.model_id.strip(), 'teacher identity required')
        self.learning_config_sha256 = sha({'schema': 'saea-external-teacher-binding-v1',
            'config': self.config, 'executor_sha256': executor['sha256']})

    def _binding(self, request):
        require(isinstance(request, dict) and request.get('schema') == 'saea-tfgrpo-learning-request-v1'
                and request.get('model_id') == self.model_id
                and request.get('learning_config_sha256') == self.learning_config_sha256,
                'external learning request identity differs')
        messages = request.get('messages')
        require(isinstance(messages, list) and messages and all(
            isinstance(m, dict) and set(m) == {'role', 'content'}
            and m['role'] in ('system', 'user', 'assistant') and isinstance(m['content'], str)
            for m in messages), 'complete external text messages required')
        directory = self.root/sha(request)
        raw = canonical(messages)
        cfg = {**self.config, 'messages': {'path': str(directory/'messages.json'),
            'sha256': hashlib.sha256(raw).hexdigest()}, 'output_root': str(directory/'response')}
        return directory, cfg

    def _put(self, path, value):
        if path.exists():
            require(read(path) == value, 'external durable input changed')
        else:
            self.executor.persist(path, value)

    def preflight(self, request):
        """Pin bytes only; caller must provide verified provider capacity separately."""
        directory, cfg = self._binding(request)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.mkdir(exist_ok=True, mode=0o700)
        require(not directory.is_symlink() and not directory.stat().st_mode & 0o077,
                'private external request directory required')
        # persist() pretty-prints JSON; the message pin must cover those actual bytes.
        self._put(directory/'messages.json', request['messages'])
        cfg['messages']['sha256'] = hashlib.sha256((directory/'messages.json').read_bytes()).hexdigest()
        self._put(directory/'learner-request.json', request)
        self._put(directory/'config.json', cfg)
        return {'schema': 'saea-external-message-binding-v1', 'request_sha256': sha(request),
                'messages_sha256': cfg['messages']['sha256'], 'native_fit_completed': False,
                'provider_capacity_verified': False}

    def __call__(self, request):
        self.preflight(request)
        directory, _ = self._binding(request)
        self.executor.execute(directory/'config.json')
        observed = self.reconcile(request)
        if observed is None:
            raise UnknownLearningContact('external response unavailable; preserve costs and never retry')
        return observed['response']

    def reconcile(self, request):
        directory, expected = self._binding(request)
        root = directory/'response'
        if not (root/'result.json').exists():
            return None
        require(read(directory/'learner-request.json') == request, 'external learner request changed')
        raw = (directory/'messages.json').read_bytes()
        require(json.loads(raw) == request['messages'], 'external evidence changed')
        expected['messages']['sha256'] = hashlib.sha256(raw).hexdigest()
        require(read(directory/'config.json') == expected, 'external config changed')
        require(read(root/'intent.json') == {'config': expected,
            'messages_sha256': expected['messages']['sha256']}, 'external intent changed')
        result = read(root/'result.json')
        if result['state'] != 'response_received':
            return None
        expected_body = {'model': self.model_id,
            'messages': [{'role': 'system', 'content': self.policy_text}, *request['messages']],
            'max_tokens': expected['max_tokens'], 'stream': expected['stream']}
        if expected['stream']:
            expected_body['stream_options'] = {'include_usage': True}
        if 'thinking' in expected:
            expected_body['thinking'] = expected['thinking']
        require(read(root/'request.json') == expected_body, 'external dispatched messages differ')
        wire = (root/('response.sse' if expected['stream'] else 'response.bin')).read_bytes()
        require(hashlib.sha256(wire).hexdigest() == result['response_sha256'], 'external wire changed')
        if expected['stream']:
            # Parse verified wire without replaying disk writes or fsync operations.
            import io
            import time
            diagnostics = self.executor.ResponseDiagnostics(time.monotonic())
            response, _ = self.executor.read_stream(io.BytesIO(wire), None, diagnostics,
                                                   persist_response=False)
            require(response == read(root/'response.json'), 'external parsed response changed')
        else:
            response = json.loads(wire)
        require(response.get('model') == self.model_id, 'external returned model differs')
        provider_id = response.get('id')
        require(isinstance(provider_id, str) and provider_id, 'provider response identity required')
        choices = response.get('choices')
        require(isinstance(choices, list) and len(choices) == 1
                and choices[0].get('finish_reason') is not None, 'complete single external choice required')
        usage = response.get('usage')
        require(isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0
            for k in ('prompt_tokens', 'completion_tokens')), 'actual provider metering required')
        content = choices[0].get('message', {}).get('content')
        require(isinstance(content, str), 'external text response required')
        observed = {'model_id': self.model_id, 'receipt_id': 'external-chat:'+provider_id,
            'text': content, 'usage': {'calls': 1, 'prompt_tokens': usage['prompt_tokens'],
                'completion_tokens': usage['completion_tokens']},
            'provider_usage': copy.deepcopy(usage), 'provider_response_id': provider_id,
            'finish_reason': choices[0]['finish_reason'], 'receipt_kind': 'external-chat-v1',
            'request_sha256': sha(request), 'wire_response_sha256': result['response_sha256'],
            'learning_config_sha256': self.learning_config_sha256, 'native_fit_completed': False}
        return {'request_sha256': sha(request), 'response': observed, 'response_sha256': sha(observed)}
