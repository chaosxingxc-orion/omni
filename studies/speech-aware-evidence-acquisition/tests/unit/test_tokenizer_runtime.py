"""Configured tokenizer process preserves counts/identity and fails closed."""
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import sys
import textwrap
import weakref

import pytest

from speech_aware_evidence_acquisition.rails import grounded_synthesis_plan as plan
from speech_aware_evidence_acquisition.rails import tokenizer_runtime as m


VERSIONS = {'transformers': 'pinned-transformers', 'tokenizers': 'pinned-tokenizers', 'jinja2': 'pinned-jinja2'}


@pytest.fixture
def local_tokenizer(tmp_path):
    directory = tmp_path/'tokenizer'
    directory.mkdir()
    (directory/'tokenizer.json').write_text('{}')
    (directory/'chat_template.json').write_text(json.dumps({'chat_template': 'exact template'}))
    return directory


def interpreter(tmp_path, source):
    path = tmp_path/'selected-python'
    path.write_text('#!' + sys.executable + '\n' + textwrap.dedent(source))
    path.chmod(0o700)
    return str(path)


@pytest.fixture
def configured_python(tmp_path):
    return interpreter(tmp_path, '''
        import importlib.metadata, json, os, pathlib, runpy, sys, types
        assert 'SAEA_TOKENIZER_PYTHON' not in os.environ
        assert os.environ['HF_HUB_OFFLINE'] == os.environ['TRANSFORMERS_OFFLINE'] == '1'
        assert sys.argv[1:3] == ['-I', '-u']
        versions = {'transformers': 'pinned-transformers', 'tokenizers': 'pinned-tokenizers', 'jinja2': 'pinned-jinja2'}
        importlib.metadata.version = versions.__getitem__
        class ExactTokenizer:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                assert kwargs == {'trust_remote_code': True, 'local_files_only': True}
                with (pathlib.Path(path)/'loads').open('a') as stream:
                    stream.write('load\\n')
                return cls()
            def apply_chat_template(self, messages, **kwargs):
                assert self.chat_template == 'exact template'
                assert kwargs == {'tokenize': False, 'add_generation_prompt': True}
                if messages == [{'role': 'user', 'content': 'FAIL'}]:
                    raise ValueError('private prompt must not be exposed')
                return json.dumps(messages, ensure_ascii=False) + '<assistant>'
            def encode(self, rendered, **kwargs):
                assert kwargs == {'add_special_tokens': False, 'truncation': False}
                return list(rendered.encode('utf-8'))
        sys.modules['transformers'] = types.SimpleNamespace(AutoTokenizer=ExactTokenizer)
        sys.argv = sys.argv[3:]
        runpy.run_path(sys.argv[0], run_name='__main__')
    ''')


def expected_count(messages):
    return len((json.dumps(messages, ensure_ascii=False) + '<assistant>').encode())


def test_selected_worker_loads_once_preserves_full_identity_and_chat_count(local_tokenizer, configured_python, monkeypatch):
    monkeypatch.setenv(m.TOKENIZER_PYTHON_ENV, configured_python)
    count, identity = plan.load_token_counter(local_tokenizer)
    process = count._process
    try:
        expected = {name: hashlib.sha256((local_tokenizer/name).read_bytes()).hexdigest()
                    for name in ('tokenizer.json', 'chat_template.json')}
        expected.update(runtime_versions=VERSIONS, tokenizer_class='ExactTokenizer',
                        rendering={'add_generation_prompt': True, 'add_special_tokens': False, 'truncation': False})
        assert identity == expected
        for messages in ([{'role': 'user', 'content': '你好'}], [{'role': 'system', 'content': 'full'},
                          {'role': 'user', 'content': 'all messages'}], []):
            assert count(messages) == expected_count(messages)
        assert (local_tokenizer/'loads').read_text() == 'load\n'
        assert count._process is process and process.poll() is None
    finally:
        count.close()
    assert process.poll() is not None
    with pytest.raises(m.TokenizerRuntimeError, match='closed'):
        count([])


def test_unconfigured_helper_keeps_inprocess_behavior(monkeypatch):
    monkeypatch.delenv(m.TOKENIZER_PYTHON_ENV, raising=False)
    marker = (lambda messages: 17, {'original': True})
    observed = []
    monkeypatch.setattr(m, 'load_local_token_counter', lambda path: observed.append(path) or marker)
    assert plan.load_token_counter('original-path') is marker
    assert observed == ['original-path']


@pytest.mark.parametrize('selected', ['', 'python3', '/definitely/missing/python'])
def test_invalid_configured_interpreter_never_falls_back(local_tokenizer, selected, monkeypatch):
    monkeypatch.setenv(m.TOKENIZER_PYTHON_ENV, selected)
    monkeypatch.setattr(m, 'load_local_token_counter', lambda path: pytest.fail('silent fallback'))
    with pytest.raises(plan.LearningError, match='absolute interpreter'):
        plan.load_token_counter(local_tokenizer)


def test_bridge_refuses_nonlocal_model_before_spawn(tmp_path, configured_python):
    with pytest.raises(m.TokenizerRuntimeError, match='local directory'):
        m.SubprocessTokenCounter(tmp_path/'missing-model', configured_python)


def test_shared_worker_serializes_concurrent_counts(local_tokenizer, configured_python):
    messages = [[{'role': 'user', 'content': str(i)}] for i in range(20)]
    with m.SubprocessTokenCounter(local_tokenizer, configured_python) as count:
        with ThreadPoolExecutor(max_workers=4) as pool:
            assert list(pool.map(count, messages)) == list(map(expected_count, messages))
    assert (local_tokenizer/'loads').read_text() == 'load\n'


def test_tokenizer_error_closes_worker_without_prompt_or_retry(local_tokenizer, configured_python):
    count = m.SubprocessTokenCounter(local_tokenizer, configured_python)
    process = count._process
    with pytest.raises(m.TokenizerRuntimeError) as caught:
        count([{'role': 'user', 'content': 'FAIL'}])
    assert 'private prompt' not in str(caught.value)
    assert process.poll() is not None and count._closed
    with pytest.raises(m.TokenizerRuntimeError, match='closed'):
        count([])
    assert (local_tokenizer/'loads').read_text() == 'load\n'


def test_worker_lifetime_follows_callable_collection(local_tokenizer, configured_python):
    count = m.SubprocessTokenCounter(local_tokenizer, configured_python)
    process, reference = count._process, weakref.ref(count)
    del count
    gc.collect()
    assert reference() is None and process.poll() is not None


def test_context_error_closes_worker(local_tokenizer, configured_python):
    with pytest.raises(ValueError, match='caller failed'):
        with m.SubprocessTokenCounter(local_tokenizer, configured_python) as count:
            process = count._process
            raise ValueError('caller failed')
    assert process.poll() is not None


def test_large_request_refused_before_any_count_and_closes(local_tokenizer, configured_python, monkeypatch):
    with m.SubprocessTokenCounter(local_tokenizer, configured_python) as count:
        process = count._process
        monkeypatch.setattr(m, 'MAX_REQUEST_BYTES', 128)
        with pytest.raises(m.TokenizerRuntimeError, match='byte limit'):
            count([{'role': 'user', 'content': 'x' * 200}])
        assert process.poll() is not None


@pytest.mark.parametrize('failure', ['exit', 'invalid_json', 'wrong_id', 'duplicate', 'oversize', 'negative', 'bool', 'extra'])
def test_corrupt_worker_response_fails_closed(tmp_path, local_tokenizer, failure):
    source = '''
        import json, sys
        first = json.loads(sys.stdin.buffer.readline())
        print(json.dumps({'id': first['id'], 'ok': True, 'identity': {'fixture': True}}), flush=True)
        request = json.loads(sys.stdin.buffer.readline())
    '''
    actions = {
        'exit': 'raise SystemExit(1)',
        'invalid_json': "print('not JSON', flush=True)",
        'wrong_id': "print(json.dumps({'id': 99, 'ok': True, 'count': 1}), flush=True)",
        'duplicate': "print('{\"id\":2,\"id\":2,\"ok\":true,\"count\":1}', flush=True)",
        'oversize': "print('x' * 70000, flush=True)",
        'negative': "print(json.dumps({'id': 2, 'ok': True, 'count': -1}), flush=True)",
        'bool': "print(json.dumps({'id': 2, 'ok': True, 'count': True}), flush=True)",
        'extra': "print(json.dumps({'id': 2, 'ok': True, 'count': 1, 'extra': 'no'}), flush=True)",
    }
    python = interpreter(tmp_path, textwrap.dedent(source) + actions[failure] + '\n')
    count = m.SubprocessTokenCounter(local_tokenizer, python)
    process = count._process
    with pytest.raises(m.TokenizerRuntimeError):
        count([])
    assert count._closed and process.poll() is not None


@pytest.mark.parametrize('phase', ['startup', 'count'])
def test_worker_timeout_closes_without_restart(tmp_path, local_tokenizer, phase):
    source = '''
        import json, sys, time
        request = json.loads(sys.stdin.buffer.readline())
    '''
    if phase == 'count':
        source = textwrap.dedent(source) + "print(json.dumps({'id': request['id'], 'ok': True, 'identity': {'fixture': True}}), flush=True)\nrequest = json.loads(sys.stdin.buffer.readline())\n"
    python = interpreter(tmp_path, textwrap.dedent(source) + 'time.sleep(10)\n')
    if phase == 'startup':
        with pytest.raises(m.TokenizerRuntimeError, match='timed out'):
            m.SubprocessTokenCounter(local_tokenizer, python, timeout_seconds=.2)
    else:
        count = m.SubprocessTokenCounter(local_tokenizer, python, timeout_seconds=.2)
        process = count._process
        with pytest.raises(m.TokenizerRuntimeError, match='timed out'):
            count([])
        assert process.poll() is not None


@pytest.mark.parametrize('timeout', [0, -1, True, float('inf'), float('nan')])
def test_invalid_timeout_refused(local_tokenizer, configured_python, timeout):
    with pytest.raises(m.TokenizerRuntimeError, match='timeout'):
        m.SubprocessTokenCounter(local_tokenizer, configured_python, timeout_seconds=timeout)
