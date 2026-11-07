"""Exercise the shipped compatibility hook and real constrained token path."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


HOOK = Path(__file__).resolve().parents[2] / 'scripts/environment/lmfe_vllm_compat'


def run(code, *, hook=True, extra=None):
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD='0', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = str(HOOK) if hook else ''
    if extra:
        env.update(extra)
    return subprocess.run([sys.executable, '-c', code], env=env, text=True,
                          capture_output=True, timeout=240)


def test_original_missing_and_hook_alias():
    code = 'import lmformatenforcer.integrations.vllm'
    missing = run(code, hook=False)
    assert missing.returncode != 0 and 'vllm.transformers_utils.tokenizer' in missing.stderr
    fixed = run(code + '; from vllm.transformers_utils.tokenizer import MistralTokenizer; '
                'from vllm.tokenizers.mistral import MistralTokenizer as actual; assert MistralTokenizer is actual')
    assert fixed.returncode == 0, fixed.stderr


def test_real_qwen_tokenizer_constrained_path():
    model = os.environ['SAEA_TEST_MODEL_DIR']
    code = '''
import json, os
from transformers import AutoTokenizer
from lmformatenforcer import JsonSchemaParser, TokenEnforcer
from lmformatenforcer.integrations.vllm import build_vllm_token_enforcer_tokenizer_data
t = AutoTokenizer.from_pretrained(os.environ['SAEA_TEST_MODEL_DIR'], trust_remote_code=True, local_files_only=True)
data = build_vllm_token_enforcer_tokenizer_data(t, use_bitmask=True, vocab_size=len(t))
schema = {"type":"object", "properties":{"x":{"const":"yes"}}, "required":["x"], "additionalProperties":False}
enforcer = TokenEnforcer(data, JsonSchemaParser(schema))
tokens = t.encode('{"x":"yes"}', add_special_tokens=False)
prefix=[]
for token in tokens:
    assert enforcer.get_allowed_tokens(prefix).is_token_allowed(token), (prefix, token)
    prefix.append(token)
assert not enforcer.get_allowed_tokens([]).is_token_allowed(t.encode('Z', add_special_tokens=False)[0])
print('tokenizer',type(t).__name__,'tokens',len(tokens),'vocab',len(t))
'''
    result = run(code, extra={'SAEA_TEST_MODEL_DIR': model})
    assert result.returncode == 0, result.stderr


def test_existing_legacy_module_is_not_shadowed(tmp_path):
    package = tmp_path / 'vllm/transformers_utils'
    package.mkdir(parents=True)
    (package.parent / '__init__.py').write_text('')
    (package / '__init__.py').write_text('')
    (package / 'tokenizer.py').write_text('sentinel = 17\n')
    result = run('import vllm.transformers_utils.tokenizer as t; assert t.sentinel == 17',
                 extra={'PYTHONPATH': str(HOOK) + os.pathsep + str(tmp_path)})
    assert result.returncode == 0, result.stderr
