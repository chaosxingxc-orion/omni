"""Response grammar must not silently accept prose or partial teacher objects."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/analysis/observe_train_policy_teachers.py'
spec = importlib.util.spec_from_file_location('observe_train_policy_teachers', SCRIPT)
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


@pytest.mark.parametrize('raw', [
    '{"candidates": [], "note": "原文"}',
    ' \n{"candidates": [], "note": "原文"}\t',
    '```json\n{"candidates": [], "note": "原文"}\n```',
    '```json\r\n{"candidates": [], "note": "原文"}\r\n```',
    '\n```\n{"candidates": [], "note": "原文"}\n```\n',
])
def test_complete_single_object(raw):
    assert observer.parse_proposal(raw) == {'candidates': [], 'note': '原文'}


@pytest.mark.parametrize('raw', [
    'prefix {"candidates": []}', '{"candidates": []} suffix',
    'prefix\n```json\n{}\n```', '```json\n{}\n```\nsuffix',
    '```json\n{}\n```\n```json\n{}\n```', '{} {}',
    '```json\n{}', '```python\n{}\n```', '```json {} ```',
    '{"candidates": [}', '[]', 'null', '42', '"object"', '',
    '{"value": NaN}', '```json\n{} garbage\n```', None,
])
def test_reject_garbage_partial_or_non_object(raw):
    with pytest.raises((ValueError, TypeError)):
        observer.parse_proposal(raw)
