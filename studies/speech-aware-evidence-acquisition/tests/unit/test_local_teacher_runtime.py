"""Local runtime identity checks, without package installation or model calls."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest

SCRIPT = Path(__file__).resolve().parents[2]/'scripts/environment/local_teacher_runtime.py'
spec = importlib.util.spec_from_file_location('local_teacher_runtime', SCRIPT)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def test_lock_requires_versions_and_hashes(tmp_path):
    p = tmp_path/'lock'
    p.write_text('regex==1 --hash=sha256:'+'a'*64+'\n')
    assert runtime.requirements(p) == {'regex': '1'}
    p.write_text('regex>=1\n')
    with pytest.raises(ValueError): runtime.requirements(p)


def test_existing_environment_is_never_modified(tmp_path):
    with pytest.raises(ValueError, match='fresh venv'):
        runtime.bootstrap({'venv': str(tmp_path)})


def test_interpreter_and_lock_identity(tmp_path):
    p=tmp_path/'lock';p.write_text('regex==1 --hash=sha256:'+'a'*64+'\n')
    python=str(Path(sys.executable).resolve())
    import platform
    d={'schema':'saea-local-teacher-runtime-v1','python':python,'python_sha256':runtime.sha(python),
       'venv':str(tmp_path/'venv'),'requirements':str(p),'requirements_sha256':runtime.sha(p),
       'python_version':platform.python_version(),'machine':platform.machine()}
    cfg=tmp_path/'runtime.json';cfg.write_text(json.dumps(d))
    assert runtime.config(cfg)==d
    p.write_text(p.read_text()+'# changed\n')
    with pytest.raises(ValueError, match='requirements changed'): runtime.config(cfg)
