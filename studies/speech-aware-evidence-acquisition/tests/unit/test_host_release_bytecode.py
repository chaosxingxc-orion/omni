"""Host entrypoints must not add Python caches to writable signed source."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

STUDY = Path(__file__).resolve().parents[2]
ENTRIES = ('configured_progress_observer.py', 'configured_job_queue.py',
           'configured_queue_auto_bind.py', 'configured_terminal_settlement.py',
           'configured_checkpoint_successor.py', 'configured_queue_handoff.py')


@pytest.mark.parametrize('entry', ENTRIES)
def test_host_entry_does_not_write_release_bytecode(tmp_path, entry):
    study = tmp_path / 'study'
    for name in ('src', 'scripts/environment', 'scripts/analysis'):
        shutil.copytree(STUDY / name, study / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    env = os.environ.copy()
    env.pop('PYTHONDONTWRITEBYTECODE', None)
    env.pop('PYTHONPYCACHEPREFIX', None)
    env['PYTHONPATH'] = str(study / 'src')
    script = study / 'scripts/environment' / entry
    run = subprocess.run([sys.executable, str(script), '--help'], env=env, text=True, capture_output=True)
    assert run.returncode == 0, run.stderr
    assert not list(study.rglob('*.pyc'))


def test_unprotected_observer_reproduces_bytecode_creation(tmp_path):
    study = tmp_path / 'study'
    for name in ('src', 'scripts/environment', 'scripts/analysis'):
        shutil.copytree(STUDY / name, study / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    script = study / 'scripts/environment/configured_progress_observer.py'
    source = script.read_text()
    marker = '# Host tools import pinned release modules; keep its signed file set immutable.\nimport sys\nsys.dont_write_bytecode = True\n'
    assert source.count(marker) == 1
    script.write_text(source.replace(marker, 'import sys\n', 1))
    for sibling in (study / 'scripts/environment').glob('*.py'):
        if sibling != script:
            text = sibling.read_text()
            sibling.write_text(text.replace(marker, 'import sys\n', 1))
    env = os.environ.copy()
    env.pop('PYTHONDONTWRITEBYTECODE', None)
    env.pop('PYTHONPYCACHEPREFIX', None)
    env['PYTHONPATH'] = str(study / 'src')
    result = subprocess.run([sys.executable, str(script), '--help'], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert (study / 'scripts/environment/__pycache__').is_dir()
    assert list(study.rglob('*.pyc'))
