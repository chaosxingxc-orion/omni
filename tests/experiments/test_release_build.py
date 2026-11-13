"""Exercise the actual shell entry with a recording Docker executable."""
import json
import os
import subprocess
from pathlib import Path

from test_contract import REPO


def test_release_build_uses_shared_paths_and_configured_resources(tmp_path):
    shared = tmp_path / 'shared'
    build = shared / 'release-a'
    (build / 'context').mkdir(parents=True)
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    journal = tmp_path / 'commands.jsonl'
    docker = binaries / 'docker'
    docker.write_text('#!/usr/bin/python3\nimport json,os,sys\nwith open(os.environ["MOCK_JOURNAL"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\nprint("fixture-container")\n')
    docker.chmod(0o755)
    env = {**os.environ, 'PATH': str(binaries) + ':' + os.defpath, 'MOCK_JOURNAL': str(journal),
           'OMNI_SHARED_ROOT': str(shared), 'OMNI_BUILD_CPUS': '2-3', 'OMNI_BUILD_MEMORY': '6g'}
    script = REPO / 'docker/saea-base/release/build-release.sh'
    result = subprocess.run(['bash', str(script), str(build), 'saea-unified-release:test'],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in journal.read_text().splitlines()]
    launch = commands[-1]
    assert launch[launch.index('--cpuset-cpus') + 1] == '2-3'
    assert 'type=bind,src=' + str(build) + ',dst=' + str(build) in launch
    assert launch[-4:] == ['saea-unified-release:test', '2-3', '6g', str(build)]
    assert (build / 'launch-custody/container-id.txt').exists()
    # Existing launch custody cannot be overwritten by a second invocation.
    again = subprocess.run(['bash', str(script), str(build), 'saea-unified-release:test'], env=env, capture_output=True)
    assert again.returncode != 0
    assert len(journal.read_text().splitlines()) == 2


def test_build_root_outside_shared_is_rejected_before_docker(tmp_path):
    root = tmp_path / 'outside'
    root.mkdir()
    shared = tmp_path / 'shared'
    shared.mkdir()
    result = subprocess.run(['bash', str(REPO / 'docker/saea-base/release/build-release.sh'), str(root), 'saea-unified-release:test'],
                            env={**os.environ, 'OMNI_SHARED_ROOT': str(shared), 'OMNI_BUILD_CPUS': '0'}, capture_output=True)
    assert result.returncode == 2
    assert not (root / 'launch-custody').exists()
