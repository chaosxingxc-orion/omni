"""The two bridges that let host tools read the deployment profile.

``scripts/lib/local_profile.sh`` is what shell entry points source, and
``scripts/lib/local_get.py`` is the process it calls.  A key the loader accepts
but the bridge cannot return is worse than no key at all: the deployment looks
configured and fails at the point of use, so both halves are exercised here
against the same profile the loader test uses.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GET = REPO / "scripts/lib/local_get.py"
BRIDGE = REPO / "scripts/lib/local_profile.sh"


def profile(tmp_path):
    data = {
        "schema": "omni-local-deployment-v1",
        "hosts": {"training": "train-host", "rollout": "rollout-host"},
        "storage": {"shared_root": str(tmp_path / "shared"), "hot_root": str(tmp_path / "hot"),
                    "archive_root": str(tmp_path / "archive")},
        "roots": {"model": str(tmp_path / "models"), "data": str(tmp_path / "data")},
        "image": {"reference": "fixture:1", "digest": "sha256:" + "b" * 64},
        "identity": {"uid": 1000, "gid": 1000},
        "paths": {"port_worktree": str(tmp_path / "worktree")},
        "ports": {"engine_matrix": [8200, 8201]},
    }
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(data))
    return path


def environment(tmp_path, filename):
    return {**os.environ, "OMNI_BASE_ROOT": str(REPO), "OMNI_LOCAL_CONFIG": str(filename),
            "PYTHONDONTWRITEBYTECODE": "1"}


def shell(tmp_path, script, filename):
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                          env=environment(tmp_path, filename))


def test_every_accepted_key_is_published(tmp_path):
    """``--all`` is the bridge's whole cache; a filtered list silently loses open groups."""
    filename = profile(tmp_path)
    result = subprocess.run([sys.executable, "-B", str(GET), "--config", str(filename), "--all"],
                            capture_output=True, text=True, timeout=60, env=environment(tmp_path, filename))
    assert result.returncode == 0, result.stderr
    published = dict(line.split("=", 1) for line in result.stdout.splitlines() if line)
    assert published["paths.port_worktree"] == str(tmp_path / "worktree")
    assert published["storage.shared_root"] == str(tmp_path / "shared")
    assert "ports.engine_matrix" in published


def test_shell_bridge_returns_declared_and_open_group_keys(tmp_path):
    filename = profile(tmp_path)
    script = f"""
        . "{BRIDGE}"
        printf 'shared=%s\\n' "$(lp_get storage.shared_root)"
        printf 'wt=%s\\n' "$(lp_get paths.port_worktree)"
    """
    result = shell(tmp_path, script, filename)
    assert result.returncode == 0, result.stderr
    assert f"shared={tmp_path / 'shared'}" in result.stdout
    assert f"wt={tmp_path / 'worktree'}" in result.stdout


def test_shell_bridge_keeps_a_caller_override_and_reports_a_missing_key(tmp_path):
    filename = profile(tmp_path)
    script = f"""
        . "{BRIDGE}"
        OMNI_ROOT=/explicit/override
        lp_default OMNI_ROOT storage.shared_root
        printf 'root=%s\\n' "$OMNI_ROOT"
        lp_get roots.source
        printf 'rc=%s\\n' "$?"
    """
    result = shell(tmp_path, script, filename)
    assert "root=/explicit/override" in result.stdout
    assert "no value for roots.source" in result.stderr
    assert "rc=2" in result.stdout


def test_shell_bridge_without_a_profile_says_so_instead_of_falling_back(tmp_path):
    script = f"""
        . "{BRIDGE}"
        lp_get storage.shared_root
        printf 'rc=%s\\n' "$?"
    """
    result = shell(tmp_path, script, tmp_path / "absent.json")
    assert result.returncode == 0
    assert "cannot load local profile" in result.stderr
    assert "rc=2" in result.stdout
