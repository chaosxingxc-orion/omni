"""The newcomer drill runs green from a copy of the base at an unrelated path.

This is the acceptance criterion made executable: a collaborator copies the base
anywhere, fills seven profile fields, and the fixed CLI drives an experiment from
`validate` to a terminal state.  The drill itself lives in
``scripts/experiments/onboarding_drill.py`` and is the same code a person runs by
hand, so the test cannot drift from the documented checklist.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DRILL = REPO / "scripts/experiments/onboarding_drill.py"


def run(root):
    result = subprocess.run([sys.executable, "-B", str(DRILL), "--root", str(root), "--keep"],
                            capture_output=True, text=True, timeout=600, cwd=str(REPO),
                            env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})
    return result, json.loads(result.stdout) if result.stdout.strip() else {}


def test_the_drill_passes_every_step(tmp_path):
    result, report = run(tmp_path / "drill")
    assert report, result.stderr[-3000:]
    assert report["failed"] == [], json.dumps(report["steps"], indent=2)
    assert report["ok"] is True and result.returncode == 0


def test_the_drill_needs_only_the_seven_required_fields(tmp_path):
    _, report = run(tmp_path / "drill")
    assert report["profile_fields"] == 7


def test_the_drill_refuses_a_directory_that_is_already_in_use(tmp_path):
    root = tmp_path / "busy"
    root.mkdir()
    (root / "leftover").write_text("x\n")
    result = subprocess.run([sys.executable, "-B", str(DRILL), "--root", str(root)],
                            capture_output=True, text=True, timeout=60, cwd=str(REPO))
    assert result.returncode != 0 and "not empty" in (result.stdout + result.stderr)