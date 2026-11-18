"""The base carries no local machine facts, and the guard cannot silently weaken.

The scanner is loaded by path rather than imported from the package: it lives in
``scripts/`` next to the CLI it protects, and the point of this test is to exercise
that exact file, not a copy of its rules.
"""
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load():
    path = REPO / "scripts/experiments/check_local_leak.py"
    spec = importlib.util.spec_from_file_location("check_local_leak", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scanner = _load()

# The sample leaks this file plants are assembled from pieces rather than written out,
# so that this file is itself clean under the guard it tests: a guard that has to be
# exempted from its own rule is a guard nobody can check by running it.
MOUNT = "/mnt/" + "cephfs"
HOME = "/home/" + "someone"
LOOP = "127.0.0." + "1"
LABEL = "A800" + "I-03"
HOT = "/var/lib/" + "omni-local-state"
ALIAS = "npu" + "2"


def tree(tmp_path, files, allowlist=None):
    """A throwaway root with the given files, plus an allowlist if one is asked for."""
    if allowlist is not None:
        (tmp_path / "configs/quality").mkdir(parents=True, exist_ok=True)
        (tmp_path / "configs/quality/local-leak-allowlist.json").write_text(json.dumps(allowlist))
    for name, body in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return tmp_path


def entry(**extra):
    data = {"reason": "fixture", "owner": "test", "date": "2026-11-18"}
    data.update(extra)
    return data


def rules(result):
    return sorted((item["path"], item["rule"]) for item in result["violations"])


# --- the real base ------------------------------------------------------------------------------

def test_the_base_has_no_local_facts_outside_the_allowlist():
    result = scanner.check(REPO)
    assert result["violations"] == [], rules(result)
    assert result["stale_allowlist"] == [], "an allowlist entry outlived its finding; delete it"
    assert result["clean"] is True


def test_the_allowlist_names_only_sealed_imports():
    """A tree entry must be a verbatim import, not a way to park a file the base owns."""
    _, trees = scanner.load_allowlist(REPO)
    for name in trees:
        assert name.startswith("studies/"), f"{name} is base code and must be cleaned, not allowed"


def test_every_inline_exemption_says_why():
    for item in scanner.check(REPO)["inline"]:
        assert item["reason"].strip(), item


# --- what the rules catch -----------------------------------------------------------------------

def test_each_rule_fires(tmp_path):
    root = tree(tmp_path, {
        "scripts/a.py": f"P = '{MOUNT}/someone'\n",
        "scripts/b.py": f"H = '{HOME}/work'\n",
        "scripts/c.sh": f"ssh {ALIAS} 'ls'\n",
        "scripts/d.json": '{"host": "%s"}\n' % ALIAS[:-1],
        "scripts/e.py": f"S = 'http://{LOOP}:8200'\n",
        "scripts/f.py": f"N = '{LABEL}'\n",
        "scripts/g.py": f"R = '{HOT}/x'\n",
    })
    result = scanner.check(root, ("scripts",))
    assert rules(result) == [
        ("scripts/a.py", "shared-mount"),
        ("scripts/b.py", "home-path"),
        ("scripts/c.sh", "host-alias"),
        ("scripts/d.json", "host-alias"),
        ("scripts/e.py", "loopback"),
        ("scripts/f.py", "machine-label"),
        ("scripts/g.py", "hot-state-root"),
    ]


def test_the_resource_kind_npu_is_not_a_host(tmp_path):
    """`resource: npu` is this codebase's vocabulary; only an ssh target names a machine."""
    root = tree(tmp_path, {"scripts/a.json": '{"resource": "npu", "lane": "npu"}\n'})
    assert scanner.check(root, ("scripts",))["clean"] is True


def test_configured_spellings_are_not_leaks(tmp_path):
    root = tree(tmp_path, {
        "scripts/a.py": "P = '${local.storage.shared_root}/work'\nQ = '${shared_root}/runs'\n",
        "scripts/b.sh": "D=$HOME/omni\nE=${HOME}/omni\nF=/REPLACE_WITH_DATA_ROOT\n",
        "configs/c.json": '{"path": "${config_dir}/x", "run": "${run_root}/y"}\n',
    })
    assert scanner.check(root, ("scripts", "configs"))["clean"] is True


def test_extension_less_files_are_scanned(tmp_path):
    root = tree(tmp_path, {
        "docker/Dockerfile": f"ENV DATA_DIR={MOUNT}/someone\n",
        "docker/Makefile": f"DATA={HOME}/someone\n",
        "docker/notes.unknownext": f"DATA={MOUNT}/someone\n",
    })
    assert rules(scanner.check(root, ("docker",))) == [
        ("docker/Dockerfile", "shared-mount"),
        ("docker/Makefile", "home-path"),
    ]


# --- the inline marker --------------------------------------------------------------------------

# The marker is assembled for the same reason the sample leaks are: a real one
# written here would show up in the base's own ``inline`` audit as an exemption
# nobody granted.
MARK = "# local-leak: " + "allow"


def test_inline_marker_retires_loopback_and_is_reported(tmp_path):
    root = tree(tmp_path, {"tests/a.py": f"URL = 'http://{LOOP}:1'  {MARK} own server\n"})
    result = scanner.check(root, ("tests",))
    assert result["clean"] is True
    assert result["inline"] == [{"path": "tests/a.py", "line": 1, "reason": "own server"}]


def test_inline_marker_cannot_retire_a_mount_or_a_home(tmp_path):
    """Only loopback is exemptable in place: a shared mount never becomes correct by annotation."""
    root = tree(tmp_path, {
        "tests/a.py": f"P = '{MOUNT}/someone'  {MARK} it is fine\n",
        "tests/b.py": f"H = '{HOME}'  {MARK} it is fine\n",
    })
    assert rules(scanner.check(root, ("tests",))) == [
        ("tests/a.py", "shared-mount"),
        ("tests/b.py", "home-path"),
    ]


def test_inline_marker_needs_a_reason(tmp_path):
    root = tree(tmp_path, {"tests/a.py": f"URL = 'http://{LOOP}:1'  {MARK}\n"})
    assert scanner.check(root, ("tests",))["clean"] is False


# --- the allowlist only shrinks -----------------------------------------------------------------

def test_file_entry_is_used_while_it_has_a_finding(tmp_path):
    allow = {"files": {"scripts/a.py": entry()}, "trees": {}}
    root = tree(tmp_path, {"scripts/a.py": f"P = '{MOUNT}/someone'\n"}, allow)
    assert scanner.check(root, ("scripts",))["clean"] is True


def test_file_entry_without_a_finding_is_an_error(tmp_path):
    allow = {"files": {"scripts/a.py": entry()}, "trees": {}}
    root = tree(tmp_path, {"scripts/a.py": "P = '${local.storage.shared_root}'\n"}, allow)
    result = scanner.check(root, ("scripts",))
    assert result["clean"] is False
    assert result["stale_allowlist"] == ["scripts/a.py"]


def test_tree_entry_covers_the_subtree_and_shrinks_the_same_way(tmp_path):
    allow = {"files": {}, "trees": {"sealed": entry()}}
    live = tree(tmp_path / "live", {"sealed/deep/a.py": f"P = '{MOUNT}/someone'\n"}, allow)
    result = scanner.check(live, ("sealed",))
    assert result["clean"] is True
    assert result["allowed_trees"] == {"sealed": 1}

    cleared = tree(tmp_path / "cleared", {"sealed/deep/a.py": "# clean\n"}, allow)
    result = scanner.check(cleared, ("sealed",))
    assert result["clean"] is False
    assert result["stale_allowlist"] == ["sealed"]


def test_a_tree_entry_does_not_cover_a_sibling_with_the_same_prefix(tmp_path):
    allow = {"files": {}, "trees": {"sealed": entry()}}
    root = tree(tmp_path, {
        "sealed/a.py": f"P = '{MOUNT}/someone'\n",
        "sealed-extra/b.py": f"P = '{MOUNT}/someone'\n",
    }, allow)
    assert rules(scanner.check(root, ("sealed", "sealed-extra"))) == [
        ("sealed-extra/b.py", "shared-mount"),
    ]


def test_allowlist_entries_need_reason_owner_and_date(tmp_path):
    allow = {"files": {}, "trees": {"sealed": {"reason": "r", "owner": "o"}}}
    root = tree(tmp_path, {"sealed/a.py": "x\n"}, allow)
    try:
        scanner.check(root, ("sealed",))
    except ValueError as error:
        assert "sealed needs reason, owner and date" in str(error)
    else:
        raise AssertionError("an incomplete allowlist entry must not be accepted")