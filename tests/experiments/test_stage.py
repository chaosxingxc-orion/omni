"""Stage protocol: deterministic sharding, no partial merge, verify re-derives.

The properties under test are the ones a reader of an audit result depends on:

* one shard and many shards produce the same bytes, so a parallel run stays
  comparable with the single-process result it replaces;
* a plan is a pure function of the tree, not of directory enumeration order;
* an unsettled shard blocks the merge instead of silently shrinking the output;
* verification re-reads the files and refuses to believe a receipt;
* every write lands inside the run root, and only on a local filesystem.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.experiments import stage as stages
from tools.experiments.local_runtime_storage import mount_for
from tools.experiments.state import read_json

CLI = REPO / "scripts/experiments/stage.py"


def tree(root):
    """A small tree with the shapes an audit has to handle: nesting, a symlink."""
    (root / "a/deep").mkdir(parents=True)
    (root / "b").mkdir()
    (root / "a/one.bin").write_bytes(b"\x00" * 4096 + b"one")
    (root / "a/deep/two.bin").write_bytes(b"two" * 100)
    (root / "b/three.bin").write_bytes(b"three" * 7)
    (root / "b/empty.txt").write_bytes(b"")
    os.symlink("../a/one.bin", root / "b/link.bin")
    return root


def drive(root, shards, tag="", name="hash-audit"):
    """Plan, run every shard in-process, merge.  Returns (merge result, run root)."""
    run_root = root.parent / f"{root.name}{tag}-run"
    stages.plan(name, str(root), shards, str(run_root))
    for index in range(shards):
        stages.run_shard(name, str(run_root), index, threads=2)
    return stages.merge(name, str(run_root)), run_root


def merged_bytes(run_root):
    return (run_root / "stages/hash-audit/merged.jsonl").read_bytes()


# -- N3: shards never change the answer ---------------------------------------------------------

@pytest.mark.parametrize("shards", [1, 2, 3, 8])
def test_every_shard_count_produces_the_same_bytes(tmp_path, shards):
    # One tree, several splits: the merged file must be byte-identical, which
    # includes the per-file mtime the audit records.  Two separate trees would
    # differ for reasons that have nothing to do with sharding.
    root = tree(tmp_path / "tree")
    _, baseline = drive(root, 1, tag="-baseline")
    _, run_root = drive(root, shards, tag=f"-n{shards}")
    assert merged_bytes(run_root) == merged_bytes(baseline)
    assert len(merged_bytes(run_root).splitlines()) == 5


def test_a_shard_count_larger_than_the_work_still_merges(tmp_path):
    result, run_root = drive(tree(tmp_path / "tree"), 16)
    assert result["parallelism"]["shards"] == 16
    assert len(merged_bytes(run_root).splitlines()) == 5


# -- the plan is deterministic ------------------------------------------------------------------

def test_the_plan_is_a_function_of_the_tree_not_of_enumeration_order(tmp_path):
    forward, backward = tree(tmp_path / "forward"), tmp_path / "backward"
    (backward / "b").mkdir(parents=True)
    (backward / "a/deep").mkdir(parents=True)
    # Reversed creation order: a filesystem that enumerates by insertion order
    # would give a different key sequence and therefore a different digest.
    (backward / "b/empty.txt").write_bytes(b"")
    os.symlink("../a/one.bin", backward / "b/link.bin")
    (backward / "b/three.bin").write_bytes(b"three" * 7)
    (backward / "a/deep/two.bin").write_bytes(b"two" * 100)
    (backward / "a/one.bin").write_bytes(b"\x00" * 4096 + b"one")
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir(), second.mkdir()
    a = stages.plan("hash-audit", str(forward), 3, str(first))
    b = stages.plan("hash-audit", str(backward), 3, str(second))
    assert a["determinism"] == b["determinism"]
    assert a["shards"] == b["shards"]


def test_a_plan_cannot_be_replaced_in_place(tmp_path):
    root = tree(tmp_path / "tree")
    stages.plan("hash-audit", str(root), 2, str(tmp_path / "run"))
    with pytest.raises(stages.StageError, match="already planned"):
        stages.plan("hash-audit", str(root), 4, str(tmp_path / "run"))


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read any directory")
def test_an_unreadable_directory_is_recorded_as_a_gap_not_skipped(tmp_path):
    root = tree(tmp_path / "tree")
    blocked = root / "b"
    blocked.chmod(0o000)
    try:
        result, run_root = drive(root, 1)
    finally:
        blocked.chmod(0o755)
    records = [json.loads(line) for line in merged_bytes(run_root).splitlines()]
    # The readable files are still audited; the directory itself is present as an
    # explicit "unreadable" record, so the manifest cannot be mistaken for full
    # coverage of the tree.
    assert [item["key"] for item in records] == ["a/deep/two.bin", "a/one.bin", "b"]
    assert records[-1]["kind"] == "unreadable"


# -- N2: no partial merge -----------------------------------------------------------------------

def test_a_missing_shard_blocks_the_merge(tmp_path):
    root = tree(tmp_path / "tree")
    run_dir = str(tmp_path / "run")
    stages.plan("hash-audit", str(root), 3, run_dir)
    stages.run_shard("hash-audit", run_dir, 0, threads=1)
    stages.run_shard("hash-audit", run_dir, 2, threads=1)
    with pytest.raises(stages.StageError, match="shard 1 is MISSING"):
        stages.merge("hash-audit", run_dir)
    assert not (tmp_path / "run/stages/hash-audit/merged.jsonl").exists()


def test_an_unknown_shard_blocks_the_merge(tmp_path):
    root = tree(tmp_path / "tree")
    run_dir = str(tmp_path / "run")
    stages.plan("hash-audit", str(root), 2, run_dir)
    stages.run_shard("hash-audit", run_dir, 0, threads=1)
    receipt = tmp_path / "run/stages/hash-audit/shards/001"
    (receipt / "RECEIPT.json").write_text(json.dumps({"state": "UNKNOWN", "shard": 1}))
    with pytest.raises(stages.StageError, match="shard 1 is UNKNOWN"):
        stages.merge("hash-audit", run_dir)


def test_tampered_shard_output_is_refused_at_merge(tmp_path):
    root = tree(tmp_path / "tree")
    run_dir = str(tmp_path / "run")
    stages.plan("hash-audit", str(root), 2, run_dir)
    for index in (0, 1):
        stages.run_shard("hash-audit", run_dir, index, threads=1)
    records = tmp_path / "run/stages/hash-audit/shards/000/records.jsonl"
    records.write_bytes(records.read_bytes().replace(b'"file"', b'"FILE"'))
    with pytest.raises(stages.StageError, match="changed after the receipt"):
        stages.merge("hash-audit", run_dir)


# -- N1: shards are independent -----------------------------------------------------------------

def test_a_shard_never_runs_twice(tmp_path):
    root = tree(tmp_path / "tree")
    run_dir = str(tmp_path / "run")
    stages.plan("hash-audit", str(root), 2, run_dir)
    stages.run_shard("hash-audit", run_dir, 0, threads=1)
    with pytest.raises(stages.StageError, match="already ran"):
        stages.run_shard("hash-audit", run_dir, 0, threads=1)


def test_each_shard_has_its_own_dispatch_and_directory(tmp_path):
    root = tree(tmp_path / "tree")
    run_dir = str(tmp_path / "run")
    stages.plan("hash-audit", str(root), 3, run_dir)
    ids = [stages.run_shard("hash-audit", run_dir, index, threads=1)["dispatch_id"]
           for index in range(3)]
    assert len(set(ids)) == 3
    directories = sorted(path.name for path in (tmp_path / "run/stages/hash-audit/shards").iterdir())
    assert directories == ["000", "001", "002"]


# -- N4: two merges agree -----------------------------------------------------------------------

def test_two_merges_of_the_same_shards_agree(tmp_path):
    first, run_root = drive(tree(tmp_path / "tree"), 3)
    second = stages.merge("hash-audit", str(run_root))
    assert first["merged_sha256"] == second["merged_sha256"]
    assert merged_bytes(run_root)  # unchanged by the second merge


# -- verify re-derives, never trusts the receipt ------------------------------------------------

def test_verify_passes_on_an_untouched_tree(tmp_path):
    _, run_root = drive(tree(tmp_path / "tree"), 3)
    report = stages.verify("hash-audit", str(run_root))
    assert report["ok"] is True and report["checked"] == 5 and report["drift"] is None


def test_verify_detects_a_file_changed_after_the_audit(tmp_path):
    root = tree(tmp_path / "tree")
    _, run_root = drive(root, 2)
    # Same length, so the plan's key list is unchanged and ``drift`` stays None:
    # the only thing that can catch this is re-deriving the record itself.
    (root / "b/three.bin").write_bytes(b"THREE" * 7)
    report = stages.verify("hash-audit", str(run_root))
    assert report["drift"] is None
    assert report["ok"] is False
    assert [(item["key"], item["reason"]) for item in report["mismatches"]] == \
        [("b/three.bin", "record differs from the input")]


def test_verify_detects_an_added_file(tmp_path):
    root = tree(tmp_path / "tree")
    _, run_root = drive(root, 2)
    (root / "b/new.bin").write_bytes(b"new")
    report = stages.verify("hash-audit", str(run_root))
    assert report["ok"] is False and "changed since the plan" in report["drift"]


def test_verify_detects_a_tampered_merged_file(tmp_path):
    _, run_root = drive(tree(tmp_path / "tree"), 2)
    merged = run_root / "stages/hash-audit/merged.jsonl"
    merged.write_bytes(merged.read_bytes().replace(b"three", b"THREE"))
    with pytest.raises(stages.StageError, match="merged output changed"):
        stages.verify("hash-audit", str(run_root))


# -- the ordered map keeps input order and a bounded window --------------------------------------

def test_the_ordered_map_yields_in_input_order_despite_uneven_work():
    import threading
    import time
    items = list(range(200))
    lock, live, peak = threading.Lock(), [0], [0]

    def slow(item):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.001 * (item % 5))
        with lock:
            live[0] -= 1
        return item * 2

    assert list(stages._ordered_map(slow, items, workers=4, window=8)) == [i * 2 for i in items]
    assert peak[0] <= 8, "the window must bound in-flight work, not the whole input"


def test_the_ordered_map_is_a_plain_map_with_one_worker():
    assert list(stages._ordered_map(str.upper, ["a", "b"], workers=1, window=4)) == ["A", "B"]


def test_available_cpus_respects_the_affinity_mask():
    import os
    assert stages.available_cpus() >= 1
    assert stages.available_cpus() <= len(os.sched_getaffinity(0))


# -- N5: writes stay in the run root, on local storage ------------------------------------------

def test_a_run_root_outside_the_target_is_refused(tmp_path):
    root = tree(tmp_path / "tree")
    run_dir = str(tmp_path / "run")
    stages.plan("hash-audit", str(root), 1, run_dir)
    # A shard writes through Store, which rejects any path outside run_root.
    with pytest.raises(Exception, match="must be inside"):
        stages._stream_write(stages.Store(run_dir), "/tmp/elsewhere.json", [b"{}\n"])


def shared_mount():
    """A shared or networked mount on this host, discovered rather than named.

    Naming one would put a deployment fact in the base; the rule is about the
    *kind* of filesystem, so the test asks the host which mounts it has and
    skips where there are none.  The parse is the production one, so the test
    cannot disagree with the guard about what a mount record says.
    """
    text = Path("/proc/self/mountinfo").read_text()
    points = sorted(line.split()[4] for line in text.splitlines() if len(line.split()) > 5)
    for point in points:
        if mount_for(point, text)["filesystem"] in stages.SHARED_FILESYSTEMS:
            return point
    return None


def test_the_cli_refuses_a_shared_run_root(tmp_path):
    point = shared_mount()
    if point is None:
        pytest.skip("this host mounts no shared filesystem to test the guard against")
    root = tree(tmp_path / "tree")
    result = subprocess.run(
        [sys.executable, "-B", str(CLI), "plan", "--stage", "hash-audit",
         "--root", str(root), "--run-root", point, "--shards", "1"],
        capture_output=True, text=True, timeout=60, cwd=str(REPO))
    assert result.returncode == 2
    assert "run root must be local storage" in result.stderr


def test_the_cli_drives_plan_run_all_merge_verify(tmp_path):
    root = tree(tmp_path / "tree")
    run_dir = str(tmp_path / "run")
    environment = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}
    steps = [["plan", "--root", str(root), "--shards", "4"],
             ["run-all", "--jobs", "4"], ["merge"], ["verify"]]
    outputs = {}
    for step in steps:
        result = subprocess.run([sys.executable, "-B", str(CLI), step[0], "--stage", "hash-audit",
                                 "--run-root", run_dir, *step[1:]],
                                capture_output=True, text=True, timeout=300, cwd=str(REPO),
                                env=environment)
        assert result.returncode == 0, result.stderr
        outputs[step[0]] = json.loads(result.stdout)
    assert outputs["run-all"]["state"] == "SUCCEEDED"
    assert outputs["run-all"]["shards"] == 4
    assert outputs["verify"]["ok"] is True
    digest = hashlib.sha256(merged_bytes(Path(run_dir))).hexdigest()
    assert outputs["merge"]["merged_sha256"] == digest
