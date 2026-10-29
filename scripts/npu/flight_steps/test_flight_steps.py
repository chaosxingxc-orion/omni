"""Offline tests for the operator flight steps (CPU only: fake operations, no docker, no card, no network, no model).

Run on npu2:  cd /mnt/cephfs/xingchao/omni/work/scripts/flight_steps && taskset -c 176-191 python3 -m pytest -q test_flight_steps.py
"""
import json
import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cellinfo  # noqa: E402
import preflight as pf  # noqa: E402

STUDY = Path(cellinfo.DEFAULT_STUDY)
SAMPLE_SMI = Path("/mnt/cephfs/xingchao/omni/work/npu-smi-sample.txt")
PROFILE_ID = pf.DEFAULT_PROFILE
PROFILE = json.loads((STUDY / "configs/deployment" / f"{PROFILE_ID}.json").read_text())
STEPS = ["lease", "smoke_worker", "fit", "smoke_read", "signoff", "campaign", "seal", "score", "read", "ledger_close"]
RECIPE_REL = "docs/readiness/2026-10-28-slurp-l338-lever-column-fold-b-npu-port-v1/analysis-recipe-v1.json"  # the port recipe


# ----------------------------------------------------------------------------------------------- preflight (python)

SMI_OK = "\n".join(
    f"| {i}     910B4-1             | OK            | 90.0                 45                      0    / 0                |\n"
    f"| 0                         | 0000:C1:00.0  | 0                    0    / 0                {hbm}/ 65536            |"
    for i, hbm in enumerate([61000] * 8)
)


def smi_text(aicore=0, hbm=61000, card=0):
    rows = []
    for i in range(8):
        a, h = (aicore, hbm) if i == card else (0, 61000)
        rows.append(f"| {i}     910B4-1             | OK            | 90.0                 45                      0    / 0                |")
        rows.append(f"| 0                         | 0000:C1:00.0  | {a}                    0    / 0                {h}/ 65536            |")
    return "\n".join(rows)


class FakeOps(pf.Ops):
    def __init__(self, tmp: Path, **over):
        self.tmp = tmp
        self.state = {
            "hostname": "A800I-03", "dirty": "", "head": "a" * 40, "pin": None, "free_gib": 5000,
            "registration": (0, "flight registration check: PASS"), "cli": (0, json.dumps({"preflight": {}, "scientific_readiness": {"state": "ready", "reason": ""}})),
            "smi": (0, smi_text()), "health": 200, "metrics": "vllm:num_requests_running{engine=\"0\"} 0.0\n",
            "image_id": "sha256:abc", "container_running": True, "started": "2026-10-30T01:00:00.000000000Z",
            "receipt": {"schema": "saea-runtime-receipt-vllm-v1", "container": {"image_id": "sha256:abc"},
                        "deployment": {"instances": [{"port": i["port"]} for i in PROFILE["instances"]]}},
            "receipt_mtime": time.mktime(time.strptime("2026-10-30T02:00:00", "%Y-%m-%dT%H:%M:%S")) - time.timezone,
            "receipt_exists": True, "session": True,
        }
        self.state.update(over)
        self.calls = []

    def run(self, argv, timeout=600, env=None):
        self.calls.append(argv)
        s = self.state
        if argv[0] == "git" and "status" in argv:
            return 0, s["dirty"]
        if argv[0] == "git" and "rev-parse" in argv:
            return 0, s["head"] + "\n"
        if argv[0] == "npu-smi":
            return s["smi"]
        if argv[:3] == ["docker", "image", "inspect"]:
            return 0, s["image_id"] + "\n"
        if argv[:2] == ["docker", "inspect"]:
            return (0, f"{'true' if s['container_running'] else 'false'} {s['started']}\n")
        raise AssertionError(f"unexpected command {argv}")

    def study_python(self, args, timeout=900):
        self.calls.append(["study_python", *args])
        return self.state["registration"] if "flight_registration_check.py" in args[0] else self.state["cli"]

    def http_get(self, url, timeout=5.0):
        if url.endswith("/health"):
            return self.state["health"], ""
        return 200, self.state["metrics"]

    def disk_free_bytes(self, path):
        return int(self.state["free_gib"] * 2**30)

    def hostname(self):
        return self.state["hostname"]

    def exists(self, path):
        path = str(path)
        if path.endswith("commit.txt"):
            return self.state["pin"] is not None
        if path.endswith("session.json"):
            return self.state["session"]
        if path.endswith("session-stop.json"):
            return False
        if path.endswith("runtime-vllm-bf16-thinking-npu.json"):
            return self.state["receipt_exists"]
        return Path(path).exists()

    def read_text(self, path):
        return self.state["pin"] if str(path).endswith("commit.txt") else Path(path).read_text()

    def read_json(self, path):
        if str(path).endswith("runtime-vllm-bf16-thinking-npu.json"):
            return self.state["receipt"]
        return json.loads(Path(path).read_text())

    def mtime(self, path):
        return self.state["receipt_mtime"]


def run_pf(tmp_path, stage="smoke", **over):
    ops = FakeOps(tmp_path, **over)
    checker = pf.Preflight(ops, stage=stage, study=str(STUDY), omni=str(tmp_path), profile_id=PROFILE_ID,
                           state_root=str(tmp_path / "state"), min_free_gib=100)
    findings = checker.run()
    return {f.name: f for f in findings}, ops


def test_preflight_all_good(tmp_path):
    found, _ = run_pf(tmp_path)
    assert all(f.ok for f in found.values()), [f for f in found.values() if not f.ok]
    assert {"registration", "cli-check-read-only", "npu-smi", "endpoints", "runtime-receipt", "cephfs-free-space",
            "resident-session", "study-tree-clean", "flight-commit-pin", "host"} <= set(found)


def test_preflight_registration_blockers_are_printed(tmp_path):
    out = "flight registration check: FAIL\n    [not-ready] readiness.state is 'held'\n    [receipt-missing] x: no receipt\n"
    found, _ = run_pf(tmp_path, registration=(1, out))
    assert not found["registration"].ok
    assert found["registration"].items == ["[not-ready] readiness.state is 'held'", "[receipt-missing] x: no receipt"]


def test_preflight_cli_check_blockers(tmp_path):
    doc = json.dumps({"preflight": {}, "blockers": ["stale prepared artifact A"], "scientific_readiness": {"state": "held", "reason": "held"}})
    found, ops = run_pf(tmp_path, cli=(1, "noise\n" + doc))
    assert not found["cli-check-read-only"].ok
    assert any("stale prepared artifact A" in i for i in found["cli-check-read-only"].items)
    called = [c for c in ops.calls if c[0] == "study_python" and "check" in c]
    assert called and "--read-only" in called[0] and "--cell" in called[0]


def test_preflight_cli_check_traceback_tail(tmp_path):
    found, _ = run_pf(tmp_path, cli=(1, "Traceback (most recent call last):\n  File x\nGateClosed: boom\n"))
    assert found["cli-check-read-only"].items == ["GateClosed: boom"]


def test_parse_npu_smi_real_sample():
    cards = pf.parse_npu_smi(SAMPLE_SMI.read_text())
    assert sorted(cards) == list(range(8))
    assert cards[0]["hbm_used_mib"] == 58405 and cards[0]["hbm_total_mib"] == 65536 and cards[0]["aicore_pct"] == 0


def test_preflight_busy_card_blocks(tmp_path):
    found, _ = run_pf(tmp_path, smi=(0, smi_text(aicore=60, card=3)))
    assert not found["npu-smi"].ok and any("card 3" in i and "busy" in i for i in found["npu-smi"].items)


def test_preflight_model_not_resident_blocks(tmp_path):
    found, _ = run_pf(tmp_path, smi=(0, smi_text(hbm=300, card=5)))
    assert not found["npu-smi"].ok and any("card 5" in i for i in found["npu-smi"].items)


def test_preflight_unreadable_npu_smi_blocks(tmp_path):
    found, _ = run_pf(tmp_path, smi=(127, ""))
    assert not found["npu-smi"].ok


def test_preflight_dead_endpoint_blocks(tmp_path):
    found, _ = run_pf(tmp_path, health=0)
    assert not found["endpoints"].ok and len(found["endpoints"].items) == 4


def test_preflight_endpoint_in_use_blocks(tmp_path):
    found, _ = run_pf(tmp_path, metrics='vllm:num_requests_running{engine="0"} 3.0\n')
    assert not found["endpoints"].ok and "already running" in found["endpoints"].items[0]


def test_parse_running_requests():
    assert pf.parse_running_requests("# HELP\nvllm:num_requests_running{a=\"b\"} 2.0\nvllm:num_requests_waiting 1\n") == 2.0
    assert pf.parse_running_requests("nothing") is None


def test_preflight_receipt_missing(tmp_path):
    found, _ = run_pf(tmp_path, receipt_exists=False)
    assert not found["runtime-receipt"].ok and "does not exist" in found["runtime-receipt"].detail


def test_preflight_receipt_stale_against_container_start(tmp_path):
    found, _ = run_pf(tmp_path, started="2026-10-30T09:00:00.000000000Z")
    assert not found["runtime-receipt"].ok and any("stale" in i for i in found["runtime-receipt"].items)


def test_preflight_receipt_image_and_ports_mismatch(tmp_path):
    receipt = {"schema": "saea-runtime-receipt-vllm-v1", "container": {"image_id": "sha256:other"},
               "deployment": {"instances": [{"port": 8200}]}}
    found, _ = run_pf(tmp_path, receipt=receipt)
    items = " | ".join(found["runtime-receipt"].items)
    assert not found["runtime-receipt"].ok and "ports" in items and "image id" in items


def test_preflight_container_not_running(tmp_path):
    found, _ = run_pf(tmp_path, container_running=False)
    assert not found["runtime-receipt"].ok and any("not running" in i for i in found["runtime-receipt"].items)


def test_preflight_low_disk(tmp_path):
    found, _ = run_pf(tmp_path, free_gib=10)
    assert not found["cephfs-free-space"].ok


def test_preflight_dirty_tree_and_commit_pin(tmp_path):
    found, _ = run_pf(tmp_path, dirty=" M src/x.py\n", pin="b" * 40)
    assert not found["study-tree-clean"].ok and not found["flight-commit-pin"].ok


def test_preflight_wrong_host_skips_live_checks(tmp_path):
    found, ops = run_pf(tmp_path, hostname="localhost.localdomain")
    assert not found["host"].ok and not found["live-checks"].ok
    assert "registration" not in found and not any(c[0] == "study_python" for c in ops.calls)


def test_preflight_no_resident_session(tmp_path):
    found, _ = run_pf(tmp_path, session=False)
    assert not found["resident-session"].ok


def test_preflight_post_stage_is_offline_only(tmp_path):
    found, ops = run_pf(tmp_path, stage="post")
    assert set(found) == {"deployment-profile", "host", "study-tree-clean", "flight-commit-pin", "cephfs-free-space"}
    assert not any(c[0] in ("npu-smi", "docker", "study_python") for c in ops.calls)


def test_preflight_main_exit_codes(tmp_path, capsys):
    ops = FakeOps(tmp_path)
    assert pf.main(["--stage", "post", "--omni-root", str(tmp_path), "--state-root", str(tmp_path / "s"), "--min-free-gib", "1"], ops=ops) == 0
    bad = FakeOps(tmp_path, free_gib=0)
    assert pf.main(["--stage", "post", "--omni-root", str(tmp_path), "--state-root", str(tmp_path / "s"), "--min-free-gib", "1"], ops=bad) == 1
    assert "BLOCKER" in capsys.readouterr().out


def test_preflight_has_no_side_effect_calls(tmp_path):
    _, ops = run_pf(tmp_path)
    for call in ops.calls:
        if call[0] == "docker":
            assert call[1] in ("inspect", "image"), call
        assert call[0] in ("git", "npu-smi", "docker", "study_python")
        if call[0] == "git":
            assert call[3] in ("status", "rev-parse")


# ----------------------------------------------------------------------------------------------- cellinfo

def test_cellinfo_smoke_and_flight():
    smoke, flight = cellinfo.load("smoke"), cellinfo.load("flight")
    assert smoke["ROWS"] == "4" and flight["ROWS"] == "1080"
    assert smoke["RUN_ID"].endswith("SMOKE-4-NPU-PORT-V1") and flight["RUN_ID"].endswith("FOLD-B-NPU-PORT-V1")
    assert smoke["FIT_CONTACT_CAP"] == "240" and flight["FIT_CONTACT_CAP"] == "64800"
    assert smoke["SESSION_ID"] == flight["SESSION_ID"]
    with pytest.raises(cellinfo.CellInfoError):
        cellinfo.load("nope")


# ----------------------------------------------------------------------------------------------- shell steps (fake ops)

FAKE_RUNNER = """#!/usr/bin/env bash
# fake port_py.sh: records argv; behaves as a lease holder for lease_cli
echo "env CPUSET=$CPUSET PORT_DOCKER_FLAGS=${PORT_DOCKER_FLAGS:-} :: $*" >> "$FAKE_LOG"
case "$*" in
  *lease_cli*)
    while [ $# -gt 0 ]; do [ "$1" = --token-output ] && REL=$2; shift; done
    echo '{"token":"t"}' > "$OMNI_ROOT/$REL"
    read -r LINE
    if [ "$LINE" = release ]; then rm -f "$OMNI_ROOT/$REL"; exit 0; fi
    exit 2 ;;
  *) [ -n "${FAKE_SLEEP:-}" ] && sleep "$FAKE_SLEEP"; exit "${FAKE_RC:-0}" ;;
esac
"""
FAKE_PREFLIGHT = """#!/usr/bin/env bash
echo "fake preflight $*" >> "$FAKE_LOG"
[ "${FAKE_PF_RC:-0}" = 0 ] && { echo "preflight: PASS"; exit 0; }
echo "[BLOCKER] fake: forced failure"; exit 1
"""


@pytest.fixture()
def env(tmp_path):
    omni = tmp_path / "omni"
    (omni / "coordination").mkdir(parents=True)
    runner = tmp_path / "fake_runner.sh"
    runner.write_text(FAKE_RUNNER)
    preflight = tmp_path / "fake_preflight.sh"
    preflight.write_text(FAKE_PREFLIGHT)
    for path in (runner, preflight):
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    values = dict(os.environ)
    values.update(OMNI_ROOT=str(omni), FLIGHT_STEPS_STUDY=str(STUDY), FLIGHT_STEPS_STATE=str(tmp_path / "state"), FLIGHT_STEPS_RUNNER=str(runner),
                  FLIGHT_STEPS_PREFLIGHT=f"bash {preflight}", FAKE_LOG=str(tmp_path / "fake.log"), FLIGHT_STEPS_LEASE_WAIT="20")
    values.pop("FAKE_RC", None)
    values.pop("FAKE_PF_RC", None)
    (tmp_path / "state").mkdir()
    (tmp_path / "fake.log").write_text("")
    return {"env": values, "tmp": tmp_path, "omni": omni, "state": tmp_path / "state", "log": tmp_path / "fake.log"}


def step(env_, name, *args, **extra):
    values = dict(env_["env"])
    values.update({k: str(v) for k, v in extra.items()})
    done = subprocess.run(["bash", str(HERE / f"step_{name}.sh"), *args], env=values, capture_output=True, text=True, timeout=120)
    return done


def log_lines(env_):
    return [line for line in env_["log"].read_text().splitlines() if "::" in line]


def mark(env_, *names):
    for name in names:
        (env_["state"] / f"{name}.done").write_text("{}")


def wait_for(path: Path, seconds=30):
    end = time.time() + seconds
    while time.time() < end:
        if path.exists():
            return True
        time.sleep(0.2)
    return False


@pytest.mark.parametrize("name", STEPS)
def test_dry_run_without_confirm_executes_nothing(env, name):
    mark(env, *STEPS[: STEPS.index(name)])  # even with every earlier marker present
    args = ["--recipe", RECIPE_REL] if name == "read" else []
    done = step(env, name, *args)
    assert done.returncode == 0, done.stderr
    assert "DRY-RUN: nothing was executed" in done.stdout and "command 1/" in done.stdout
    assert log_lines(env) == []                       # the study runner was never called
    assert not (env["state"] / f"{name}.done").exists() and not (env["state"] / f"{name}.log").exists()
    assert not list((env["omni"] / "coordination").glob("*lease*"))


@pytest.mark.parametrize("name", STEPS[1:])
def test_refuses_without_previous_marker(env, name):
    args = ["--recipe", RECIPE_REL] if name == "read" else []
    done = step(env, name, "--confirm", *args)
    assert done.returncode == 3, (done.stdout, done.stderr)
    assert "previous step" in done.stderr and STEPS[STEPS.index(name) - 1] in done.stderr
    assert log_lines(env) == [] and not (env["state"] / f"{name}.done").exists()


def test_dry_run_reports_missing_marker_without_failing(env):
    done = step(env, "campaign")
    assert done.returncode == 0 and "WOULD REFUSE (exit 3)" in done.stdout


@pytest.mark.parametrize("name", STEPS)
def test_refuses_when_preflight_fails(env, name):
    mark(env, *STEPS[: STEPS.index(name)])
    (env["state"] / "signoff.marker").write_text("ok")
    args = ["--recipe", RECIPE_REL] if name == "read" else []
    done = step(env, name, "--confirm", *args, FAKE_PF_RC=1)
    assert done.returncode == 4, (done.stdout, done.stderr)
    assert "preflight" in done.stderr and "forced failure" in done.stderr
    assert not [l for l in log_lines(env) if "lease_cli" in l or "cli worker" in l or "cli campaign" in l]
    assert not (env["state"] / f"{name}.done").exists()


def test_preflight_stage_per_step(env):
    mark(env, *STEPS[:4])
    step(env, "signoff", FAKE_PF_RC=0)       # dry-run still consults preflight for the report
    (env["state"] / "smoke_worker.done").unlink()
    step(env, "smoke_worker")
    mark(env, "signoff")
    step(env, "campaign")
    stages = re.findall(r"--stage (\w+)", env["log"].read_text())
    assert stages == ["post", "smoke", "flight"]


def test_unknown_flag_and_missing_recipe(env):
    assert step(env, "lease", "--bogus").returncode == 2
    mark(env, *STEPS[: STEPS.index("read")])
    assert step(env, "read", "--confirm").returncode == 2
    done = step(env, "read", "--confirm", "--recipe", "docs/nope.json")
    assert done.returncode == 6 and "does not exist" in done.stderr


def test_commit_pin_moved_refuses(env):
    mark(env, "lease")
    (env["state"] / "commit.txt").write_text("0" * 40)
    done = step(env, "smoke_worker", "--confirm")
    assert done.returncode == 7 and "HEAD moved" in done.stderr


def test_already_done_refuses(env):
    mark(env, *STEPS[:4], "signoff")
    done = step(env, "signoff", "--confirm")
    assert done.returncode == 5 and "already done" in done.stderr


def test_command_strings(env):
    common = ("--repo-root %s --data-root %s --runtime-root %s/work/scratch/runtime-root --lock-path %s"
              % (STUDY, env["omni"], env["omni"], cellinfo.load("flight")["LOCK_PATH"]))
    lease = step(env, "lease").stdout
    assert "PORT_DOCKER_FLAGS=-i\\ --name\\ saea-flight-lease" in lease and "core.flight.lease_cli --stage worker" in lease
    assert "--expected-session-id serving-slurp-thinking-module-fold-b-npu-port-pending-deployment-profile-v1" in lease
    assert "--token-output coordination/slurp-native-stage3-l338-lever-column-fold-b-smoke-4-npu-port-v1.worker-lease.json" in lease
    worker = step(env, "smoke_worker").stdout
    assert f"core.flight.cli worker {common}" in worker and "--lease-token-file" in worker and "SPEECHRL_DATA_DIR=" in worker
    assert f"CPUSET=160-191" in worker and "--experiment slurp-native-stage3-l338-lever-column-fold-b-smoke-4-npu-port-v1" in worker
    campaign = step(env, "campaign").stdout
    assert f"core.flight.cli campaign {common}" in campaign and "--owner root-slurp-native-stage3-l338-lever-column-fold-b-npu-port-v1" in campaign
    assert "--cell" not in campaign.split("command 1/1")[1]
    score = step(env, "score").stdout
    assert "stage3_rollout_score.py --mode perception-branches --stage all --run SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-NPU-PORT-V1" in score
    assert "--roster docs/receipts/rosters/slurp-evo-fold-b-v2.json" in score and "--scorer-id slurp-slu-official-train" in score
    ledger = step(env, "ledger_close", "--today", "2026-10-31").stdout
    assert ledger.count("ledger_mark_consumed.py") == 2 and ledger.count("--run-id") == 8 and ledger.count("--apply") == 1
    signoff = step(env, "signoff").stdout
    assert "smoke_signoff.py --conditions docs/readiness/2026-10-28-slurp-l338-lever-column-fold-b-npu-port-v1/signoff-conditions-v1.json" in signoff
    table_rel = json.loads((STUDY / "docs/readiness/2026-10-28-slurp-l338-lever-column-fold-b-npu-port-v1/signoff-conditions-v1.json").read_text())["table_relpath"]
    assert table_rel == "analysis/SAEA-SLURP-L338-LEVER-COLUMN-FOLD-B-READ-NPU-PORT-V1/smoke-decode-cost.json"
    smoke_read = step(env, "smoke_read").stdout
    assert "scripts/analysis/slurp_l338_port_column_read.py --stage smoke" in smoke_read and "--stage read" not in smoke_read
    assert f"--data-root {env['omni']}" in smoke_read and f"--analysis-root {env['omni']}/analysis/SAEA-SLURP-L338-LEVER-COLUMN-FOLD-B-READ-NPU-PORT-V1" in smoke_read
    assert f"--repo-root {STUDY}" in smoke_read and "--recipe" not in smoke_read
    seal = step(env, "seal").stdout
    assert "batch_inspect.py %s SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-NPU-PORT-V1" % env["omni"] in seal and " arm 1080 15 1 " in seal
    fit = step(env, "fit").stdout
    assert "fit_evidence.py" in fit and " smoke 4 15 1 " in fit
    read = step(env, "read", "--recipe", RECIPE_REL).stdout
    assert f"read_gate.py --recipe {RECIPE_REL}" in read and "--stage read" in read
    assert "scripts/analysis/slurp_l338_port_column_read.py" in read and f"--recipe {RECIPE_REL}" in read
    assert "analysis/SAEA-SLURP-L338-LEVER-COLUMN-FOLD-B-READ-NPU-PORT-V1" in read and "FOLD-B-READ-V1" not in read


def test_full_chain_with_fake_ops_in_order(env):
    omni, state = env["omni"], env["state"]
    # lease
    done = step(env, "lease", "--confirm")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert (state / "lease.done").exists() and (state / "commit.txt").read_text().strip()
    token = omni / "coordination/slurp-native-stage3-l338-lever-column-fold-b-smoke-4-npu-port-v1.worker-lease.json"
    assert token.exists()
    assert step(env, "lease", "--confirm").returncode == 5          # a second lease is refused
    # smoke worker (detached): returns at once, then the marker and the lease release appear
    done = step(env, "smoke_worker", "--confirm", FAKE_SLEEP=1)
    assert done.returncode == 0 and "started detached" in done.stdout
    assert not (state / "smoke_worker.done").exists()               # not yet: the worker sleeps
    assert wait_for(state / "smoke_worker.done") and wait_for(state / "lease.released")
    assert not token.exists()
    # fit (foreground): FIT evidence + smoke inspect
    cell = cellinfo.load("smoke", omni=str(omni))
    fit_dir = omni / "prepared" / cell["FIT_RUN_ID"] / cell["FIT_ATTEMPT_ID"]
    fit_dir.mkdir(parents=True)
    (fit_dir / "fit-receipt.json").write_text("{}")
    run_dir = omni / "runs" / cell["RUN_ID"] / cell["ATTEMPT_ID"]
    run_dir.mkdir(parents=True)
    (run_dir / "flight-receipts.json").write_text("{}")
    done = step(env, "fit", "--confirm")
    assert done.returncode == 0, (done.stdout, done.stderr)
    # smoke_read: signoff needs its marker; the step runs the port smoke reader (the fake runner stands in) and writes its own marker
    assert step(env, "signoff", "--confirm").returncode == 3
    done = step(env, "smoke_read", "--confirm")
    assert done.returncode == 0, (done.stdout, done.stderr) and (state / "smoke_read.done").exists()
    # signoff: refuses without the table, passes with it (the fake runner stands in for smoke_signoff.py)
    assert step(env, "signoff", "--confirm").returncode == 6
    rel = json.loads((STUDY / cell["SIGNOFF_CONDITIONS"]).read_text())["table_relpath"]
    assert rel.startswith("analysis/SAEA-SLURP-L338-LEVER-COLUMN-FOLD-B-READ-NPU-PORT-V1/")
    (omni / rel).parent.mkdir(parents=True)
    (omni / rel).write_text("{}")
    assert step(env, "signoff", "--confirm").returncode == 0
    # campaign refuses without the sign-off marker, runs with it
    assert step(env, "campaign", "--confirm").returncode == 6
    (state / "signoff.marker").write_text("machine sign-off")
    done = step(env, "campaign", "--confirm")
    assert done.returncode == 0 and "started detached" in done.stdout
    assert wait_for(state / "campaign.done")
    assert step(env, "seal", "--confirm").returncode == 0
    done = step(env, "score", "--confirm")
    assert done.returncode == 0 and wait_for(state / "score.done")
    assert step(env, "read", "--confirm", "--recipe", RECIPE_REL).returncode == 0
    assert step(env, "ledger_close", "--confirm", "--today", "2026-10-31").returncode == 0
    for name in STEPS:
        assert (state / f"{name}.done").exists(), name
    commands = " ".join(log_lines(env))
    order = ["lease_cli", "cli worker", "batch_inspect.py", "port_column_read.py --stage smoke", "smoke_signoff.py", "cli campaign", "stage3_rollout_score.py", "read_gate.py", "ledger_mark_consumed.py"]
    positions = [commands.find(token) for token in order]
    assert -1 not in positions and positions == sorted(positions)


def test_failed_detached_step_writes_failed_marker_and_preserves_lease(env):
    assert step(env, "lease", "--confirm").returncode == 0
    done = step(env, "smoke_worker", "--confirm", FAKE_RC=3)
    assert done.returncode == 0
    assert wait_for(env["state"] / "smoke_worker.failed")
    assert not (env["state"] / "smoke_worker.done").exists()
    assert not (env["state"] / "lease.released").exists()           # preserved, not released
    assert "lease NOT released" in (env["state"] / "smoke_worker.log").read_text()


def test_failed_foreground_step_writes_failed_marker(env):
    mark(env, *STEPS[: STEPS.index("seal")])
    done = step(env, "seal", "--confirm", FAKE_RC=4)
    assert done.returncode == 8 and (env["state"] / "seal.failed").exists() and not (env["state"] / "seal.done").exists()


# ----------------------------------------------------------------------------------------------- static checks

def test_bash_syntax_of_every_script():
    scripts = sorted(HERE.glob("*.sh"))
    assert len(scripts) >= 12
    for script in scripts:
        done = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        assert done.returncode == 0, (script, done.stderr)


def test_no_wsl_paths_and_english_only():
    pattern = re.compile(r"/mnt/[de]/|/home/chao|\\\\wsl|chao_workspace|\.venvs/speechrl|saea-worktrees")
    for path in sorted(HERE.iterdir()):
        if path.suffix in (".sh", ".py", ".md") and path.name != Path(__file__).name:
            text = path.read_text(encoding="utf-8")
            assert not pattern.search(text), path
            assert all(ord(ch) < 128 for ch in text), path
    own = Path(__file__).read_text(encoding="utf-8")
    assert all(ord(ch) < 128 for ch in own)


def test_every_step_requires_confirm_and_names_a_marker():
    for name in STEPS:
        text = (HERE / f"step_{name}.sh").read_text()
        assert f"STEP={name}" in text and "fs_main" in text and "fs_usage_common" in text


def test_steps_never_call_docker_or_ssh_directly():
    """Study commands go through port_py.sh; the step scripts themselves hold no docker/ssh/npu-smi invocation."""
    for path in HERE.glob("step_*.sh"):
        code = "\n".join(line for line in path.read_text().splitlines() if not line.lstrip().startswith("#"))
        assert not re.search(r"\bdocker run\b|\bssh\b|\bnpu-smi\b", code), path


def test_smoke_read_is_dry_by_default_write_once_and_runs_the_port_reader_for_the_smoke_cell(env):
    mark(env, *STEPS[: STEPS.index("smoke_read")])
    dry = step(env, "smoke_read")
    assert dry.returncode == 0 and "DRY-RUN" in dry.stdout and log_lines(env) == [] and not (env["state"] / "smoke_read.done").exists()
    table = env["omni"] / "analysis/SAEA-SLURP-L338-LEVER-COLUMN-FOLD-B-READ-NPU-PORT-V1/smoke-decode-cost.json"
    table.parent.mkdir(parents=True)
    table.write_text("{}")
    assert step(env, "smoke_read", "--confirm").returncode == 6 and "write-once" in step(env, "smoke_read", "--confirm").stderr
    table.unlink()
    done = step(env, "smoke_read", "--confirm")
    assert done.returncode == 0 and (env["state"] / "smoke_read.done").exists()
    assert len([l for l in log_lines(env) if "port_column_read.py --stage smoke" in l]) == 1
    custom = step(env, "smoke_read", "--read-id", "OTHER-READ")
    assert "analysis/OTHER-READ" in custom.stdout or custom.returncode == 5          # a done step reports instead of re-planning


def test_signoff_requires_the_smoke_read_marker(env):
    mark(env, *STEPS[: STEPS.index("smoke_read")])
    done = step(env, "signoff", "--confirm")
    assert done.returncode == 3 and "smoke_read" in done.stderr


def test_signoff_conditions_never_read_the_null_gpu_hours_ceiling(env):
    """registered_gpu_hours_ceiling is null in the port smoke table (the historical 4-slot GPU-hour figures do not describe the port); confirm
    smoke_signoff.evaluate() never looks it up, so null is "not checked", not a failure. Only arm_mix reads throughput (the slot-independent
    measured/registered row-mix ratio); no condition path names registered_gpu_hours_ceiling or the slot count."""
    sys.path.insert(0, str(STUDY / "scripts" / "flights" / "batch"))
    import smoke_signoff  # noqa: PLC0415

    cell = cellinfo.load("smoke")
    conditions = json.loads((STUDY / cell["SIGNOFF_CONDITIONS"]).read_text())
    assert not any("gpu_hours" in c.get("path", "") for c in conditions.get("extras", []))
    table = {
        "schema": conditions["table_schema"], "arms": {arm: {"perception": {"contacts": 1, "mode": "think", "cap_hits": 0, "empty_content": 0,
                 "no_json_object": 0, "finish_reason_length": 0}, "solver": {"contacts": 1, "mode": "think", "cap_hits": 0, "empty_content": 0,
                 "no_json_object": 0, "finish_reason_length": 0}} for arm in conditions["expected_arms"]},
        "throughput": {"arm_mix_measured_rows": 1.0, "arm_mix_registered_rows": 1.0}, "unattributed_contacts": 0, "absent_columns": 0,
        "registered_gpu_hours_ceiling": None,
    }
    checks = smoke_signoff.evaluate(conditions, table)
    assert checks and all(c["ok"] for c in checks)
    assert not any("gpu_hours" in c["id"] for c in checks)
    table["registered_gpu_hours_ceiling"] = "not a number"           # even a malformed value changes nothing: no condition reads the key
    assert smoke_signoff.evaluate(conditions, table) == checks
