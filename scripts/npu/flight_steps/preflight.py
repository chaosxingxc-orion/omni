#!/usr/bin/env python3
"""Read-only preflight for the operator flight steps (smoke-4 cell and the Fold B port flight).

usage: preflight.py --stage {smoke,flight,post} [--profile ID] [--json] [--min-free-gib N] [--trace-read-id ID]

It reads, it never starts, stops or contacts a model. The one thing it writes is the rollout-trace directory
<data root>/analysis/<READ id>/ (created if absent, probed with a temporary file that is removed again). Exit 0 = no blocker;
exit 1 = at least one blocker (each is printed); exit 2 = the preflight itself could not run.

Rollout trace (study core.rollout_trace, env SAEA_ROLLOUT_TRACE_READ_ID): `--trace-read-id ID` is the READ id the step is about to
export. The check `trace-read-id` refuses an empty id, an id that does not match the study's run-id pattern, the smoke id when the
stage is flight/post (and the flight id when the stage is smoke: the two never share a file), and an analysis directory that cannot
be created or written. Without the flag the stage's own id is checked for smoke and flight, nothing for post.

Stages
  smoke   smoke-4 cell, full checks (registration, `core.flight.cli check --read-only`, cards, endpoints, receipt, disk)
  flight  flight cell, the same full checks
  post    offline subset used by the steps that need no live server (disk, clean tree, flight-commit pin)

Every external effect goes through one `Ops` object so the unit tests inject a fake and never touch a host.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cellinfo  # noqa: E402

PORT_PY = "/mnt/cephfs/xingchao/omni/work/scripts/port_py.sh"
DEFAULT_PROFILE = "qwen3-omni-thinking-bf16-181-4xtp2-v2"
DEFAULT_STATE_ROOT = "/mnt/cephfs/xingchao/omni/work/flight_state/fold-b-npu-port"
READ_ONLY_CPUSET = "176-191"
RECEIPT_SCHEMA = "saea-runtime-receipt-vllm-v1"
AICORE_BUSY_PCT = 5.0
DEFAULT_MIN_FREE_GIB = 200.0


@dataclass
class Finding:
    name: str
    ok: bool
    detail: str = ""
    items: list[str] = field(default_factory=list)


class Ops:
    """Every effect the preflight has. Subclass or replace in tests."""

    def run(self, argv: list[str], timeout: float = 600, env: dict | None = None) -> tuple[int, str]:
        full_env = dict(os.environ)
        full_env.update(env or {})
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=full_env)
        except (OSError, subprocess.TimeoutExpired) as error:
            return 127, f"{type(error).__name__}: {error}"
        return done.returncode, (done.stdout or "") + (done.stderr or "")

    def study_python(self, args: list[str], timeout: float = 900) -> tuple[int, str]:
        """Run `python <args>` inside saea-base through port_py.sh on the read-only CPU set."""
        runner = os.environ.get("FLIGHT_STEPS_RUNNER", PORT_PY)
        return self.run(["bash", runner, *args], timeout=timeout, env={"CPUSET": READ_ONLY_CPUSET})

    def http_get(self, url: str, timeout: float = 5.0) -> tuple[int, str]:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 (loopback health URL)
                return response.status, response.read(200000).decode("utf-8", "replace")
        except urllib.error.HTTPError as error:
            return error.code, ""
        except (OSError, ValueError) as error:
            return 0, f"{type(error).__name__}: {error}"

    def disk_free_bytes(self, path: str) -> int:
        return shutil.disk_usage(path).free

    def hostname(self) -> str:
        return socket.gethostname()

    def read_json(self, path: Path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def exists(self, path: Path) -> bool:
        return Path(path).exists()

    def mtime(self, path: Path) -> float:
        return Path(path).stat().st_mtime

    def read_text(self, path: Path) -> str:
        return Path(path).read_text(encoding="utf-8")

    def ensure_writable_dir(self, path: Path) -> str | None:
        """Create `path` (parents too) and prove a file can be written and removed there; None = fine, else the error text."""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            probe = Path(path) / f".trace-write-probe-{os.getpid()}"
            probe.write_bytes(b"")
            probe.unlink()
        except OSError as error:
            return f"{type(error).__name__}: {error}"
        return None


# --------------------------------------------------------------------------------------------- parsing helpers

def parse_npu_smi(text: str) -> dict[int, dict[str, float]]:
    """Per-card AICore % and HBM used/total MiB from `npu-smi info` text (the two-line card rows)."""
    cards: dict[int, dict[str, float]] = {}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^\|\s*(\d+)\s+\S+\s*\|\s*\S+\s*\|", line)
        if not match or index + 1 >= len(lines):
            continue
        second = lines[index + 1]
        hbm = re.search(r"(\d+)\s*/\s*(\d+)\s*\|\s*$", second)
        cells = [cell.strip() for cell in second.split("|")]
        aicore = None
        if len(cells) >= 4:
            first_number = re.match(r"^(\d+(?:\.\d+)?)", cells[3])
            aicore = float(first_number.group(1)) if first_number else None
        if hbm and aicore is not None:
            cards[int(match.group(1))] = {"aicore_pct": aicore, "hbm_used_mib": float(hbm.group(1)), "hbm_total_mib": float(hbm.group(2))}
    return cards


def parse_running_requests(metrics: str) -> float | None:
    total, seen = 0.0, False
    for line in metrics.splitlines():
        if line.startswith("vllm:num_requests_running"):
            try:
                total += float(line.rsplit(" ", 1)[1])
                seen = True
            except (IndexError, ValueError):
                continue
    return total if seen else None


def parse_docker_time(text: str) -> float | None:
    text = text.strip()
    match = re.match(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?Z$", text)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


# --------------------------------------------------------------------------------------------- checks

class Preflight:
    def __init__(self, ops: Ops, *, stage: str, study: str, omni: str, profile_id: str, state_root: str,
                 min_free_gib: float, trace_read_id: str | None = None):
        self.ops, self.stage, self.study, self.omni = ops, stage, Path(study), omni
        self.profile_id, self.state_root, self.min_free_gib = profile_id, Path(state_root), min_free_gib
        self.trace_read_id = trace_read_id
        self.findings: list[Finding] = []
        self.profile: dict | None = None
        self.cell: dict | None = None

    def add(self, name: str, ok: bool, detail: str = "", items: list[str] | None = None) -> None:
        self.findings.append(Finding(name, ok, detail, items or []))

    # -- individual checks --------------------------------------------------------------------
    def check_profile(self) -> None:
        path = self.study / "configs/deployment" / f"{self.profile_id}.json"
        try:
            self.profile = self.ops.read_json(path)
            self.add("deployment-profile", True, f"{self.profile_id}: {len(self.profile['instances'])} instance(s)")
        except (OSError, ValueError, KeyError) as error:
            self.add("deployment-profile", False, f"cannot read {path}: {error}")

    def check_host(self) -> None:
        if self.profile is None:
            return
        want, have = str(self.profile.get("host")), self.ops.hostname()
        self.add("host", have == want, f"hostname {have!r}, profile host {want!r} (host 181 only)")

    def check_tree(self) -> None:
        code, out = self.ops.run(["git", "-C", str(self.study), "status", "--porcelain", "--untracked-files=no"], timeout=60)
        dirty = [line for line in out.splitlines() if line.strip()]
        self.add("study-tree-clean", code == 0 and not dirty,
                 "tracked files unchanged" if code == 0 and not dirty else f"git status rc={code}",
                 [f"modified: {line}" for line in dirty[:20]])
        code, head = self.ops.run(["git", "-C", str(self.study), "rev-parse", "HEAD"], timeout=60)
        head = head.strip()
        pin = self.state_root / "commit.txt"
        if code != 0:
            self.add("flight-commit-pin", False, "cannot read the study HEAD")
        elif self.ops.exists(pin):
            pinned = self.ops.read_text(pin).strip()
            self.add("flight-commit-pin", pinned == head, f"pinned {pinned[:12]}, HEAD {head[:12]}")
        else:
            self.add("flight-commit-pin", True, f"no pin yet (step_lease records HEAD {head[:12]})")

    def check_disk(self) -> None:
        try:
            free = self.ops.disk_free_bytes(self.omni) / 2**30
        except OSError as error:
            self.add("cephfs-free-space", False, f"cannot stat {self.omni}: {error}")
            return
        self.add("cephfs-free-space", free >= self.min_free_gib, f"{free:,.0f} GiB free, need >= {self.min_free_gib:,.0f} GiB")

    def check_trace_dir(self) -> None:
        """The rollout-trace READ id (SAEA_ROLLOUT_TRACE_READ_ID of the step) is well formed, belongs to this stage, and its directory is writable."""
        read_id = self.trace_read_id
        if read_id is None:
            if self.stage not in ("smoke", "flight"):
                return
            try:
                read_id = cellinfo.load(self.stage, str(self.study), self.omni)["TRACE_READ_ID"]
            except cellinfo.CellInfoError as error:
                self.add("trace-read-id", False, f"cannot derive the {self.stage} trace READ id: {error}")
                return
        if not read_id:
            self.add("trace-read-id", False, "the READ id for the rollout trace (SAEA_ROLLOUT_TRACE_READ_ID) is empty; the flight would run untraced")
            return
        if cellinfo.TRACE_READ_ID_PATTERN.fullmatch(read_id) is None:
            self.add("trace-read-id", False, f"READ id {read_id!r} does not match the run-id pattern {cellinfo.TRACE_READ_ID_PATTERN.pattern}")
            return
        try:
            ids = {s: cellinfo.load(s, str(self.study), self.omni)["TRACE_READ_ID"] for s in ("smoke", "flight")}
        except cellinfo.CellInfoError:
            ids = {}
        other = {"smoke": "flight", "flight": "smoke", "post": "smoke"}[self.stage]
        if ids and read_id == ids[other]:
            self.add("trace-read-id", False, f"READ id {read_id} is the {other} trace id; the {self.stage} stage must write its own file")
            return
        directory = Path(self.omni) / "analysis" / read_id
        error = self.ops.ensure_writable_dir(directory)
        self.add("trace-read-id", error is None,
                 f"{directory} is writable (rollout-trace.jsonl goes there)" if error is None else f"{directory} is not writable: {error}")

    def check_registration(self) -> None:
        info = self.cell
        args = ["scripts/checks/flight_registration_check.py", "--repo-root", str(self.study), "--data-root", self.omni,
                "--cell", f"{info['CI_EXPERIMENT']}:{info['CI_CELL']}"]
        code, out = self.ops.study_python(args)
        blockers = [line.strip() for line in out.splitlines() if re.match(r"^\s+\[[a-z0-9-]+\]", line) or "ERROR" in line]
        if code == 0:
            self.add("registration", True, "flight_registration_check: no blocking finding")
        else:
            detail = f"flight_registration_check exit {code}: {len(blockers)} blocker line(s)"
            self.add("registration", False, detail, blockers or [out.strip()[-400:]])

    def check_cli(self) -> None:
        info = self.cell
        args = ["-m", "speech_aware_evidence_acquisition.core.flight.cli", "check", "--read-only",
                "--repo-root", str(self.study), "--data-root", self.omni,
                "--runtime-root", f"{self.omni}/work/scratch/runtime-root", "--lock-path", info["CI_LOCK_PATH"],
                "--experiment", info["CI_EXPERIMENT"], "--cell", info["CI_CELL"]]
        code, out = self.ops.study_python(args)
        items: list[str] = []
        for line in reversed(out.strip().splitlines()):
            try:
                document = json.loads(line)
            except ValueError:
                continue
            if isinstance(document, dict):
                items = [str(item)[:300] for item in document.get("blockers", [])]
                readiness = document.get("scientific_readiness") or {}
                if readiness.get("state") not in (None, "ready"):
                    items.append(f"readiness {readiness.get('state')}: {str(readiness.get('reason'))[:200]}")
                break
        else:
            tail = [line for line in out.strip().splitlines() if line.strip()]
            items = [tail[-1][:400]] if code != 0 and tail else []
        self.add("cli-check-read-only", code == 0 and not items, f"core.flight.cli check --read-only exit {code}", items)

    def check_cards(self) -> None:
        if self.profile is None:
            return
        code, out = self.ops.run(["npu-smi", "info"], timeout=60)
        cards = parse_npu_smi(out) if code == 0 else {}
        if not cards:
            self.add("npu-smi", False, f"npu-smi info rc={code} produced no card table")
            return
        threshold = float(self.profile.get("busy_threshold_mib", 4096))
        wanted = sorted({card for inst in self.profile["instances"] for card in inst["cards"]})
        items = []
        for card in wanted:
            row = cards.get(card)
            if row is None:
                items.append(f"card {card}: not in npu-smi output")
                continue
            if row["hbm_used_mib"] < threshold:
                items.append(f"card {card}: HBM {row['hbm_used_mib']:.0f} MiB < {threshold:.0f} MiB (model not resident)")
            if row["aicore_pct"] > AICORE_BUSY_PCT:
                items.append(f"card {card}: AICore {row['aicore_pct']:.0f}% > {AICORE_BUSY_PCT:.0f}% (card busy)")
        self.add("npu-smi", not items, f"{len(wanted)} deployment card(s): resident and idle" if not items else "cards not ready", items)

    def check_endpoints(self) -> None:
        if self.profile is None:
            return
        host = self.profile.get("bind_host", "127.0.0.1")
        items = []
        for inst in self.profile["instances"]:
            base = f"http://{host}:{inst['port']}"
            status, _ = self.ops.http_get(base + "/health")
            if status != 200:
                items.append(f"{inst['name']} {base}/health: HTTP {status}")
                continue
            status, body = self.ops.http_get(base + "/metrics")
            running = parse_running_requests(body) if status == 200 else None
            if running:
                items.append(f"{inst['name']}: {running:g} request(s) already running (another client is using it)")
        self.add("endpoints", not items, f"{len(self.profile['instances'])} endpoint(s) healthy and idle" if not items else "endpoint problem", items)

    def check_receipt(self) -> None:
        if self.profile is None:
            return
        fragment_path = self.study / "configs/model" / f"{self.profile['model_fragment']}.json"
        try:
            relpath = self.ops.read_json(fragment_path).get("runtime_receipt", "receipts/runtime.json")
        except (OSError, ValueError) as error:
            self.add("runtime-receipt", False, f"cannot read the model fragment: {error}")
            return
        receipt_path = self.study / "docs" / relpath
        if not self.ops.exists(receipt_path):
            self.add("runtime-receipt", False, f"{receipt_path} does not exist (written from the running servers by scripts/write_vllm_runtime_receipt.py)")
            return
        items = []
        try:
            receipt = self.ops.read_json(receipt_path)
            receipt_mtime = self.ops.mtime(receipt_path)
        except (OSError, ValueError) as error:
            self.add("runtime-receipt", False, f"unreadable receipt: {error}")
            return
        if receipt.get("schema") != RECEIPT_SCHEMA:
            items.append(f"schema {receipt.get('schema')!r} != {RECEIPT_SCHEMA!r}")
        have_ports = sorted(int(i.get("port", -1)) for i in (receipt.get("deployment") or {}).get("instances", []))
        want_ports = sorted(int(i["port"]) for i in self.profile["instances"])
        if have_ports != want_ports:
            items.append(f"receipt instance ports {have_ports} != profile ports {want_ports}")
        code, image_id = self.ops.run(["docker", "image", "inspect", "--format", "{{.Id}}", str(self.profile["image"])], timeout=60)
        receipt_image = (receipt.get("container") or {}).get("image_id")
        if code != 0:
            items.append("cannot read the local image id")
        elif image_id.strip() != receipt_image:
            items.append(f"receipt image id {str(receipt_image)[:19]} != local {image_id.strip()[:19]}")
        newest_start = 0.0
        for inst in self.profile["instances"]:
            code, out = self.ops.run(["docker", "inspect", "--format", "{{.State.Running}} {{.State.StartedAt}}", inst["name"]], timeout=60)
            parts = out.strip().split(" ", 1)
            if code != 0 or parts[0] != "true":
                items.append(f"container {inst['name']}: not running")
                continue
            started = parse_docker_time(parts[1]) if len(parts) > 1 else None
            if started is None:
                items.append(f"container {inst['name']}: unreadable start time")
            else:
                newest_start = max(newest_start, started)
        if newest_start and receipt_mtime < newest_start:
            age = (newest_start - receipt_mtime) / 3600
            items.append(f"receipt is {age:.1f} h older than the newest container start (stale: servers restarted after it was written)")
        self.add("runtime-receipt", not items, f"receipt {receipt_path.name} agrees with the live containers" if not items else "receipt not fresh", items)

    def check_session(self) -> None:
        if self.cell is None:
            return
        session = self.cell["CI_SESSION_ID"]
        directory = Path(self.omni) / "serving" / session
        resident = self.ops.exists(directory / "session.json") and not self.ops.exists(directory / "session-stop.json")
        self.add("resident-session", resident,
                 f"serving session {session} is recorded resident" if resident else
                 f"no resident session record {directory}/session.json (or a session-stop.json sits beside it); lease_cli and the worker refuse without it")

    # -- driver ---------------------------------------------------------------------------------
    def run(self) -> list[Finding]:
        self.check_profile()
        self.check_host()
        self.check_tree()
        self.check_disk()
        self.check_trace_dir()
        if any(f.name == "host" and not f.ok for f in self.findings):
            self.add("live-checks", False, "skipped: this is not host 181; run the steps on npu2 only")
            return self.findings
        if self.stage in ("smoke", "flight"):
            try:
                self.cell = {f"CI_{k}": v for k, v in cellinfo.load(self.stage, str(self.study), self.omni).items()}
            except cellinfo.CellInfoError as error:
                self.add("cell-identities", False, str(error))
            else:
                self.check_registration()
                self.check_cli()
                self.check_session()
            self.check_cards()
            self.check_endpoints()
            self.check_receipt()
        return self.findings


def render(findings: list[Finding]) -> str:
    lines = []
    for finding in findings:
        lines.append(f"[{'ok' if finding.ok else 'BLOCKER'}] {finding.name}: {finding.detail}")
        for item in finding.items:
            lines.append(f"      - {item}")
    blockers = sum(1 for f in findings if not f.ok)
    lines.append(f"preflight: {'PASS' if not blockers else 'FAIL'} ({blockers} blocker check(s) of {len(findings)})")
    return "\n".join(lines)


def main(argv=None, ops: Ops | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=("smoke", "flight", "post"))
    parser.add_argument("--profile", default=os.environ.get("FLIGHT_STEPS_PROFILE", DEFAULT_PROFILE))
    parser.add_argument("--study", default=os.environ.get("FLIGHT_STEPS_STUDY", cellinfo.DEFAULT_STUDY))
    parser.add_argument("--omni-root", default=os.environ.get("OMNI_ROOT", cellinfo.DEFAULT_OMNI))
    parser.add_argument("--state-root", default=os.environ.get("FLIGHT_STEPS_STATE", DEFAULT_STATE_ROOT))
    parser.add_argument("--min-free-gib", type=float, default=float(os.environ.get("FLIGHT_STEPS_MIN_FREE_GIB", DEFAULT_MIN_FREE_GIB)))
    parser.add_argument("--trace-read-id", default=None,
                        help="the READ id the step exports as SAEA_ROLLOUT_TRACE_READ_ID (default: the stage's own id for smoke/flight)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    checker = Preflight(ops or Ops(), stage=args.stage, study=args.study, omni=args.omni_root, profile_id=args.profile,
                        state_root=args.state_root, min_free_gib=args.min_free_gib, trace_read_id=args.trace_read_id)
    findings = checker.run()
    if args.json:
        print(json.dumps([f.__dict__ for f in findings], indent=2))
    else:
        print(render(findings))
    return 0 if all(f.ok for f in findings) else 1


if __name__ == "__main__":
    raise SystemExit(main())
