#!/usr/bin/env python3
"""Run the newcomer drill: start an experiment from the base with a small local profile.

This is the executable form of ``docs/engineering/Newcomer-Onboarding.md``.  It
copies the base into a fresh directory at a different absolute path -- nobody's
checkout, nobody's home, no shared mount -- writes one deployment profile and one
experiment configuration, and then drives the real CLI through ``validate``,
``plan``, ``prepare`` and ``run``.

What it establishes, in order:

* the base starts from a copy, at any absolute path, with no dependence on where
  it was developed;
* the profile needs only the required seven fields, and the committed example
  already names them;
* ``validate``/``plan``/``prepare``/``run``/``status`` all work from that copy;
* a missing profile key names the exact ``local.<key>`` and ``--local-config``;
* a broken ``shared_root`` fails before any state is written;
* changing a referenced profile value makes a prepared run refuse reuse, even
  when the resolved experiment configuration is unchanged;
* ``status --from-snapshot`` still observes the run after the profile is gone.

Usage:
    python3 scripts/experiments/onboarding_drill.py [--root DIR] [--keep]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REQUIRED = ("storage.shared_root", "storage.hot_root", "storage.archive_root",
            "roots.model", "roots.data", "image.reference", "image.digest")
# The parts of the base a newcomer needs in order to run an experiment.  scripts/npu
# and scripts/probe are host-operator tools; the drill does not stage them.
STAGED = ("tools", "scripts/experiments", "scripts/lib", "configs")
EXPERIMENT_ID = "onboarding-drill"


class Drill:
    def __init__(self, root: Path, keep: bool):
        self.root, self.keep, self.steps = root, keep, []
        self.started = time.monotonic()

    def step(self, name, ok, detail=""):
        self.steps.append({"step": name, "ok": bool(ok), "detail": detail,
                           "at_seconds": round(time.monotonic() - self.started, 3)})
        print(f"[{'ok' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}", file=sys.stderr)
        return ok

    # -- staged base -------------------------------------------------------------------------

    def stage(self):
        base = self.root / "base"
        base.mkdir(parents=True)
        for name in STAGED:
            source = REPO / name
            target = base / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
            else:
                shutil.copy2(source, target)
        (base / "configs/local").mkdir(parents=True, exist_ok=True)
        self.base = base
        return self.step("stage a copy of the base elsewhere", True, str(base))

    def cli(self, command, *args, config=None, profile=True):
        argv = [sys.executable, "-B", str(self.base / "scripts/experiments/experiment.py"), command]
        if config is None:
            argv += ["--from-snapshot", str(self.run_dir)]
        else:
            argv += ["--config", str(config)]
        if profile and self.profile is not None:
            argv += ["--local-config", str(self.profile)]
        argv += list(args)
        result = subprocess.run(argv, capture_output=True, text=True, timeout=300,
                                cwd=str(self.root), env={"PATH": "/usr/bin:/bin", "HOME": str(self.root)})
        try:
            payload = json.loads(result.stdout) if result.stdout.strip() else {}
        except json.JSONDecodeError:
            payload = {"unparsed": result.stdout[-2000:]}
        return result.returncode, payload, result.stderr

    # -- the two files the newcomer writes ---------------------------------------------------

    def write_profile(self, storage: Path, extra=None, drop=()):
        """Fill the required fields only: seven, all of them already named by the example."""
        example = json.loads((self.base / "configs/local/deployment.example.json").read_text())
        filled = {
            "storage.shared_root": str(storage),
            "storage.hot_root": str(storage / "hot"),
            "storage.archive_root": str(storage / "archive"),
            "roots.model": str(storage / "models"),
            "roots.data": str(storage / "data"),
            "image.reference": "onboarding-drill-base:local",
            "image.digest": "sha256:" + "d" * 64,
        }
        if extra:
            filled.update(extra)
        missing = set(REQUIRED) - set(filled)
        assert not missing, missing
        # Delete every key that is not required, and prove the example already named
        # each required one -- otherwise "copy this file" would be a lie.
        profile = {"schema": example["schema"]}
        for name in filled:
            if name in drop:
                continue
            group, leaf = name.split(".", 1)
            profile.setdefault(group, {})[leaf] = filled[name]
        # Count the dotted keys the newcomer actually typed, not the top-level groups
        # they nest into: the acceptance claim is about how many fields to fill in.
        self.filled = sum(len(v) for g, v in profile.items() if isinstance(v, dict))
        path = self.base / "configs/local/deployment.json"
        path.write_text(json.dumps(profile, indent=2) + "\n")
        self.profile = path
        declared = {f"{g}.{k}" for g, v in example.items() if isinstance(v, dict) for k in v}
        return path, set(REQUIRED) - declared

    def write_experiment(self, name=EXPERIMENT_ID, storage: Path = None):
        template = json.loads((self.base / "configs/experiments/onboarding-drill.template.json").read_text())
        template["experiment"]["id"] = name
        path = self.root / "configs" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(template, indent=2) + "\n")
        (storage / "data").mkdir(parents=True, exist_ok=True)
        (storage / "data/onboarding-drill-manifest.json").write_text('{"samples": []}\n')
        return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="empty directory to drill in (default: a fresh temp dir)")
    parser.add_argument("--keep", action="store_true", help="keep the drilled tree for inspection")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else Path(tempfile.mkdtemp(prefix="omni-onboarding-"))
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"{root} is not empty; the drill needs a fresh directory")
    root.mkdir(parents=True, exist_ok=True)
    drill = Drill(root, args.keep)
    drill.profile, drill.run_dir = None, None
    storage = root / "storage"
    storage.mkdir()

    drill.stage()
    profile, unknown = drill.write_profile(storage)
    drill.step("write the deployment profile from the committed example",
               not unknown, f"{drill.filled} required fields; example names all of them: {not unknown}")
    experiment = drill.write_experiment(storage=storage)
    drill.run_dir = storage / "execution" / "experiments" / EXPERIMENT_ID

    code, payload, error = drill.cli("validate", config=experiment)
    drill.step("validate", code == 0, error.strip() or f"config_sha256 {payload.get('config_sha256', '')[:12]}")
    code, payload, error = drill.cli("plan", config=experiment)
    drill.step("plan", code == 0 and payload.get("order") == ["layout", "summarise"],
               error.strip() or f"order {payload.get('order')}")
    code, payload, error = drill.cli("prepare", config=experiment)
    frozen = sorted(p.name for p in drill.run_dir.glob("*.json")) if drill.run_dir.is_dir() else []
    drill.step("prepare", code == 0 and {"CONFIG.json", "CONFIG-LOCK.json", "LOCAL.json"} <= set(frozen),
               error.strip() or f"frozen {frozen}")

    code, _, error = drill.cli("run", "--phase", "cpu", config=experiment)
    states, deadline = {}, time.monotonic() + 120
    while code == 0 and time.monotonic() < deadline:
        code, payload, error = drill.cli("status", "--phase", "cpu", config=experiment)
        states = {t["id"]: t["state"] for t in payload.get("tasks", [])}
        if states and all(s in ("SUCCEEDED", "FAILED", "UNKNOWN", "BLOCKED") for s in states.values()):
            break
        time.sleep(0.5)
    drill.step("run and observe to a terminal state",
               code == 0 and set(states.values()) == {"SUCCEEDED"}, error.strip() or str(states))

    # A terminal task state is not the same as a released lock: the controller writes
    # EXIT and only then drops RUN.lock when its process ends.  Wait for that, so the
    # steps below exercise the refusal they are about instead of a busy run.
    released, deadline = False, time.monotonic() + 60
    while time.monotonic() < deadline:
        code, payload, error = drill.cli("status", "--phase", "cpu", config=experiment)
        alive = payload.get("controller", {}).get("process", {}).get("state") == "alive"
        if code == 0 and not alive:
            released = True
            break
        time.sleep(0.2)
    drill.step("the controller exits and releases RUN.lock", released,
               "" if released else (error.strip() or "controller still alive after 60s"))

    # -- the negative cases a newcomer will actually hit -------------------------------------

    dropped = drill.base / "configs/local/dropped.json"
    drill.write_profile(storage, drop=("roots.data",))
    shutil.copy2(drill.profile, dropped)
    code, _, error = drill.cli("validate", config=experiment)
    drill.step("a missing profile key names local.<key> and --local-config",
               code != 0 and "local.roots.data" in error and "--local-config" in error, error.strip())
    shutil.copy2(dropped, drill.profile)

    broken = root / "absent-root"
    drill.write_profile(broken)
    code, _, error = drill.cli("prepare", config=experiment)
    half = (broken / "execution").exists()
    drill.step("a broken shared_root fails with no half state",
               code != 0 and "shared_root must already exist" in error and not half,
               error.strip() or f"half state written: {half}")

    # A symlinked root resolves to the same bytes, so the experiment digest cannot see
    # the edit and only the frozen local identity can.
    alias = root / "data-alias"
    alias.symlink_to(storage / "data")
    drill.write_profile(storage, extra={"roots.data": str(alias)})
    code, _, error = drill.cli("prepare", config=experiment)
    drill.step("a changed referenced value refuses reuse",
               code != 0 and "local deployment profile changed" in error, error.strip())

    drill.write_profile(storage)
    drill.profile.unlink()
    drill.profile = None
    code, payload, error = drill.cli("status")
    drill.step("status --from-snapshot works after the profile is gone",
               code == 0 and payload.get("prepared") is True, error.strip())

    failed = [s["step"] for s in drill.steps if not s["ok"]]
    report = {"root": str(root), "kept": bool(args.keep), "steps": drill.steps,
              "profile_fields": drill.filled, "ok": not failed, "failed": failed,
              "elapsed_seconds": round(time.monotonic() - drill.started, 3)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.keep and not failed:
        shutil.rmtree(root, ignore_errors=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
