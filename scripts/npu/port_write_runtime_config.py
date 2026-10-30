"""Write (create-only) the external runtime config of one registered port encoder run, bound to the clean HEAD.

usage: port_write_runtime_config.py <study-root> <fold-b|fold-a|keys> <data-root> [--cpuset CPUSET] [--check-only]
Prints "<config path> <sha256>". Refuses a dirty checkout, because the runtime refuses one too.

--cpuset CPUSET   the cores the encoder container will be pinned to (docker --cpuset-cpus syntax, e.g. 128-191 or 0-3,8). The pinned CPU encoder inventory
                  fixes its worker slices (first_core, workers x threads_per_worker); the pool refuses to start unless every core of that range is inside the
                  process's affinity, so a cpuset that does not cover the range is refused here, before any container is started.
--check-only      resolve the run, verify the protocol bytes and report the pool-core coverage; write nothing, touch no git state, exit 0 if covered (or if no
                  cpuset is given), 3 if the cpuset does not cover the pinned pool cores.
Defaults are the historical Fold B behaviour: `<study-root> fold-b <data-root>` is unchanged.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REGISTRATIONS = {
    "fold-b": "docs/readiness/2026-10-28-slurp-tfrl-fold-b-speech-query-encoding-registration-npu-port-v1.json",
    "fold-a": "docs/readiness/2026-10-30-slurp-tfrl-fold-a-speech-query-encoding-registration-npu-port-v1.json",
}
KEYS_REGISTRATION = "docs/readiness/2026-10-28-slurp-intent-boundary-key-encoding-npu-port-v1/registration-v1.json"
CPU_INVENTORY_SCHEMA = "saea-cpu-encoder-inventory-v1"


def parse_cpuset(spec):
    """Docker --cpuset-cpus syntax ("0-3,8") as a set of core numbers."""
    cores = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"empty element in cpuset {spec!r}")
        low, _, high = part.partition("-")
        cores.update(range(int(low), int(high or low) + 1))
    return cores


def pool_cores(inventory):
    first = inventory["first_core"]
    return set(range(first, first + inventory["workers"] * inventory["threads_per_worker"]))


def _span(cores):
    return f"{min(cores)}-{max(cores)}"


def pool_report(protocol_path, data, cpuset):
    """(covered, message) for the pinned CPU encoder pool of the protocol at `protocol_path` against `cpuset` (None = no cpuset given)."""
    protocol = json.loads(protocol_path.read_bytes())
    pin = protocol.get("encoder")
    if not isinstance(pin, dict) or "path" not in pin:
        return True, "pool: the protocol pins no CPU encoder inventory (nothing to cover)"
    inventory_path = Path(pin["path"].replace("${SAEA_DATA_ROOT}", str(data)))
    inventory = json.loads(inventory_path.read_bytes())
    if inventory.get("schema") != CPU_INVENTORY_SCHEMA:
        return True, f"pool: encoder inventory schema {inventory.get('schema')!r} is not a CPU worker pool (nothing to cover)"
    need = pool_cores(inventory)
    line = (f"pool: {inventory['workers']} workers x {inventory['threads_per_worker']} threads from first_core {inventory['first_core']} "
            f"= cores {_span(need)} ({len(need)} cores)")
    if cpuset is None:
        return True, line + "; no cpuset given"
    have = parse_cpuset(cpuset)
    missing = sorted(need - have)
    if missing:
        return False, (line + f"; cpuset {cpuset} does NOT cover it (missing {len(missing)} cores, e.g. {missing[:4]}): the pool would refuse to start "
                       "('CPU encoder worker slices exceed this process's cores'); change the cpuset or pin a different encoder inventory")
    return True, line + f"; cpuset {cpuset} covers it"


def resolve(study, which):
    """(registration doc, plan, session, protocol_rel, protocol_sha, lock) of one run."""
    if which in REGISTRATIONS:
        document = json.loads((study / REGISTRATIONS[which]).read_bytes())
        entry = document["plans"][0]
        return (entry["execution_plan"], entry["session_id"], entry["protocol"]["relpath"], entry["protocol"]["sha256"],
                document["umbrella_lock"])
    if which == "keys":
        fold_b = json.loads((study / REGISTRATIONS["fold-b"]).read_bytes())
        registration = json.loads((study / KEYS_REGISTRATION).read_bytes())
        return (registration["plan"], "local-nemotron-cpu-slurp-intent-boundary-keys-npu-port-v1", registration["protocol_repo_path"],
                registration["protocol_sha256"], fold_b["umbrella_lock"])
    sys.exit(f"unknown run {which!r}")


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", type=Path)
    parser.add_argument("which")
    parser.add_argument("data", type=Path)
    parser.add_argument("--cpuset", default=None)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    study, which, data = args.study, args.which, args.data
    plan, session, protocol_rel, protocol_sha, lock = resolve(study, which)
    protocol_path = study / protocol_rel
    if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != protocol_sha:
        sys.exit("refusing: the protocol bytes differ from the registration")
    covered, message = pool_report(protocol_path, data, args.cpuset)
    if args.check_only:
        print(f"run {which} ({plan['run_id']})")
        print(message)
        return 0 if covered else 3
    if not covered:
        sys.exit("refusing: " + message)
    head = subprocess.check_output(["git", "-C", str(study), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(study), "status", "--porcelain"], text=True):
        sys.exit("refusing: the study checkout is not clean")
    config = {"plan": plan, "protocol": {"path": str(protocol_path), "sha256": protocol_sha}, "session_id": session,
              "study_head": head, "umbrella_lock": lock}
    out = data / "coordination" / f"{plan['run_id']}-runtime-config-{head[:12]}.json"
    raw = (json.dumps(config, indent=2, sort_keys=True) + "\n").encode()
    if out.exists():
        if out.read_bytes() != raw:
            sys.exit(f"refusing: {out} exists with different bytes")
    else:
        with out.open("xb") as handle:
            handle.write(raw)
    print(out, hashlib.sha256(raw).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
