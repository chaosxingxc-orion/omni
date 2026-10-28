"""Write (create-only) the external runtime config of one registered port encoder run, bound to the clean HEAD.

usage: port_write_runtime_config.py <study-root> <fold-b|keys> <data-root>
Prints "<config path> <sha256>". Refuses a dirty checkout, because the runtime refuses one too.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

study, which, data = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
head = subprocess.check_output(["git", "-C", str(study), "rev-parse", "HEAD"], text=True).strip()
if subprocess.check_output(["git", "-C", str(study), "status", "--porcelain"], text=True):
    sys.exit("refusing: the study checkout is not clean")
fold_b = json.loads((study / "docs/readiness/2026-10-28-slurp-tfrl-fold-b-speech-query-encoding-registration-npu-port-v1.json").read_bytes())
lock = fold_b["umbrella_lock"]
if which == "fold-b":
    entry = fold_b["plans"][0]
    plan, session = entry["execution_plan"], entry["session_id"]
    protocol_rel, protocol_sha = entry["protocol"]["relpath"], entry["protocol"]["sha256"]
elif which == "keys":
    registration = json.loads((study / "docs/readiness/2026-10-28-slurp-intent-boundary-key-encoding-npu-port-v1/registration-v1.json").read_bytes())
    plan, session = registration["plan"], "local-nemotron-cpu-slurp-intent-boundary-keys-npu-port-v1"
    protocol_rel, protocol_sha = registration["protocol_repo_path"], registration["protocol_sha256"]
else:
    sys.exit(f"unknown run {which!r}")
protocol_path = study / protocol_rel
if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != protocol_sha:
    sys.exit("refusing: the protocol bytes differ from the registration")
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
