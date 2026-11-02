"""Recover a missing handoff event from a verified FINALIZED bundle; no model calls."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.core.bundle.verify import verify_bundle


def release(data, manifest, out, expected_run):
    data, manifest, out = Path(data).resolve(), Path(manifest).resolve(), Path(out).resolve()
    if not manifest.is_relative_to(data / "runs") or not out.is_relative_to(data / "work/codex"):
        raise ValueError("owned manifest and control output required")
    if not out.parent.is_dir():
        raise ValueError("existing owned control directory required")
    # Same shared guard as generation. This never deletes a lease or modifies the source.
    with (data / "locks/generation.guard").open("r+b") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (data / "locks/flight.lock").exists():
            raise ValueError("generation lease present; reconcile before handoff")
        bundle = verify_bundle(manifest)
        plan = bundle.manifest["plan"]
        if bundle.attempt_state != "FINALIZED" or plan["run_id"] != expected_run:
            raise ValueError("exact FINALIZED predecessor required")
        expected = data / "runs" / expected_run / plan["attempt_id"] / (expected_run + ".run-manifest.json")
        if manifest != expected:
            raise ValueError("predecessor manifest location differs")
        pin = {"path": str(manifest), "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
        if out.exists():
            doc = json.loads(out.read_bytes())
            if (doc.get("schema") != "saea-generation-released-v1"
                    or doc.get("run_id") != expected_run or doc.get("attempt_id") != plan["attempt_id"]
                    or doc.get("artifacts") != {"finalized_manifest": pin}
                    or doc.get("basis") != "verified-finalized-bundle-after-completion"
                    or not all(doc.get(k) is True for k in ("generation_sealed", "generation_lease_released", "scoring_complete"))):
                raise ValueError("existing handoff event differs")
            return doc
        doc = {"schema": "saea-generation-released-v1", "run_id": expected_run,
               "attempt_id": plan["attempt_id"], "generation_sealed": True,
               "generation_lease_released": True, "scoring_complete": True,
               "basis": "verified-finalized-bundle-after-completion",
               "observed_at_unix": time.time(), "original_release_time_unknown": True,
               "artifacts": {"finalized_manifest": pin},
               "consumer_requires_own_lease_and_readiness": True}
        publish_new(out, (json.dumps(doc, sort_keys=True, indent=2) + "\n").encode())
        return doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "manifest", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--expected-run", required=True)
    print(json.dumps(release(**vars(parser.parse_args()))))
