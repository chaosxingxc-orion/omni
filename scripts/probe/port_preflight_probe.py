"""DIAGNOSTIC ONLY (npu-vllm-port): list what stands between the port cell and a flight, without authorising one.

The real FrozenCoreGate is constructed and each of its post-WSL sub-checks is run and RECORDED (never skipped
silently); the read-only preflight then continues past the gate to its first own blocker. Nothing is written to the
data root (read-only check path), no server is started, no model is contacted. This is not a gate bypass for any
run: it lives outside the repo and only prints.
usage: python port_preflight_probe.py <umbrella tree> <experiment> <cell>
"""
import json
import sys
import traceback
from pathlib import Path

tree, experiment, cell_id = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
repo = tree / "studies/speech-aware-evidence-acquisition"
data_root = Path("/mnt/cephfs/xingchao/omni")
from speech_aware_evidence_acquisition.contracts.gate import FrozenCoreGate
from speech_aware_evidence_acquisition.core.flight.campaign import load_campaign_cells
from speech_aware_evidence_acquisition.core.flight.cli import build_flight_runtime

findings = []


def note(stage, error):
    findings.append({"stage": stage, "error": f"{type(error).__name__}: {error}"[:900]})


class RecordingGate:
    def __init__(self, *args, **kwargs):
        self.real = None
        try:
            self.real = FrozenCoreGate.for_study_repo(*args, **kwargs)
        except Exception as error:  # noqa: BLE001
            note("gate construction", error)

    def assert_model_touch_allowed(self, plan, arm=None, **_kwargs):
        note("gate: execution authority", RuntimeError("WSL2 check NOT evaluated here (owner decision 9.1)"))
        if self.real is None:
            return None
        lock = None
        try:
            lock, _sha = self.real._load_lock_with_hash()
        except Exception as error:  # noqa: BLE001
            note("gate: lock", error)
        for name, call in (
            ("gate: construction role", lambda: self.real._require_construction_role_for_visible_targets(plan, arm)),
            ("gate: exposure preregistration", lambda: self.real._require_exposure_preregistration(plan, lock)),
            ("gate: carrier dataset entry", lambda: self.real._resolve_carrier_dataset_entry(plan.carrier_lock_key, lock)),
            ("gate: e0", lambda: self.real._validate_e0()),
            ("gate: runtime receipt", lambda: self.real._validate_runtime(lock)),
        ):
            try:
                call()
            except Exception as error:  # noqa: BLE001
                note(name, error)
        return None


cells = [c for c in load_campaign_cells(repo, experiment) if c.document["cell_id"] == cell_id]
cell = cells[0]
runtime = build_flight_runtime(
    repo, data_root, model=cell.document["model_fragment"], dataset=cell.document["dataset_fragment"],
    baseline=cell.document["baseline_fragment"], experiment=experiment, run_id=cell.run_id,
    attempt_id=cell.attempt_id, runtime_root=Path("/mnt/cephfs/xingchao/omni/work/scratch/runtime-root"),
    lock_path=tree / "docs/datasets.lock.json", gate_factory=RecordingGate,
)
runtime._write_resume_declaration = lambda plan: None


def recorded_exposure_check(run_id):
    note("preflight: exposure row (governance, recorded)", RuntimeError(f"no check for {run_id}"))


runtime._require_exposure_row_unconsumed = recorded_exposure_check
try:
    result = runtime.preflight(enforce_prepared_artifacts=False)
    findings.append({"stage": "preflight", "result": json.loads(json.dumps(result, default=str))})
except Exception as error:  # noqa: BLE001
    note("preflight (first post-gate blocker)", error)
    frames = [f for f in traceback.extract_tb(error.__traceback__) if "speech_aware" in f.filename]
    findings.append({"where": [f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in frames[-4:]]})
print(json.dumps(findings, indent=1))
