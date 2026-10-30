#!/usr/bin/env python3
"""Read the identities of one port cell from the study tree (stdlib only, read-only, no side effects).

usage: cellinfo.py <smoke|flight> [--study DIR] [--omni-root DIR] [--format env|json]

The cell ids, run ids, attempt ids, FIT ids and the service session come from the committed experiment and baseline
fragments; only the two row counts (4 and 1080) and the per-row contact ceiling (15) are constants taken from the
package README (they are not stored in a fragment in a directly readable form).
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

DEFAULT_STUDY = "/mnt/cephfs/xingchao/omni/work/dev/umbrella/studies/speech-aware-evidence-acquisition"
DEFAULT_OMNI = "/mnt/cephfs/xingchao/omni"
PACKAGE = "docs/readiness/2026-10-28-slurp-l338-lever-column-fold-b-npu-port-v1"
ROSTER = "docs/receipts/rosters/slurp-evo-fold-b-v2.json"
LOCK_REL = "docs/datasets.lock.json"  # relative to the umbrella root (two levels above studies/)
SCORER_ID = "slurp-slu-official-train"
# README of the package: 4-row smoke, 1,080-row flight, program phase-native-module-v1-c15 (15 contacts per row).
STAGES = {
    "smoke": {"experiment": "slurp-native-stage3-l338-lever-column-fold-b-smoke-4-npu-port-v1", "rows": 4},
    "flight": {"experiment": "slurp-native-stage3-l338-lever-column-fold-b-npu-port-v1", "rows": 1080},
}
CEILING_PER_ROW = 15
# Rollout-trace READ ids (study core.rollout_trace, env SAEA_ROLLOUT_TRACE_READ_ID): one file analysis/<id>/rollout-trace.jsonl per flight.
# The flight (campaign, score, read) uses the port READ run id; the smoke worker uses the same id plus this suffix, so the two never share a file.
SMOKE_TRACE_SUFFIX = "-SMOKE"
TRACE_READ_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")  # the study's RUN_ID_PATTERN (contracts.identifiers)
READ_RUN_ID_FALLBACK = "SAEA-SLURP-L338-LEVER-COLUMN-FOLD-B-READ-NPU-PORT-V1"  # used only while the package carries no analysis recipe


class CellInfoError(RuntimeError):
    pass


def load(stage: str, study: str = DEFAULT_STUDY, omni: str = DEFAULT_OMNI) -> dict:
    if stage not in STAGES:
        raise CellInfoError(f"unknown stage {stage!r}; expected smoke or flight")
    study_path = Path(study)
    experiment = STAGES[stage]["experiment"]
    try:
        document = json.loads((study_path / "configs/experiment" / f"{experiment}.json").read_text(encoding="utf-8"))
        cell = document["cells"][0]
        baseline = json.loads(
            (study_path / "configs/baseline" / f"{cell['baseline_fragment']}.json").read_text(encoding="utf-8")
        )
        fit = baseline["native_contact_fit"]
    except (OSError, KeyError, IndexError, ValueError) as error:
        raise CellInfoError(f"cannot read the {stage} cell from {study}: {error}") from error
    umbrella = study_path.parents[1]
    read_id = READ_RUN_ID_FALLBACK
    try:  # the package's own recipe declares the READ run id (read_run_id or run_ids.read)
        recipe = json.loads((study_path / PACKAGE / "analysis-recipe-v1.json").read_text(encoding="utf-8"))
        read_id = recipe.get("read_run_id") or (recipe.get("run_ids") or {}).get("read") or read_id
    except (OSError, ValueError):
        pass
    return {
        "STAGE": stage,
        "EXPERIMENT": experiment,
        "CELL": cell["cell_id"],
        "RUN_ID": cell["run_id"],
        "ATTEMPT_ID": cell["attempt_id"],
        "FIT_RUN_ID": fit["run_id"],
        "FIT_ATTEMPT_ID": fit["attempt_id"],
        "FIT_CONTACT_CAP": str(fit["contact_cap"]),
        "SESSION_ID": document["service_session"],
        "ROWS": str(STAGES[stage]["rows"]),
        "CEILING_PER_ROW": str(CEILING_PER_ROW),
        "OWNER": f"root-{cell['cell_id']}",
        "PACKAGE": PACKAGE,
        "ROSTER": ROSTER,
        "LOCK_PATH": str(umbrella / LOCK_REL),
        "SCORER_ID": SCORER_ID,
        "SIGNOFF_CONDITIONS": f"{PACKAGE}/signoff-conditions-v1.json",
        "READ_RUN_ID_DEFAULT": read_id,
        "TRACE_READ_ID": read_id + (SMOKE_TRACE_SUFFIX if stage == "smoke" else ""),
        "DATA_ROOT": omni,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=sorted(STAGES))
    parser.add_argument("--study", default=DEFAULT_STUDY)
    parser.add_argument("--omni-root", default=DEFAULT_OMNI)
    parser.add_argument("--format", choices=("env", "json"), default="env")
    args = parser.parse_args(argv)
    try:
        info = load(args.stage, args.study, args.omni_root)
    except CellInfoError as error:
        print(f"cellinfo: {error}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps(info, indent=2, sort_keys=True))
    else:
        for key, value in info.items():
            print(f"CI_{key}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
