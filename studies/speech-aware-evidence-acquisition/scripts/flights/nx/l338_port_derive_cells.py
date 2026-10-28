#!/usr/bin/env python3
"""Derive the two cells of the NPU-port twin of L338 (config only): the port smoke and the port lever-column flight, TRAIN Fold B.

WHAT THE CELLS ARE. Each port cell is the HISTORICAL L338 cell (``l338_family.CELLS``: the smoke ``...-smoke-4-v1`` and the flight ``...-v1``, both
already derived from the 268-B lock by ``l338_derive_cells.py`` and committed) with exactly the port transform applied:

* baseline: ``arm`` = the raw-speech serving arm, the promoted speech-packet receipt removed, the knowledge stores (re-authored intent-boundaries and
  slot-contracts, the port retrieval sources), the two module-graph pin sites (``agree-vote-seed.agreement_source``, ``centroid-boundary-key.facet_source``)
  and the port's preparation source receipts taken from ``l338_port_pins.json`` (the cell's own roster receipt is kept), ``native_contact_fit`` under the
  PORT FIT ids (the committed port flight still carried the historical FIT run and attempt ids: fixed here), the port ``what_this_is``, and for the flight
  the committed first-phase supply pin;
* experiment: the port ids, the BF16 / vLLM-Ascend model fragment, the port decision requirement (``coordination/slurp-l338-lever-column-fold-b-npu-port-v1/
  root-admission-v1.json``, sha zero until minted), the port agreement table, the cell's own port supply requirement, the deployment-neutral service
  session name and the port texts. No text states a sequence count, replica count or tensor-parallel size: the deployment is the named placeholder
  ``PORT_DEPLOYMENT_PROFILE`` (``l338_port_family``). Every cap, plan number, population, arm, program and policy is the historical cell's.

The derive REFUSES when the port flight it produces differs from the committed port flight beyond :data:`FLIGHT_MOVED_LEAVES`, so the pins file cannot drift
from what was re-pinned and sealed. CONFIG ONLY: no contact, no ledger row, no receipt, no flip. Both cells are HELD.

    python -B scripts/flights/nx/l338_port_derive_cells.py [--dry-run]
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import l338_family as fam  # noqa: E402
import l338_port_family as pf  # noqa: E402

REPO_ROOT = pf.REPO_ROOT
ZERO = "0" * 64
ROSTER_PREFIX = "docs/receipts/rosters/"
#: Leaves of the committed port flight this derive is allowed to move (prefix match on the dotted leaf path); anything else must be reproduced exactly.
FLIGHT_MOVED_LEAVES = {
    "baseline": ("native_contact_fit.run_id", "native_contact_fit.attempt_id", "what_this_is"),
    "experiment": ("purpose", "date", "service_session", "generation_budget.definition", "core_read_design.", "cells[0].readiness.reason",
                   "cells[0].readiness.requirements[0].", "cells[0].readiness.requirements[2].reason"),
}
PORT_SESSION_TEXT = f"the port's BF16 / vLLM-Ascend Thinking deployment ({pf.DEPLOYMENT_REF})"
#: Historical phrases that state the llama.cpp np4 geometry, replaced in every carried-over text (and refused if any survives).
GEOMETRY_PHRASES = (
    ("the resident np4 Thinking session (serving-slurp-thinking-module-fold-b-np4-budget3072-max4096-v1)", PORT_SESSION_TEXT),
    ("the resident np4 Thinking session", PORT_SESSION_TEXT),
)
FORBIDDEN = re.compile(r"\bnp4\b|-np 4|c98304|4 slots|four 24,?576|--max-num-seqs|\bnp\d\b|\b(4|four) (sequences|slots)\b", re.I)


def _load(kind: str, name: str, root: Path) -> dict:
    return json.loads((Path(root) / "configs" / kind / f"{name}.json").read_text(encoding="utf-8"))


def _leaves(node, prefix: str = ""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaves(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _leaves(value, f"{prefix}[{index}]")
    else:
        yield prefix, node


def moved_leaves(before: dict, after: dict) -> list[str]:
    a, b = dict(_leaves(before)), dict(_leaves(after))
    return sorted(k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>"))


def port_text(text: str) -> str:
    for old, new in GEOMETRY_PHRASES:
        text = text.replace(old, new)
    return text


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"expected exactly one {old!r} in a carried-over text")
    return text.replace(old, new)


def port_baseline(key: str, root: Path = REPO_ROOT) -> dict:
    spec = pf.CELLS[key]
    base = copy.deepcopy(_load("baseline", spec["historical_baseline"], root))
    pins = pf.PINS
    base["arm"] = pins["arm"]
    for name in pins["removed_keys"]:
        base.pop(name, None)
    base["knowledge_stores"] = copy.deepcopy(pins["knowledge_stores"])
    seen = set()
    for component in base["native_episode"]["consumption_module"]["components"]:
        params = pins["component_params"].get(component.get("component_id"))
        if params is None:
            continue
        if component.get("state") != "on":
            raise ValueError(f"{component['component_id']} is not armed in the historical {key} baseline")
        component["params"].update(copy.deepcopy(params))
        seen.add(component["component_id"])
    if seen != set(pins["component_params"]):
        raise ValueError(f"module-graph pin sites not found in the historical {key} baseline: {sorted(set(pins['component_params']) - seen)}")
    rosters = [r for r in base["preparation_source_receipts"] if r["relpath"].startswith(ROSTER_PREFIX)]
    if len(rosters) != 1:
        raise ValueError(f"the historical {key} baseline must carry exactly one roster receipt")
    base["preparation_source_receipts"] = sorted(copy.deepcopy(pins["preparation_source_receipts_common"]) + rosters,
                                                 key=lambda item: (item["root"], item["relpath"]))
    fit = base["native_contact_fit"]
    if (fit["run_id"], fit["attempt_id"]) != (spec["historical_fit_run"], fam.CELLS[key]["fit_attempt"]) or fit["contact_cap"] != pf.fit_calls(key):
        raise ValueError(f"the historical {key} native_contact_fit is not the historical family's")
    base["native_contact_fit"] = dict(fit, run_id=spec["fit_run"], attempt_id=spec["fit_attempt"])
    base["what_this_is"] = (
        f"The NPU-port twin of {spec['historical_cell']} (design doc {pf.DESIGN_DOC}): the historical baseline {spec['historical_baseline']} with the "
        f"BF16 / vLLM-Ascend serving path (arm {pins['arm']}: the original recordings under {pf.PROMPT_ORDER}, no promoted speech packets), the "
        "re-authored semantic stores (intent-boundaries-npu-port-v1, slot-contracts-npu-port-v2) and every retrieval input re-pinned to the port's own "
        "sealed artifacts (scripts/flights/nx/l338_port_pins.json); native_contact_fit under the port FIT ids. The serving deployment is "
        f"{pf.DEPLOYMENT_REF}; the per-sequence window is {pf.PER_SEQUENCE_WINDOW_TOKENS:,} tokens. Everything else is the historical cell's: "
        + port_text(base["what_this_is"]))
    if key == "arm":
        base["prepared_phase_supply"] = copy.deepcopy(pins["flight_phase_supply"])
    else:
        base.pop("prepared_phase_supply", None)                # prepared under the smoke's own run id by the re-pin tool
    return base


def port_experiment(key: str, root: Path = REPO_ROOT) -> dict:
    spec = pf.CELLS[key]
    hist = fam.CELLS[key]
    doc = copy.deepcopy(_load("experiment", hist["cell"], root))
    pins = pf.PINS
    if doc["experiment_id"] != hist["run"]:
        raise ValueError(f"the historical {key} experiment_id is not its run id")
    doc["experiment_id"] = spec["run"]
    doc["date"] = pf.DATE
    doc["service_session"] = pf.SERVICE_SESSION
    cell = doc["cells"][0]
    if (cell["cell_id"], cell["run_id"], cell["attempt_id"], cell["baseline_fragment"]) != (hist["cell"], hist["run"], hist["attempt"], hist["baseline"]):
        raise ValueError(f"the historical {key} experiment is not the historical family's")
    cell.update(cell_id=spec["cell"], run_id=spec["run"], attempt_id=spec["attempt"], baseline_fragment=spec["baseline"], model_fragment=pf.MODEL_FRAGMENT)
    reqs = cell["readiness"]["requirements"]
    kinds = [(r["kind"], "AGREEMENT" in r.get("relpath", ""), "phase-supply" in r.get("relpath", "")) for r in reqs]
    if kinds != [("decision", False, False), ("preparation", True, False), ("preparation", False, True)]:
        raise ValueError(f"unexpected readiness requirement layout in the historical {key} experiment: {kinds}")
    reqs[0] = {"expected_decision_id": pf.DECISION_ID, "kind": "decision",
               "reason": f"Decision {pf.DECISION_ID} pinned by scripts/flights/repin_cell.py.", "relpath": pf.DECISION_RELPATH, "sha256": ZERO}
    reqs[1] = copy.deepcopy(pins["agreement_requirement"])
    supply_reason = (pins["flight_phase_supply_reason"] + " Re-prepared under the same run id by scripts/flights/repin_cell.py (core.flight.repin) "
                     "whenever the composed config moves; the next phase-supply-v<N> must reproduce the output byte for byte.")
    if key == "arm":
        reqs[2] = {"kind": "preparation", "reason": supply_reason, **copy.deepcopy(pins["flight_phase_supply"])}
    else:
        reqs[2] = {"kind": "preparation", "reason": supply_reason, "relpath": spec["supply_relpath"], "sha256": ZERO}
    cell["readiness"]["reason"] = (
        f"{hist['role'][0].upper()}{hist['role'][1:]}, NPU port: the twin of {hist['run']} on the BF16 / vLLM-Ascend Thinking core. Registered before contact "
        f"on {pf.DATE} by script (scripts/flights/nx/l338_port_derive_cells.py) from the historical L338 cell and the port pins ({pf.OWNER_DECISIONS}). "
        f"HELD: root countersigns the DRAFT rows, the main thread mints the admission receipt {pf.DECISION_RELPATH} (scripts/flights/nx/l338_port_mint_receipt.py), "
        "the deployment profile is filled from the measured speed sweep (PORT_DEPLOYMENT_PROFILE) and the provisional NPU card-hour bound re-derived from it, "
        "the cell's own first-phase supply is (re-)prepared under its own run id by scripts/flights/repin_cell.py (item E16; --allow-output-change must never "
        "be passed) and the cell is flipped. Nothing here authorises a flight, a readiness flip or an admission entry.")
    gpu_old = (f"The GPU-hour ceiling is a BOUND (rows x {fam.ARM_MIX_ROWS} lock-row equivalents / {fam.ROWS_PER_MINUTE_FULL} rows per minute / 60 x "
               f"{fam.GPU_MARGIN}, ceil 0.1), never a target.")
    gpu_new = (f"The accelerator bound is {pf.CARD_HOUR_FORMULA[key]}; it is PROVISIONAL, a BOUND never a target, and planned_gpu_hours "
               f"({spec['plan_gpu_hours_placeholder']}) is the historical figure kept only as the numeric placeholder the plan field requires until the sweep re-derives it.")
    definition = _replace_once(doc["generation_budget"]["definition"], hist["fit_run"], spec["fit_run"])
    doc["generation_budget"]["definition"] = _replace_once(definition, gpu_old, gpu_new)
    crd = doc["core_read_design"]
    crd["arms"] = {arm: spec["run"] for arm in crd["arms"]}
    crd["chain"] = (f"deployment per {pf.DEPLOYMENT_REF} -> smoke -> machine sign-off (coordinator veto window) -> the column flight")
    doc["purpose"] = (
        f"{spec['run']}: the NPU-port twin of {hist['run']} (package {pf.HISTORICAL_PACKAGE}), served by the BF16 / vLLM-Ascend Thinking core "
        f"({pf.MODEL_FRAGMENT}; per-sequence window {pf.PER_SEQUENCE_WINDOW_TOKENS:,} tokens; deployment {pf.DEPLOYMENT_REF}) over the original recordings "
        f"({pf.PROMPT_ORDER}) with the port's re-authored knowledge and regenerated retrieval vectors; {pf.DIFFERENCES_TEXT}. It is read like the historical "
        "cell, never as a byte replay of it. The historical purpose, carried over: " + port_text(doc["purpose"]))
    return doc


def verify_against_committed(key: str, base: dict, exp: dict, root: Path) -> dict:
    """The derived port flight must reproduce the committed port flight except the declared moved leaves."""

    spec = pf.CELLS[key]
    out = {}
    for kind, derived, name in (("baseline", base, spec["baseline"]), ("experiment", exp, spec["cell"])):
        path = Path(root) / "configs" / kind / f"{name}.json"
        if not path.is_file():
            out[kind] = {"committed": False, "moved": []}
            continue
        moved = moved_leaves(json.loads(path.read_text(encoding="utf-8")), derived)
        bad = [leaf for leaf in moved if not any(leaf == p or (p.endswith(".") and leaf.startswith(p)) for p in FLIGHT_MOVED_LEAVES[kind])]
        if key == "arm" and bad:
            raise ValueError(f"the derived port flight {kind} moves undeclared leaves: {bad[:12]}")
        out[kind] = {"committed": True, "moved": moved}
    return out


def scan_geometry(doc: dict) -> list[str]:
    """Text leaves that still state a llama.cpp slot geometry or a sequence count (refused)."""

    return [path for path, value in _leaves(doc) if isinstance(value, str) and FORBIDDEN.search(value)]


def derive(repo_root: Path = REPO_ROOT, dry_run: bool = False) -> dict:
    root = Path(repo_root)
    result = {}
    for key in pf.ALL_KEYS:
        spec = pf.CELLS[key]
        base, exp = port_baseline(key, root), port_experiment(key, root)
        leaks = [f"baseline:{p}" for p in scan_geometry(base)] + [f"experiment:{p}" for p in scan_geometry(exp)]
        if leaks:
            raise ValueError(f"port {key} texts still state a slot geometry: {leaks}")
        check = verify_against_committed(key, base, exp, root)
        if not dry_run:
            for kind, doc, name in (("baseline", base, spec["baseline"]), ("experiment", exp, spec["cell"])):
                (root / "configs" / kind / f"{name}.json").write_text(json.dumps(doc, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result[key] = {"cell": spec["cell"], "baseline": spec["baseline"], "run_id": spec["run"], "fit_run_id": spec["fit_run"], "versus_committed": check}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--dry-run", action="store_true", help="derive and verify, write nothing")
    args = parser.parse_args(argv)
    print(json.dumps(derive(Path(args.repo_root), args.dry_run), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
