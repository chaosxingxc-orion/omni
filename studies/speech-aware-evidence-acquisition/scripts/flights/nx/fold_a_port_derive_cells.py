#!/usr/bin/env python3
"""Derive the Fold A confirmation cell of the NPU-port L338 flight (config only): baseline, experiment, and the never-flown Fold A comparator pair.

WHAT THE CELL IS. The port Fold B flight (``l338_port_derive_cells.port_baseline / port_experiment('arm')``, derived in memory, so the budget switch and the c17 arm
table follow automatically) with ONE declared transform: every Fold B token of an identity becomes its Fold A token (``fold_a_port_family.swap``), the dataset fragment, roster
receipt, population ids and split identity are Fold A's, the audio ceiling is the Fold A roster's, and every sha256 that pins a Fold A artefact that does not exist yet
(FA1-FA4, the cell's first-phase supply) is the zero sentinel until ``fold_a_port_pins.json`` carries the real value (the roster receipt, a committed file, is pinned for real).
Prose leaves (``what_this_is``, ``purpose``, ``readiness.reason``) are NOT swapped: they carry a Fold A head followed by the Fold B port text unchanged, so a historical
twin name inside them stays the Fold B cell it names.

The derive REFUSES when the result moves a leaf outside the declared set, or when a Fold B token survives in a non-prose leaf. CONFIG ONLY: no contact, no ledger row, no
receipt, no flip. The cell is HELD. ``--dry-run`` writes nothing.

    python -B scripts/flights/nx/fold_a_port_derive_cells.py [--repo-root DIR] [--dry-run]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fold_a_port_family as fa  # noqa: E402
import l338_family as fam  # noqa: E402
import l338_port_derive_cells as pdc  # noqa: E402
import l338_port_family as pf  # noqa: E402

from speech_aware_evidence_acquisition.rails import consumption_module as cm  # noqa: E402
from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks  # noqa: E402

REPO_ROOT = fa.REPO_ROOT
ZERO = fa.ZERO
#: Leaves that carry historical-twin names; they get a Fold A head and the Fold B text unchanged.
PROSE_LEAVES = {"baseline": ("what_this_is",), "experiment": ("purpose", "cells[0].readiness.reason")}
#: Historical descriptions that name Fold B on purpose (the exclusion list "TRAIN Fold A, Fold B or DEV-1000", the filler family's pin history): carried byte for byte, never swapped.
UNCHANGED_LEAVES = {"baseline": ("what_the_train_exemplars_store_is",), "experiment": ()}
#: Leaves the transform may move relative to the port Fold B flight document (prefix match through ``pdc.leaf_allowed``).
FOLD_A_MOVED_LEAVES = {
    "baseline": ("knowledge_stores.", "native_contact_fit.", "native_episode.consumption_module.components[3].params.agreement_source.", "preparation_source_receipts[",
                 "prepared_phase_supply.", "what_this_is"),
    "experiment": ("cells[0].", "core_read_design.", "execution_plan.planned_speech_audio_seconds", "experiment_id", "family_budget.repeated_speech_audio_seconds_cap",
                   "fragment_kind", "generation_budget.definition", "population.", "populations[", "purpose", "split_id"),
}


def _leaf_paths(node, prefix: str = ""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaf_paths(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _leaf_paths(value, f"{prefix}[{index}]")
    else:
        yield prefix, node


def _walk(node, path: str, fn):
    """Rebuild ``node`` with ``fn(path, leaf)`` applied to every leaf (dict/list structure preserved, keys unchanged)."""

    if isinstance(node, dict):
        return {key: _walk(value, f"{path}.{key}" if path else key, fn) for key, value in node.items()}
    if isinstance(node, list):
        return [_walk(value, f"{path}[{index}]", fn) for index, value in enumerate(node)]
    return fn(path, node)


def _repin(node, repo_root: Path):
    """For every ``{relpath, sha256}`` pair whose relpath is a Fold B name: swap the relpath and re-pin it (real file digest, a pins-file value, else the zero sentinel)."""

    if isinstance(node, dict):
        out = {key: _repin(value, repo_root) for key, value in node.items()}
        rel = node.get("relpath")
        if isinstance(rel, str) and isinstance(node.get("sha256"), str) and fa.has_fold_b(rel):
            out["relpath"] = fa.swap(rel)
            out["sha256"] = fa.sha_for(out["relpath"], repo_root)
        return out
    if isinstance(node, list):
        return [_repin(value, repo_root) for value in node]
    return node


def _swap_leaves(doc: dict, kind: str, repo_root: Path, head: dict) -> dict:
    prose = set(PROSE_LEAVES[kind])
    keep = set(UNCHANGED_LEAVES[kind])
    doc = _repin(doc, repo_root)

    def fn(path, leaf):
        if not isinstance(leaf, str) or path in keep:
            return leaf
        if path in prose:
            return f"{head[path]} The Fold B port text, carried over unchanged: {leaf}"
        return fa.swap(leaf)

    return _walk(doc, "", fn)


def _prose_heads(kind: str, key: str) -> dict:
    spec = fa.CELLS[key]
    fold_b = pf.CELLS[key]
    if kind == "baseline":
        return {"what_this_is": (
            f"The Fold A CONFIRMATION twin of {fold_b['baseline']} (plan docs/superpowers/plans/2026-10-30-tfrl-three-way-comparison-and-improvement-plan.md, section 8 R2): the port "
            f"Fold B lever-column baseline moved onto TRAIN Fold A ({fa.SPLIT_ID}, {fam.ROWS} rows, roster receipt {fa.ROSTER_PATH}, dataset fragment {fa.DATASET_FRAGMENT}) by one "
            "declared transform (Fold B identity tokens become Fold A tokens; the Fold A speech-query vectors, intent-boundary alias, content derivative, exemplar-intent agreement "
            "table and first-phase supply are the Fold A twins of the Fold B port preparations FA1-FA4, pinned by the zero sentinel until they are run and recorded in "
            "scripts/flights/nx/fold_a_port_pins.json; the roster receipt is pinned for real); native_contact_fit under the Fold A FIT ids; the arm table, program, stores, "
            "templates and decoding are the Fold B port cell's.")}
    return {
        "purpose": (f"{spec['run']}: the Fold A confirmation pass of the NPU-port lever column (M2: a lever is accepted only if its Fold A lower CI95 is above zero), the twin of "
                    f"{fold_b['run']} on TRAIN Fold A ({fa.SPLIT_ID}); one flight of the {pf.K} arms ({', '.join(pf.ARM_ORDER)}) under {pf.PROGRAM}, no smoke (the programs and harness "
                    "bytes were smoked on Fold B); Fold A is a held-out discovery fold, rollout-exposed by prior cells, never a test claim; the confirmation reading is a "
                    "second-fold check, not a claim."),
        "cells[0].readiness.reason": (
            f"{spec['role'][0].upper()}{spec['role'][1:]}, NPU port: the Fold A confirmation twin of {fold_b['run']}. Registered before contact on {fa.DATE} by script "
            "(scripts/flights/nx/fold_a_port_derive_cells.py) from the port Fold B cell. HELD: the Fold A preparations FA1-FA4 and the cell's first-phase supply are run and "
            f"pinned first (zero sentinel until then), root countersigns the DRAFT rows, the main thread mints the admission receipt {fa.DECISION_RELPATH}, the deployment profile is "
            "filled from the measured speed sweep (PORT_DEPLOYMENT_PROFILE) and the provisional NPU card-hour bound re-derived, and the cell is flipped by scripts/flights/repin_cell.py. "
            "Nothing here authorises a flight, a readiness flip or an admission entry.")}


def fold_a_baseline(root: Path = REPO_ROOT) -> dict:
    base = pdc.port_baseline("arm", root)
    out = _swap_leaves(base, "baseline", root, _prose_heads("baseline", "arm"))
    spec = fa.CELLS["arm"]
    # the receipts list stays sorted by (root, relpath) after the swap
    out["preparation_source_receipts"] = sorted(out["preparation_source_receipts"], key=lambda item: (item["root"], item["relpath"]))
    # the one prepared first-phase supply: this cell's own, sha pending until the flip's re-pin prepares it
    supply = out["prepared_phase_supply"]
    if supply["relpath"] != spec["supply_relpath"]:
        raise ValueError("the Fold A supply relpath is not the cell's own")
    block = out["native_episode"]["consumption_module"]
    composed = cm.compile_module(block, carrier_module_hooks(repo_root=root), repo_root=root)
    if composed.program_id != pf.PROGRAM or composed.program_ceiling != pf.CONTACTS_PER_ROW:
        raise ValueError(f"the Fold A module compiles to {composed.program_id} / {composed.program_ceiling}, not the port family's {pf.PROGRAM}")
    return out


def fold_a_experiment(root: Path = REPO_ROOT) -> dict:
    exp = pdc.port_experiment("arm", root)
    doc = _swap_leaves(exp, "experiment", root, _prose_heads("experiment", "arm"))
    spec = fa.CELLS["arm"]
    doc["population"].update(roster_receipt=fa.ROSTER_PATH, split_identity_hash=fa.SPLIT_IDENTITY_HASH, dataset_fragment=fa.DATASET_FRAGMENT)
    doc["populations"] = [{"dataset_fragment": fa.DATASET_FRAGMENT, "ids": [item["sample_id"] for item in fa.ROSTER["items"]]}]
    doc["split_id"] = fa.SPLIT_ID
    audio = fa.audio_seconds()
    doc["execution_plan"]["planned_speech_audio_seconds"] = audio
    doc["family_budget"]["repeated_speech_audio_seconds_cap"] = audio
    # the readiness requirements: decision (sha zero, swapped ids), agreement + supply (sha pending)
    reqs = doc["cells"][0]["readiness"]["requirements"]
    if [r["kind"] for r in reqs] != ["decision", "preparation", "preparation"]:
        raise ValueError("unexpected readiness requirement layout")
    reqs[0] = {"expected_decision_id": fa.DECISION_ID, "kind": "decision", "reason": f"Decision {fa.DECISION_ID} pinned by scripts/flights/repin_cell.py.",
               "relpath": fa.DECISION_RELPATH, "sha256": ZERO}
    reqs[1]["sha256"] = fa.sha_for(reqs[1]["relpath"], root)
    reqs[2]["sha256"] = fa.sha_for(reqs[2]["relpath"], root)
    if reqs[2]["relpath"] != spec["supply_relpath"]:
        raise ValueError("the Fold A supply requirement is not the cell's own")
    return doc


def _comparator_audio(factor: float) -> int:
    return math.ceil(fa.ROSTER_AUDIO_SECONDS * factor - 1e-9)


def fold_a_comparator(root: Path = REPO_ROOT) -> tuple[dict, dict]:
    """The never-flown, agreement-free Fold A comparator pair (the source identity of the Fold A agreement table, FA4): the committed Fold B port comparator moved by the
    same transform. Its caps keep the comparator's own c15 arithmetic (30 audio factor) on the Fold A roster."""

    load = lambda kind, name: json.loads((Path(root) / "configs" / kind / f"{name}.json").read_text(encoding="utf-8"))  # noqa: E731
    name = "slurp-native-stage3-l338-lever-column-fold-b-comparator-npu-port-v1"
    base_b, exp_b = load("baseline", name), load("experiment", name)
    head_b = {"what_this_is": (f"The Fold A twin of the never-flown agreement-free COMPARATOR {name} (source identity of the Fold A exemplar-intent agreement table, FA4): the "
                               "Fold B comparator moved onto TRAIN Fold A by the Fold A confirmation transform; every Fold A artefact pin is the zero sentinel until it is run.")}
    head_e = {"purpose": (f"The Fold A twin of the never-flown agreement-free COMPARATOR {name}: a model-free builder input for the Fold A agreement table; NEVER FLOWN."),
              "cells[0].readiness.reason": "NEVER FLOWN. Allowed stage check only; readiness held forever; no exposure row, no admission receipt, no decision receipt."}
    baseline = _swap_leaves(base_b, "baseline", root, head_b)
    baseline["preparation_source_receipts"] = sorted(baseline["preparation_source_receipts"], key=lambda item: (item["root"], item["relpath"]))
    # the comparator's own readiness reason is replaced wholesale (its Fold B text is not a historical twin reference)
    exp = copy.deepcopy(exp_b)
    exp = _swap_leaves(exp, "experiment", root, head_e)
    exp["cells"][0]["readiness"]["reason"] = exp_b["cells"][0]["readiness"]["reason"]
    factor = round(exp_b["family_budget"]["repeated_speech_audio_seconds_cap"] / sum(float(i["audio_seconds"]) for i in json.loads(
        (Path(root) / fam.ROSTER["arm"]).read_text(encoding="utf-8"))["items"])) if (Path(root) / fam.ROSTER["arm"]).is_file() else 30
    audio = _comparator_audio(factor)
    exp["population"].update(roster_receipt=fa.ROSTER_PATH, split_identity_hash=fa.SPLIT_IDENTITY_HASH, dataset_fragment=fa.DATASET_FRAGMENT)
    exp["populations"] = [{"dataset_fragment": fa.DATASET_FRAGMENT, "ids": [item["sample_id"] for item in fa.ROSTER["items"]]}]
    exp["split_id"] = fa.SPLIT_ID
    exp["execution_plan"]["planned_speech_audio_seconds"] = audio
    exp["family_budget"]["repeated_speech_audio_seconds_cap"] = audio
    return baseline, exp


def scan_fold_b(doc: dict, kind: str) -> list[str]:
    """Non-prose leaves that still carry a Fold B token (refused)."""

    prose = set(PROSE_LEAVES[kind]) | set(UNCHANGED_LEAVES[kind])
    return [path for path, leaf in _leaf_paths(doc) if isinstance(leaf, str) and path not in prose and fa.has_fold_b(leaf)]


def verify_moves(kind: str, before: dict, after: dict) -> list[str]:
    moved = pdc.moved_leaves(before, after)
    return [leaf for leaf in moved if not pdc.leaf_allowed(leaf, FOLD_A_MOVED_LEAVES[kind])]


def _dump(doc: dict) -> str:
    return json.dumps(doc, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def derive(repo_root: Path = REPO_ROOT, dry_run: bool = False) -> dict:
    root = Path(repo_root)
    spec = fa.CELLS["arm"]
    fold_b_base, fold_b_exp = pdc.port_baseline("arm", root), pdc.port_experiment("arm", root)
    base, exp = fold_a_baseline(root), fold_a_experiment(root)
    leaks = [f"baseline:{p}" for p in scan_fold_b(base, "baseline")] + [f"experiment:{p}" for p in scan_fold_b(exp, "experiment")]
    if leaks:
        raise ValueError(f"the Fold A cell still carries Fold B tokens outside the prose leaves: {leaks}")
    bad = [f"baseline:{p}" for p in verify_moves("baseline", fold_b_base, base)] + [f"experiment:{p}" for p in verify_moves("experiment", fold_b_exp, exp)]
    if bad:
        raise ValueError(f"the Fold A transform moved undeclared leaves: {bad[:12]}")
    stale = [p for kind, doc in (("baseline", base), ("experiment", exp)) for p, v in _leaf_paths(doc) if isinstance(v, str) and pdc.STALE_ARM_FACTS.search(v)]
    geometry = [p for doc in (base, exp) for p in pdc.scan_geometry(doc)]
    if stale or geometry:
        raise ValueError(f"stale arm facts {stale} or slot geometry {geometry} in the Fold A cell")
    cmp_base, cmp_exp = fold_a_comparator(root)
    cmp_leaks = [p for p in scan_fold_b(cmp_base, "baseline") + scan_fold_b(cmp_exp, "experiment")]
    if cmp_leaks:
        raise ValueError(f"the Fold A comparator still carries Fold B tokens: {cmp_leaks}")
    outputs = (("baseline", spec["baseline"], base), ("experiment", spec["cell"], exp), ("baseline", fa.COMPARATOR_CELL, cmp_base), ("experiment", fa.COMPARATOR_CELL, cmp_exp))
    if not dry_run:
        for kind, name, doc in outputs:
            (root / "configs" / kind / f"{name}.json").write_text(_dump(doc), encoding="utf-8")
    return {"cell": spec["cell"], "baseline": spec["baseline"], "run_id": spec["run"], "fit_run_id": spec["fit_run"], "comparator": fa.COMPARATOR_CELL,
            "files": [f"configs/{kind}/{name}.json" for kind, name, _ in outputs], "pending_pins": sorted(
                {leaf["relpath"] for doc in (base, exp) for leaf in _iter_pins(doc) if leaf.get("sha256") == ZERO})}


def _iter_pins(node):
    if isinstance(node, dict):
        if isinstance(node.get("relpath"), str) and isinstance(node.get("sha256"), str):
            yield node
        for value in node.values():
            yield from _iter_pins(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_pins(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--dry-run", action="store_true", help="derive and verify, write nothing")
    parser.add_argument("--model-fragment", choices=pf.MODEL_FRAGMENT_CHOICES, default="default")
    args = parser.parse_args(argv)
    pf.select_model_fragment(args.model_fragment)
    print(json.dumps(derive(Path(args.repo_root), args.dry_run), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
