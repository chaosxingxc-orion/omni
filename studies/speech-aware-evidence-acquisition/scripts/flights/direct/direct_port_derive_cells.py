#!/usr/bin/env python3
"""Derive the cells of the NPU-port twin of the 266-E direct-only cell (config only): arm D of the three-way comparison, TRAIN Fold B.

WHAT THE CELLS ARE. Each port cell is the COMMITTED 266-E Thinking cell (``direct_fb_family`` cells ``smoke`` and ``thinking``, derived from the sealed Fold A direct
baseline by ``direct_fb_derive_cells.py`` and flown/re-pinned since) with exactly the port transform applied (draft section 2):

* baseline: ``arm`` = ``raw-speech-agent-native-tools`` (the original recordings, no llama.cpp packets), the promoted speech-packet receipt and the packet
  identity keys removed, ``live_phase_fit`` under the PORT FIT ids with ``contact_cap`` = rows x 1 contact x 1 counting endpoint (the value
  ``core.preparation_fit.build_direct_only_raw_fit_gate`` demands) and ``live_phase_fit_audio_seconds_cap`` = the generation plan's audio ceiling, the port prose;
* experiment: the port ids, the BF16 / vLLM-Ascend model fragment, the port decision requirement (sha zero until minted), the deployment-neutral service session
  name, the caps of ``direct_port_family``, the port texts. No text states a sequence count, replica count or tensor-parallel size: the deployment is the named
  placeholder ``PORT_DEPLOYMENT_PROFILE``.

The program, templates, knowledge view, scorer, roster and dataset fragment are the 266-E cell's, unchanged. ``lock``, ``aa`` and ``ab`` are request-identical: the
three baselines differ only in the ``live_phase_fit`` identity (a FIT run is a one-time spend identity) and the prose. The derive REFUSES when a port baseline differs
from the committed 266-E baseline beyond :data:`BASELINE_MOVED_LEAVES`, or when one pass differs from the lock beyond the FIT identity and prose, or when a text states
a llama.cpp slot geometry. CONFIG ONLY: no contact, no ledger row, no receipt, no flip. All cells are HELD.

    python -B scripts/flights/direct/direct_port_derive_cells.py [--dry-run] [--repo-root DIR]
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

import direct_port_family as fam  # noqa: E402
import l338_port_family as pf  # noqa: E402

REPO_ROOT = fam.REPO_ROOT
ZERO = "0" * 64
#: Leaves of the committed 266-E Thinking baseline this derive may move (exact leaf or a prefix of a dotted path).
BASELINE_MOVED_LEAVES = (
    "arm", "speech_packet_promotion_receipt", "speech_packet_promotion_receipt_sha256",
    "speech_feature_key_representation_id", "speech_feature_matcher_policy_id",
    "live_phase_fit", "live_phase_fit_audio_seconds_cap", "what_this_is",
)
#: Leaves one request-identical pass may differ from the lock in (the FIT spend identity and prose; neither enters a request).
PASS_PROSE_LEAVES = ("live_phase_fit", "what_this_is")
#: Historical phrases that state the llama.cpp geometry or a sequence count (refused in every port text).
FORBIDDEN = re.compile(r"\bnp4\b|-np 4|c98304|4 slots|four 24,?576|--max-num-seqs|\bnp\d\b|\b(4|four) (sequences|slots)\b|--reasoning-format deepseek", re.I)


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


def leaf_allowed(leaf: str, prefixes) -> bool:
    return any(leaf == p or leaf.startswith(p + ".") or leaf.startswith(p + "[") for p in prefixes)


def scan_geometry(doc: dict) -> list[str]:
    return [path for path, value in _leaves(doc) if isinstance(value, str) and FORBIDDEN.search(value)]


def dump(doc: dict) -> str:
    return json.dumps(doc, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


# ---- baselines --------------------------------------------------------------------------------------------------------------------------


def what_this_is(key: str) -> str:
    spec = fam.CELLS[key]
    role = ("The four-row decode-cost smoke of arm D: the same request form as the lock pass on the first four Fold B rows; it seals and reports the decode cost of "
            "one Solver contact per row under the port budget before the three 1,080-row passes are queued. No score, no read."
            if key == "smoke" else
            f"Arm D of the three-way comparison, {spec['role']}: the sealed direct-only form, one complete answer per row through the shared controller's "
            "direct-only phase program, on TRAIN Fold B 1,080. No knowledge supply, no Perception node, no analytical tool, no native tool authority."
            + ("" if key == "lock" else " Request-identical to the lock pass: only the live_phase_fit identity and this prose differ, and neither enters a request."))
    return (f"The NPU-port twin of {spec['historical_cell']} (cell 266-E Thinking arm; plan {fam.PLAN} section 5, draft {fam.DRAFT}): the historical baseline "
            f"{spec['historical_baseline']} with the BF16 / vLLM-Ascend serving path (arm {fam.ARM}: the original recordings under {pf.PROMPT_ORDER}, no promoted "
            f"speech packets), live_phase_fit under the port FIT ids with the cap of core.preparation_fit.build_direct_only_raw_fit_gate (rows x 1 contact x 1 counting "
            f"endpoint). The serving deployment is {pf.DEPLOYMENT_REF}; the per-sequence window is {pf.PER_SEQUENCE_WINDOW_TOKENS:,} tokens. {role}")


def port_baseline(key: str, root: Path = REPO_ROOT) -> dict:
    spec = fam.CELLS[key]
    base = copy.deepcopy(_load("baseline", spec["historical_baseline"], root))
    if base.get("policy") != fam.PROGRAM or base.get("arm") != "speech-feature-query" or "native_episode" in base:
        raise ValueError(f"the committed {spec['historical_baseline']} is not the 266-E direct-only packet baseline")
    base["arm"] = fam.ARM
    for name in ("speech_packet_promotion_receipt", "speech_packet_promotion_receipt_sha256",
                 "speech_feature_key_representation_id", "speech_feature_matcher_policy_id"):
        base.pop(name, None)
    base["live_phase_fit"] = {"attempt_id": spec["fit_attempt"], "contact_cap": fam.fit_calls(key), "run_id": spec["fit_run"]}
    base["live_phase_fit_audio_seconds_cap"] = fam.audio_seconds(key)
    base["what_this_is"] = what_this_is(key)
    return base


# ---- experiments ------------------------------------------------------------------------------------------------------------------------

HELD = ("HELD: root countersigns the DRAFT rows, the main thread mints the admission receipt (scripts/flights/nx/l338_port_mint_receipt.py pattern; the D receipt is "
        f"{fam.DECISION_RELPATH}), the deployment profile is filled from the measured speed sweep (PORT_DEPLOYMENT_PROFILE) and the provisional NPU card-hour bound "
        "re-derived from it, and the cells are re-pinned and flipped by scripts/flights/repin_cell.py (the program declares no supplied first phase, so the re-pin "
        "creates no prepared artifact; --allow-output-change must never be passed). Nothing here authorises a flight, a readiness flip or an admission entry.")


def port_experiment(key: str, root: Path = REPO_ROOT) -> dict:
    spec = fam.CELLS[key]
    hist = copy.deepcopy(_load("experiment", spec["historical_cell"], root))
    smoke = key == "smoke"
    if hist["experiment_id"] != spec["historical_run"] or hist["cells"][0]["dataset_fragment"] != fam.DATASET:
        raise ValueError(f"the committed {spec['historical_cell']} is not the 266-E cell")
    doc = copy.deepcopy(hist)
    calls, fit_cap, audio = fam.generation_calls(key), fam.fit_calls(key), fam.audio_seconds(key)
    doc["experiment_id"] = spec["run"]
    doc["date"] = fam.DATE
    doc["fragment_kind"] = f"saea-{fam.PKG_NAME}"
    doc["service_session"] = fam.SERVICE_SESSION
    doc["preserve_wire_responses"] = False
    doc["execution_plan"] = {"execution_profile": "bounded-discovery-probe", "planned_gpu_hours": spec["gpu_hours"], "planned_model_calls": calls,
                             "planned_speech_audio_seconds": audio, "split_role": "discovery"}
    doc["family_budget"] = {"asr_calls": 0, "ceilings_are_not_targets": True, "encoder_calls": 0, "fit_endpoint_calls_cap": fit_cap,
                            "generation_calls_cap": calls, "gpu_hours_fit_inclusive": spec["gpu_hours"], "native_task_contract_calls_cap": 0,
                            "official_complete_scoring_jobs": spec["scoring_jobs"], "repeated_speech_audio_seconds_cap": audio}
    cell = doc["cells"][0]
    cell.update(attempt_id=spec["attempt"], baseline_fragment=spec["baseline"], cell_id=spec["cell"], model_fragment=fam.MODEL_FRAGMENT, run_id=spec["run"])
    cell["readiness"] = {
        "reason": (f"{spec['role'][0].upper()}{spec['role'][1:]}, NPU port: the twin of {spec['historical_run']} on the BF16 / vLLM-Ascend Thinking core. Registered "
                   f"before contact on {fam.DATE} by script (scripts/flights/direct/direct_port_derive_cells.py) from the committed 266-E cell ({fam.PLAN}, ruling 6 of "
                   f"section 9). {HELD}"),
        "requirements": [{"expected_decision_id": fam.DECISION_ID, "kind": "decision",
                         "reason": f"Decision {fam.DECISION_ID} pinned by scripts/flights/repin_cell.py.", "relpath": fam.DECISION_RELPATH, "sha256": ZERO}],
        "state": "held",
    }
    doc["core_read_design"] = {
        "arms": {k: fam.CELLS[k]["run"] for k in fam.ALL_KEYS},
        "chain": f"deployment per {pf.DEPLOYMENT_REF} -> smoke -> machine sign-off (coordinator veto window) -> lock -> aa -> ab (one resident deployment, sequential runs)",
        "reading": ("official pooled SLU-F1 and intent (scenario_action_exact) of lock, with the per-row F1 beside it; the replicate variance is read from lock / aa / ab "
                    "(noise model M1, k = 2 replicates); twin admissibility |SLU(aa) - SLU(lock)| <= 1.20 pp; the A-to-D gap is a cross-run reading at the 2.5 pp bar"),
        "stage3_measurement_discipline": "docs/superpowers/specs/2026-09-21-slurp-stage3-training-free-rl-design.md",
        "this_cell": spec["role"],
    }
    doc["generation_budget"] = {
        "cap": calls, "target": calls,
        "definition": (f"The direct-only phase program (configs/policy/{fam.PROGRAM}.json: one phase, max_contacts 1, always-run-full-program) makes exactly ONE contact per row, so "
                       f"{spec['rows']:,} x 1 = {spec['rows']:,} is the program's exact maximum; the plan ceiling adds {fam.FALLBACK_CONTACTS_PER_ROW} answer-only fallback contact per row "
                       f"(one per budgeted-thinking contact, core.reasoning_knobs): {calls:,} = {spec['rows']:,} x {fam.GENERATION_CONTACTS_PER_ROW}. The live phase fit "
                       f"(core.preparation_fit.build_direct_only_raw_fit_gate) makes {spec['rows']} x 1 x {fam.DIRECT_ONLY_ENDPOINTS_PER_CONTACT} = {fit_cap:,} tokenizer endpoint calls under its own "
                       f"run id {spec['fit_run']}. The accelerator bound is {fam.CARD_HOUR_FORMULA[key]}; it is PROVISIONAL, a BOUND never a target, and planned_gpu_hours "
                       f"({spec['gpu_hours']}) is the 266-E figure kept only as the numeric placeholder the plan field requires until the sweep re-derives it. Budgets are "
                       "run-cumulative across every attempt of a run id."),
    }
    roster = fam.roster_doc(spec["roster"])
    doc["population"] = {
        "count": spec["rows"], "dataset_fragment": fam.DATASET,
        "decision_population": (f"TRAIN Fold B ({roster['roster_id']}): {spec['rows']} rows of {fam.ROSTER[spec['roster']]}"
                                + (", the first four rows in roster order; structural decode-cost proof only, no score and no read." if smoke else
                                   "; discovery role, rollout-exposed, never a test claim (owner decisions 28 and 86).")),
        "roster_receipt": fam.ROSTER[spec["roster"]], "split_identity_hash": roster["split_identity_hash"],
    }
    doc["populations"] = [{"dataset_fragment": fam.DATASET, "ids": [item["sample_id"] for item in roster["items"]]}]
    doc["stage3_algorithm"] = {
        "id": "none", "operators_used": [],
        "note": ("The D port cell carries NO training-free RL operator and says so explicitly: it is a bare-core capability read -- the sealed direct-only form on the port "
                 "core -- with no rollout, no per-sample scoring, no credit assignment, no update operator, no evolving store and no mechanism."),
    }
    doc["purpose"] = (
        f"{spec['run']}: " + (
            "four-row decode-cost smoke of arm D on the BF16 / vLLM-Ascend Thinking core: that the raw-audio request form, the direct-only live phase fit and the seal work, and what "
            "one Solver contact costs under the port budget (completion tokens per answer, seconds per contact). Scored by no one and read by no one: four rows are never evidence."
            if smoke else
            f"{spec['role']} of the three-way comparison on TRAIN Fold B 1,080 (plan {fam.PLAN} section 5): the direct model, one Solver call per row, no knowledge, no program. "
            "Development numbers only (owner decisions 28 and 86); never a test claim. A capability read, no mechanism, no claim.")
        + f" Served by the BF16 / vLLM-Ascend Thinking core ({fam.MODEL_FRAGMENT}; per-sequence window {pf.PER_SEQUENCE_WINDOW_TOKENS:,} tokens; deployment {pf.DEPLOYMENT_REF}) "
        f"over the original recordings ({pf.PROMPT_ORDER}); {fam.DIFFERENCES_TEXT}. Read beside the 266-E Thinking arm, never as a byte replay of it.")
    return doc


def verify_against_committed(key: str, base: dict, root: Path) -> dict:
    """The port baseline must reproduce the committed 266-E baseline except the declared leaves; a pass must equal the lock except FIT identity and prose."""

    spec = fam.CELLS[key]
    hist = _load("baseline", spec["historical_baseline"], root)
    moved = moved_leaves(hist, base)
    bad = [leaf for leaf in moved if not leaf_allowed(leaf, BASELINE_MOVED_LEAVES)]
    if bad:
        raise ValueError(f"the derived {key} baseline moves undeclared leaves of the committed 266-E baseline: {bad[:12]}")
    return {"historical": spec["historical_baseline"], "moved": moved}


def derive(repo_root: Path = REPO_ROOT, dry_run: bool = False) -> dict:
    root = Path(repo_root)
    bases = {key: port_baseline(key, root) for key in fam.ALL_KEYS}
    exps = {key: port_experiment(key, root) for key in fam.ALL_KEYS}
    result: dict = {}
    for key in fam.ALL_KEYS:
        leaks = [f"baseline:{p}" for p in scan_geometry(bases[key])] + [f"experiment:{p}" for p in scan_geometry(exps[key])]
        if leaks:
            raise ValueError(f"port {key} texts still state a llama.cpp geometry or packet fact: {leaks}")
        check = verify_against_committed(key, bases[key], root)
        if key in fam.REPLICATE_KEYS:
            diff = moved_leaves(bases["lock"], bases[key])
            bad = [leaf for leaf in diff if not leaf_allowed(leaf, PASS_PROSE_LEAVES)]
            if bad:
                raise ValueError(f"pass {key} is not request-identical to the lock: {bad}")
            check["versus_lock"] = diff
        result[key] = {"cell": fam.CELLS[key]["cell"], "baseline": fam.CELLS[key]["baseline"], "run_id": fam.CELLS[key]["run"],
                       "fit_run_id": fam.CELLS[key]["fit_run"], "versus_committed_266e": check}
    campaign = {"cells": [{"cell": fam.CELLS[k]["cell"], "experiment": fam.CELLS[k]["cell"]} for k in fam.PASS_KEYS], "schema": "saea-campaign-plan-v1"}
    outputs = {f"configs/baseline/{fam.CELLS[k]['baseline']}.json": dump(bases[k]) for k in fam.ALL_KEYS}
    outputs.update({f"configs/experiment/{fam.CELLS[k]['cell']}.json": dump(exps[k]) for k in fam.ALL_KEYS})
    outputs[f"configs/campaign/{fam.CELLS['lock']['cell']}.json"] = json.dumps(campaign, indent=2, sort_keys=True) + "\n"
    names266 = ("slurp-direct-only-fold-b-thinking", "slurp-direct-only-fold-b-instruct")
    if any(Path(p).name.startswith(names266) for p in outputs):
        raise ValueError("the derive would write a 266-E file")
    result["outputs"] = sorted(outputs)
    if not dry_run:
        for relpath, text in outputs.items():
            target = root / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
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
