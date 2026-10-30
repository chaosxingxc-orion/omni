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

import l338_derive_cells as dc  # noqa: E402  (the historical derive: its text constants are the sentences the port rewrites for the replicate)
import l338_family as fam  # noqa: E402
import l338_port_family as pf  # noqa: E402

from speech_aware_evidence_acquisition.rails import consumption_module as cm  # noqa: E402
from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks  # noqa: E402

REPO_ROOT = pf.REPO_ROOT
ZERO = "0" * 64
#: The consumption-module lock the port cells declare (plan 3.14; ``rails/adapters/slurp_module_lock_v2_2_npu_port.py``): lock v2.2 with the ten port store-pin leaves. The
#: historical cells name ``v2.2``, whose layer carries the historical 268-B store pins, so ``verify_declared_module`` refuses a config that pins the re-authored stores.
PORT_MODULE_LOCK = "v2.2-npu-port"
ROSTER_PREFIX = "docs/receipts/rosters/"
#: Leaves of the committed port flight this derive is allowed to move (prefix match on the dotted leaf path); anything else must be reproduced exactly.
FLIGHT_MOVED_LEAVES = {
    "baseline": ("native_contact_fit.run_id", "native_contact_fit.attempt_id", "what_this_is", "native_episode.consumption_module.lock",
                 # noise model M1 (the registered replicate of lock): the frame names it, the compiled block carries its row, the program grows by 2 contacts
                 "native_contact_fit.contact_cap", "policy", "native_episode.consumption_module.sampling_frame.replicates",
                 "native_episode.roles.perception.sampling_variants.arms["),
    "experiment": ("purpose", "date", "service_session", "generation_budget.", "core_read_design.", "cells[0].readiness.reason",
                   "cells[0].readiness.requirements[0].", "cells[0].readiness.requirements[2].reason",
                   # noise model M1: the caps follow the port arm table (c17)
                   "execution_plan.planned_model_calls", "execution_plan.planned_speech_audio_seconds", "family_budget."),
}
#: Under the thinking-budget switch (``l338_port_family.select_model_fragment``) the derive additionally moves the supply bound, the fit-estimate completion ceiling
#: (both sites of the baseline) and the experiment's model fragment name; nothing else. Empty for ``default``.
SWITCH_MOVED_LEAVES = {
    "baseline": ("native_episode.consumption_module.sampling_frame.max_supplied_items", "native_episode.consumption_module.sampling_frame.fit_estimate.",
                 "native_episode.roles.perception.sampling_variants.max_supplied_items", "native_episode.roles.perception.sampling_variants.fit_estimate."),
    "experiment": ("cells[0].model_fragment",),
}


def moved_leaf_allowance(kind: str) -> tuple[str, ...]:
    """The prefixes of the committed port flight the derive may move: :data:`FLIGHT_MOVED_LEAVES` plus, under a non-default switch, :data:`SWITCH_MOVED_LEAVES`."""

    extra = () if pf.active_switch() == "default" else SWITCH_MOVED_LEAVES[kind]
    return tuple(FLIGHT_MOVED_LEAVES[kind]) + tuple(extra)


PORT_SESSION_TEXT = f"the port's BF16 / vLLM-Ascend Thinking deployment ({pf.DEPLOYMENT_REF})"
#: Historical phrases that state the llama.cpp np4 geometry, replaced in every carried-over text (and refused if any survives).
GEOMETRY_PHRASES = (
    ("the resident np4 Thinking session (serving-slurp-thinking-module-fold-b-np4-budget3072-max4096-v1)", PORT_SESSION_TEXT),
    ("the resident np4 Thinking session", PORT_SESSION_TEXT),
)
#: Facts of the HISTORICAL arm table (c15, K = 4) that must not survive in any port text: the port arm table has the replicate (c17, K = 5).
STALE_ARM_FACTS = re.compile(r"\bc15\b|-c15\b|\bK = 4\b|\b4[- ]arms?\b|\b15 contacts\b|\b7 Perception\b|\b8 Solver\b|\b1,080 x 15\b|\b4 x 15\b|\b32,400\b|\b64,800\b|\b16,200\b|\b94,644\b")
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


def leaf_allowed(leaf: str, prefixes) -> bool:
    """True when ``leaf`` is one of ``prefixes`` or below one: ``p`` itself, ``p.`` / ``p[`` continuations, or any leaf starting with a prefix that ends in ``.`` or ``[``."""

    return any(leaf == p or leaf.startswith(p + ".") or leaf.startswith(p + "[") or (p.endswith((".", "[")) and leaf.startswith(p)) for p in prefixes)


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


def _replace_once_exact(text: str, old: str, new: str, what: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"expected exactly one {what} sentence in a carried-over text, found {text.count(old)}: {old[:90]!r}")
    return text.replace(old, new)


def _program_sentence(*, program: str, k: int, contacts: int, fallback: int, perception: int, solver: int, rows: int, calls: int, fit_cap: int,
                      endpoints: int) -> str:
    """The generation-budget sentence the historical derive writes (``l338_derive_cells.derive_experiment``), with the arm-table figures as parameters."""

    return (f"program {program} with the {k}-arm sampling column: {rows:,} x {contacts} = {rows * contacts:,} is the exact program maximum "
            f"(2 + 2 (K - 1) + (k - 1) + (n - 1), K = {k}); the plan ceiling adds {fallback} answer-only fallback contacts per row (one per budgeted-thinking contact: "
            f"{perception} Perception + {solver} Solver contacts, core.reasoning_knobs): {calls:,} = {rows:,} x {contacts + fallback}. The live phase fit "
            f"makes {rows} x {contacts} x {endpoints} = {fit_cap:,} tokenizer endpoint calls")


def program_sentences(key: str) -> tuple[str, str]:
    """(historical, port) generation-budget sentence of a cell."""

    rows = fam.CELLS[key]["rows"]
    old = _program_sentence(program=fam.PROGRAM, k=fam.K, contacts=fam.CONTACTS_PER_ROW, fallback=fam.FALLBACK_CONTACTS_PER_ROW_CEILING,
                            perception=fam.PERCEPTION_CONTACTS_PER_ROW, solver=fam.SOLVER_CONTACTS_PER_ROW, rows=rows,
                            calls=fam.original_generation_calls(key), fit_cap=fam.fit_contact_cap(rows),
                            endpoints=fam.tf.FIT_ENDPOINTS_PER_CONTACT)
    new = _program_sentence(program=pf.PROGRAM, k=pf.K, contacts=pf.CONTACTS_PER_ROW, fallback=pf.FALLBACK_CONTACTS_PER_ROW_CEILING,
                            perception=pf.PERCEPTION_CONTACTS_PER_ROW, solver=pf.SOLVER_CONTACTS_PER_ROW, rows=rows, calls=pf.generation_calls(key),
                            fit_cap=pf.fit_calls(key), endpoints=pf.RAW_FIT_ENDPOINTS_PER_CONTACT)
    return old, new


REPLICATE_PHRASE = (f"the registered replicate{'s' if len(pf.REPLICATE_ARMS) > 1 else ''} {', '.join(pf.REPLICATE_ARMS)} (noise model M1: a further request-identical "
                    f"copy of {fam.DELIVERING} beside the twin, arm_kind replicate, no axis and no seed of its own; it adds one Perception and one Solver contact per row)")


def port_purpose(key: str, text: str) -> str:
    """The carried-over ``purpose`` with its arm-table sentence rewritten for the port arm table."""

    if key == "smoke":
        old = f"that {fam.PROGRAM} (ceiling {fam.CONTACTS_PER_ROW}) with the {fam.K}-arm sampling column loads"
        new = f"that {pf.PROGRAM} (ceiling {pf.CONTACTS_PER_ROW}) with the {pf.K}-arm sampling column loads"
        return _replace_once_exact(text, old, new, "smoke purpose")
    old = (f"One flight, {fam.K} arms over TRAIN Fold B 1,080: the delivering {fam.DELIVERING} arm (the 268-B lock form), the in-chain twin {fam.TWIN} and the candidate arms ")
    new = (f"One flight, {pf.K} arms over TRAIN Fold B 1,080: the delivering {fam.DELIVERING} arm (the 268-B lock form), the in-chain twin {fam.TWIN}, "
           f"{REPLICATE_PHRASE} and the candidate arms ")
    return _replace_once_exact(text, old, new, "flight purpose")


def port_what_this_is(key: str, text: str) -> str:
    """The baseline's carried-over ``what_this_is`` with the arm-table sentences rewritten for the port arm table."""

    historical_role, port_role = dc.ROLE_TEXT[key], None
    if key == "arm":
        port_role = f"The lever-column flight of L338: {pf.K} arms over the same 1,080 Fold B rows in one flight."
    else:
        port_role = historical_role.replace(fam.PROGRAM, pf.PROGRAM).replace(f"with {fam.K} sampling arms", f"with {pf.K} sampling arms")
    text = _replace_once_exact(text, historical_role, port_role, "role")
    old_twin = f"the in-chain twin {fam.TWIN} (replicate_of {fam.DELIVERING}) and the delivering arm "
    new_twin = f"the in-chain twin {fam.TWIN} (replicate_of {fam.DELIVERING}), {REPLICATE_PHRASE} and the delivering arm "
    text = _replace_once_exact(text, old_twin, new_twin, "twin")
    old_prog = (f"Program {fam.PROGRAM}: 2 + 2 ({fam.K} - 1) + sampling extras = {fam.CONTACTS_PER_ROW} contacts per row ({fam.PERCEPTION_CONTACTS_PER_ROW} Perception + "
                f"{fam.SOLVER_CONTACTS_PER_ROW} Solver).")
    new_prog = (f"Program {pf.PROGRAM}: 2 + 2 ({pf.K} - 1) + sampling extras = {pf.CONTACTS_PER_ROW} contacts per row ({pf.PERCEPTION_CONTACTS_PER_ROW} Perception + "
                f"{pf.SOLVER_CONTACTS_PER_ROW} Solver).")
    return _replace_once_exact(text, old_prog, new_prog, "program")


def port_arm_table(base: dict, key: str, root: Path) -> dict:
    """The port baseline with the registered replicate of lock (noise model M1) added to the sampling column.

    The module block's ``sampling_frame`` gains ``replicates`` and the block is re-COMPILED by the module compiler (the rails' parser refuses the replicate rule by
    name); the compiled policy-owned keys replace the historical ones, so the sampling_variants block, the program id and every other compiled leaf are the compiler's
    output, never hand-edited. The derive asserts that only the declared leaves moved."""

    block = copy.deepcopy(base["native_episode"][cm.CONSUMPTION_MODULE_CONFIG_KEY])
    frame = block["sampling_frame"]
    if "replicates" in frame:
        raise ValueError("the historical baseline already names replicates")
    frame["replicates"] = list(pf.REPLICATE_ARMS)
    if pf.active_switch() != "default":
        # thinking-budget switch: the supply bound and the fit-estimate completion ceiling follow the raised cap (the compiler's fit refusal is the proof)
        view = pf.budget_view()
        frame["max_supplied_items"] = view["max_supplied_items"]
        frame["fit_estimate"] = dict(frame["fit_estimate"], completion_ceiling=view["completion_cap"])
    composed = cm.compile_module(block, carrier_module_hooks(repo_root=root), repo_root=root)
    if composed.program_id != pf.PROGRAM or composed.program_ceiling != pf.CONTACTS_PER_ROW:
        raise ValueError(f"compiled program {composed.program_id} / {composed.program_ceiling}, expected {pf.PROGRAM} / {pf.CONTACTS_PER_ROW}")
    # NOT ``composed.apply_to(base)``: that replaces EVERY policy-owned key with the lock layer's, including the knowledge-store pins the port re-authored. Only the two
    # compiled leaves the replicate moves are taken from the compiler's output; the assertion below proves nothing else moved.
    values = composed.values_on_roster_of(base)
    out = copy.deepcopy(base)
    out["policy"] = values["policy"]
    out["native_episode"]["roles"]["perception"]["sampling_variants"] = copy.deepcopy(values["native_episode"]["roles"]["perception"]["sampling_variants"])
    out["native_episode"][cm.CONSUMPTION_MODULE_CONFIG_KEY] = block
    arms = [arm["arm_id"] for arm in out["native_episode"]["roles"]["perception"]["sampling_variants"]["arms"]]
    if tuple(arms) != pf.ARM_ORDER:
        raise ValueError(f"the compiled sampling arms {arms} are not the port arm order {list(pf.ARM_ORDER)}")
    moved = moved_leaves(base, out)
    allowed = ("policy", "native_episode.consumption_module.sampling_frame.replicates", "native_episode.roles.perception.sampling_variants.arms[")
    if pf.active_switch() != "default":
        allowed += SWITCH_MOVED_LEAVES["baseline"]
    bad = [leaf for leaf in moved if not leaf_allowed(leaf, allowed)]
    if bad:
        raise ValueError(f"adding the replicate moved undeclared baseline leaves: {bad[:12]}")
    return out


def port_baseline(key: str, root: Path = REPO_ROOT, *, module_lock: str | None = None) -> dict:
    """The port baseline of ``key``. ``module_lock`` (default: keep the historical block's lock) is the ``lock`` key the module block declares; the L338 port and Fold A
    derives pass :data:`PORT_MODULE_LOCK`, so the block compiles onto the layer that carries the re-authored store pins."""

    spec = pf.CELLS[key]
    base = copy.deepcopy(_load("baseline", spec["historical_baseline"], root))
    if module_lock is not None:
        base["native_episode"]["consumption_module"]["lock"] = module_lock
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
    if (fit["run_id"], fit["attempt_id"]) != (spec["historical_fit_run"], fam.CELLS[key]["fit_attempt"]) or fit["contact_cap"] != fam.fit_contact_cap(spec["rows"]):
        raise ValueError(f"the historical {key} native_contact_fit is not the historical family's")
    base["native_contact_fit"] = dict(fit, run_id=spec["fit_run"], attempt_id=spec["fit_attempt"], contact_cap=pf.fit_calls(key))
    base["what_this_is"] = (
        f"The NPU-port twin of {spec['historical_cell']} (design doc {pf.DESIGN_DOC}): the historical baseline {spec['historical_baseline']} with the "
        f"BF16 / vLLM-Ascend serving path (arm {pins['arm']}: the original recordings under {pf.PROMPT_ORDER}, no promoted speech packets), the "
        "re-authored semantic stores (intent-boundaries-npu-port-v1, slot-contracts-npu-port-v2) and every retrieval input re-pinned to the port's own "
        "sealed artifacts (scripts/flights/nx/l338_port_pins.json); native_contact_fit under the port FIT ids. The serving deployment is "
        f"{pf.DEPLOYMENT_REF}; the per-sequence window is {pf.PER_SEQUENCE_WINDOW_TOKENS:,} tokens. Everything else is the historical cell's: "
        + pf.switch_text(port_text(port_what_this_is(key, base["what_this_is"]))))
    base = port_arm_table(base, key, root)
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
        f"{spec['role'][0].upper()}{spec['role'][1:]}, NPU port: the twin of {hist['run']} on the BF16 / vLLM-Ascend Thinking core. Registered before contact "
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
    definition = _replace_once(definition, gpu_old, gpu_new)
    old_program, new_program = program_sentences(key)
    doc["generation_budget"]["definition"] = _replace_once_exact(definition, old_program, new_program, "program")
    # noise model M1: the caps follow the port arm table (c17); the historical caps are never typed here, they are the family's arithmetic
    if (doc["execution_plan"]["planned_model_calls"], doc["family_budget"]["fit_endpoint_calls_cap"],
            doc["family_budget"]["repeated_speech_audio_seconds_cap"]) != (fam.original_generation_calls(key), fam.fit_contact_cap(spec["rows"]), fam.audio_cap(spec["roster"])):
        raise ValueError(f"the historical {key} caps are not the historical family's")
    calls, fit_cap, audio = pf.generation_calls(key), pf.fit_calls(key), pf.audio_seconds(key)
    doc["execution_plan"].update(planned_model_calls=calls, planned_speech_audio_seconds=audio)
    doc["family_budget"].update(generation_calls_cap=calls, native_task_contract_calls_cap=calls, fit_endpoint_calls_cap=fit_cap, repeated_speech_audio_seconds_cap=audio)
    doc["generation_budget"].update(cap=calls, target=calls)
    crd = doc["core_read_design"]
    crd["arms"] = {arm: spec["run"] for arm in pf.ARM_ORDER}
    crd["reading"] = _replace_once_exact(
        crd["reading"], f"{', '.join(fam.CANDIDATE_ARMS)} and the twin {fam.TWIN} minus the delivering {fam.DELIVERING} arm",
        f"{', '.join(fam.CANDIDATE_ARMS)}, the twin {fam.TWIN} and the replicate {', '.join(pf.REPLICATE_ARMS)} minus the delivering {fam.DELIVERING} arm", "reading")
    if key == "arm":
        crd["this_cell"] = _replace_once_exact(crd["this_cell"], hist["role"], spec["role"], "this_cell")
    crd["chain"] = (f"deployment per {pf.DEPLOYMENT_REF} -> smoke -> machine sign-off (coordinator veto window) -> the column flight")
    doc["purpose"] = (
        f"{spec['run']}: the NPU-port twin of {hist['run']} (package {pf.HISTORICAL_PACKAGE}), served by the BF16 / vLLM-Ascend Thinking core "
        f"({pf.MODEL_FRAGMENT}; per-sequence window {pf.PER_SEQUENCE_WINDOW_TOKENS:,} tokens; deployment {pf.DEPLOYMENT_REF}) over the original recordings "
        f"({pf.PROMPT_ORDER}) with the port's re-authored knowledge and regenerated retrieval vectors; {pf.DIFFERENCES_TEXT}. It is read like the historical "
        "cell, never as a byte replay of it. The historical purpose, carried over: " + port_text(port_purpose(key, doc["purpose"])))
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
        bad = [leaf for leaf in moved if not leaf_allowed(leaf, moved_leaf_allowance(kind))]
        if key == "arm" and bad:
            raise ValueError(f"the derived port flight {kind} moves undeclared leaves: {bad[:12]}")
        out[kind] = {"committed": True, "moved": moved}
    return out


def scan_geometry(doc: dict) -> list[str]:
    """Text leaves that still state a llama.cpp slot geometry or a sequence count (refused)."""

    return [path for path, value in _leaves(doc) if isinstance(value, str) and FORBIDDEN.search(value)]


def module_self_check(root: Path, baselines: dict, experiments: dict, exempt: tuple[str, ...] = ()) -> dict:
    """Writer self-check (plan 3.14): run ``verify_declared_module`` -- the call the config loader makes at ``native_episode_config.py:1924`` -- on every emitted baseline
    (``{name: document}``) and on the baseline fragment each emitted experiment (``{name: document}``) names (an emitted baseline, else the one on disk). Raises ``ValueError``
    naming every failing config BEFORE anything is written. ``exempt`` lists baseline names that are known to fail for a separate reason (the never-flown comparators,
    whose ``k`` and component pins differ from the lock); they are checked and REPORTED, never refused. Returns ``{name: "ok" | "exempt: <leaves>"}``."""

    def check(doc: dict) -> str | None:
        block = (doc.get("native_episode") or {}).get(cm.CONSUMPTION_MODULE_CONFIG_KEY)
        if block is None:
            return None
        try:
            cm.verify_declared_module(block, doc, repo_root=Path(root))
        except cm.ConsumptionModuleConfigError as error:
            return str(error)[:600]
        return None

    needed = dict(baselines)
    for name, exp in experiments.items():
        for cell in exp["cells"]:
            fragment = cell["baseline_fragment"]
            if fragment not in needed:
                needed[fragment] = _load("baseline", fragment, root)
    report, failures = {}, []
    for name in sorted(needed):
        error = check(needed[name])
        if error is None:
            report[name] = "ok"
        elif name in exempt:
            report[name] = f"exempt: {error}"
        else:
            failures.append(f"{name}: {error}")
    if failures:
        raise ValueError("verify_declared_module refuses the derived configs (nothing was written): " + "; ".join(failures))
    return report


def verify_composed_module(root: Path, values: dict, label: str) -> None:
    """The loader's call (``rails.native_episode_config`` :1924) on a COMPOSED config: ``verify_declared_module(values["native_episode"]["consumption_module"], values)``.
    Raises ``SystemExit`` naming ``label`` when it refuses; a config that declares no module block passes."""

    block = (values.get("native_episode") or {}).get(cm.CONSUMPTION_MODULE_CONFIG_KEY)
    if block is None:
        return
    try:
        cm.verify_declared_module(block, values, repo_root=Path(root))
    except cm.ConsumptionModuleConfigError as error:
        raise SystemExit(f"{label}: the composed config is refused by verify_declared_module (the config loader would refuse it): {str(error)[:600]}") from error


def derive(repo_root: Path = REPO_ROOT, dry_run: bool = False) -> dict:
    root = Path(repo_root)
    result = {}
    docs = {}
    for key in pf.ALL_KEYS:
        docs[key] = (port_baseline(key, root, module_lock=PORT_MODULE_LOCK), port_experiment(key, root))
    module_report = module_self_check(root, {pf.CELLS[k]["baseline"]: docs[k][0] for k in docs}, {pf.CELLS[k]["cell"]: docs[k][1] for k in docs})
    for key in pf.ALL_KEYS:
        spec = pf.CELLS[key]
        base, exp = docs[key]
        leaks = [f"baseline:{p}" for p in scan_geometry(base)] + [f"experiment:{p}" for p in scan_geometry(exp)]
        if leaks:
            raise ValueError(f"port {key} texts still state a slot geometry: {leaks}")
        stale = [f"{kind}:{p}" for kind, doc in (("baseline", base), ("experiment", exp)) for p, v in _leaves(doc) if isinstance(v, str) and STALE_ARM_FACTS.search(v)]
        if stale:
            raise ValueError(f"port {key} texts still state a fact of the historical arm table (c15 / K = 4): {stale}")
        check = verify_against_committed(key, base, exp, root)
        if not dry_run:
            for kind, doc, name in (("baseline", base, spec["baseline"]), ("experiment", exp, spec["cell"])):
                (root / "configs" / kind / f"{name}.json").write_text(json.dumps(doc, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result[key] = {"cell": spec["cell"], "baseline": spec["baseline"], "run_id": spec["run"], "fit_run_id": spec["fit_run"], "versus_committed": check,
                       "module_self_check": {"lock": PORT_MODULE_LOCK, "verify_declared_module": module_report[spec["baseline"]]}}
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
