#!/usr/bin/env python3
"""tfrl_v2_distiller.py -- distil the exemplar bank, the contrast-pair rule bank and the ``tfrl`` trace lines from ONE half's sampled trajectories
(TFRL-v2, plan 2026-10-30 sections 4 and 9, task T3).

NO MODEL CALL. The "reflector" step of the plan is, here, a deterministic TEMPLATE over the difference of two trajectories' parsed answers (slots
added, slots dropped, slot values changed, intent flips). It is not an LLM call and consults no model, tokenizer service, encoder, network or
card; the module constructs no client of any kind. Same inputs give the same bytes. (A model-written reflector remains possible later as a
separate, registered contact; nothing here depends on one.)

Input: the Solver ``contact`` lines and the per-row ``score`` lines of ONE half's construction flight (``saea-rollout-trace-v1``), that half's
roster, the iteration and K. The credit (reward = row F1 of the trajectory minus row F1 of the baseline arm ``lock``; positive rows) is
``tfrl_v2_sampler.credit_rows``.

Outputs, all write-once, under one directory:

(a) ``exemplar-bank.json``  scenario-keyed exemplars. One exemplar per POSITIVE row: the answer of the row's best (gold-aligned, intent-correct,
    highest row F1) trajectory plus the supplied items that answer used (``supply_items[].used``). Per scenario the entries are ranked (reward,
    row F1, row id) and kept while the rendered per-scenario block stays within ``EXEMPLAR_BLOCK_TOKEN_CAP`` (2,000) tokens and
    ``MAX_EXEMPLARS_PER_SCENARIO`` entries; the token measure is injected (``tfrl_v2_text``: the vLLM tokenize measure for a flight-bound bank, the offline ``ceil(chars / 3.5)`` estimate otherwise) and
    is named in the bank receipt.
(b) ``rule-bank.json``      consumption rules. Each positive row with a contrast pair (best vs worst trajectory of the same row, best F1 above worst
    F1) contributes the candidate rules its answer difference implies. A candidate survives only when at least ``VOTE_MIN_ROWS`` (3) DISTINCT
    rows support it (the vote n = 3); survivors are ranked (support, mean F1 gap, id) and kept while the rendered rule text stays within
    ``RULE_BANK_TOKEN_CAP`` (400) tokens. A candidate without a contrast pair cannot exist; refused candidates are listed with the reason.
(c) ``tfrl-trace.jsonl``    one ``tfrl`` line (``saea-rollout-trace-v1``) per sampled trajectory: its id, reward, whether it became an exemplar.
(d) ``bank-receipt.json``   content hashes of both banks, file hashes of every output, the source half id and split identity, the contributing
    source row ids, the consuming half (the OTHER half) and the counts. :func:`tfrl_v2_common.check_cross_fit` reads this receipt to refuse a
    half consuming a bank built from itself.

Also written: ``credit.jsonl`` (per-row credit), ``contrast-pairs.jsonl`` and ``refused-rules.jsonl``.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for _path in (str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts" / "analysis"), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from speech_aware_evidence_acquisition.contracts import rollout_trace as rt  # noqa: E402
import tfrl_v2_common as cm  # noqa: E402
import tfrl_v2_sampler as sp  # noqa: E402
import tfrl_v2_text as tx  # noqa: E402

EXEMPLAR_SCHEMA = "saea-tfrl-v2-exemplar-bank-v1"
RULE_SCHEMA = "saea-tfrl-v2-rule-bank-v1"
RECEIPT_SCHEMA = "saea-tfrl-v2-bank-receipt-v1"
PAIR_SCHEMA = "saea-tfrl-v2-contrast-pair-v1"
_LABEL = re.compile(r"[A-Za-z0-9_\-]+\Z")                 # scenario / action / slot type names that may enter a rule or an exemplar
_FILLER = re.compile(r"[ -~]+\Z")                         # printable ASCII filler text


class DistillError(ValueError):
    """A refusal of the distiller: incomplete trajectories, a row outside the source half, an unsafe label."""


# ----------------------------------------------------------------------------------------------------------------------------------------
# answers
# ----------------------------------------------------------------------------------------------------------------------------------------


def _clean(text: str) -> str:
    return " ".join(text.split())


def _norm_entity(entity: Mapping[str, str]) -> tuple[str, str]:
    return (_clean(entity["type"]).lower(), _clean(entity["filler"]).lower())


def answer_intent(answer: Mapping[str, Any]) -> str:
    return f"{answer['scenario']}_{answer['action']}"


def labels_safe(answer: Mapping[str, Any]) -> bool:
    """True when scenario, action and every slot type are plain label tokens and every filler is non-empty printable ASCII."""

    if not (_LABEL.match(answer["scenario"]) and _LABEL.match(answer["action"])):
        return False
    return all(_LABEL.match(e["type"]) and _FILLER.match(_clean(e["filler"]) or "\x00") for e in answer["entities"])


def answer_diff(best: Mapping[str, Any], worst: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The difference the worst answer shows against the best one, as rule-candidate descriptors (sorted, deterministic).

    * ``intent_flip``   scenario or action differs (descriptor: the best and the worst intent);
    * ``slot_missing``  the best answer has a slot of a type that the worst answer dropped;
    * ``slot_spurious`` the worst answer added a slot of a type the best answer does not have;
    * ``slot_value``    both answers have that slot type but with different fillers.
    """

    out: list[dict[str, Any]] = []
    if (best["scenario"], best["action"]) != (worst["scenario"], worst["action"]):
        out.append({"kind": "intent_flip", "scenario": best["scenario"], "best_intent": answer_intent(best), "worst_intent": answer_intent(worst)})
    best_ents = Counter(_norm_entity(e) for e in best["entities"])
    worst_ents = Counter(_norm_entity(e) for e in worst["entities"])
    missing = Counter(t for (t, _f), n in (best_ents - worst_ents).items() for _ in range(n))
    spurious = Counter(t for (t, _f), n in (worst_ents - best_ents).items() for _ in range(n))
    for slot_type in sorted(set(missing) | set(spurious)):
        if slot_type in missing and slot_type in spurious:
            kind = "slot_value"
        elif slot_type in missing:
            kind = "slot_missing"
        else:
            kind = "slot_spurious"
        out.append({"kind": kind, "scenario": best["scenario"], "slot_type": slot_type})
    return out


def candidate_key(candidate: Mapping[str, Any]) -> str:
    if candidate["kind"] == "intent_flip":
        return f"intent_flip|{candidate['best_intent']}|{candidate['worst_intent']}"
    return f"{candidate['kind']}|{candidate['scenario']}|{candidate['slot_type']}"


def rule_text(candidate: Mapping[str, Any]) -> str:
    """The fixed English template of a rule candidate (an instruction to the Solver)."""

    kind, scenario = candidate["kind"], candidate["scenario"]
    if kind == "slot_missing":
        return f"In {scenario} requests, report the {candidate['slot_type']} entity whenever it is spoken; do not leave it out."
    if kind == "slot_spurious":
        return f"In {scenario} requests, do not report a {candidate['slot_type']} entity unless it is clearly spoken."
    if kind == "slot_value":
        return f"In {scenario} requests, give the {candidate['slot_type']} value exactly as it is spoken."
    if kind == "intent_flip":
        return f"When a request fits {candidate['best_intent']}, do not answer {candidate['worst_intent']}."
    raise DistillError(f"unknown rule kind {kind!r}")


# ----------------------------------------------------------------------------------------------------------------------------------------
# exemplars
# ----------------------------------------------------------------------------------------------------------------------------------------


def render_exemplar(answer: Mapping[str, Any], evidence_ids: Sequence[str], resolver: tx.ItemTextResolver | None = None) -> str:
    """One exemplar.

    Without a resolver (legacy, ids only): one line ``- evidence: <ids> -> intent: ...; entities: ...``. With a resolver: the supplied items the
    answer used are quoted EXACTLY as the Solver read them (``[evidence <id>] <content>``, :func:`tfrl_v2_text.render_evidence_line`), followed by
    the answer; an unknown id is refused (:class:`DistillError`)::

        - worked example
        [evidence train-exemplars::value:1] wake me up at nine am on friday | alarm_set | date=friday; time=nine am
        answer: intent: alarm_set; entities: date=friday; time=nine am
    """

    entities = "; ".join(f"{e['type']}={_clean(e['filler'])}" for e in answer["entities"]) or "none"
    if resolver is None:
        evidence = ", ".join(evidence_ids) or "none"
        return f"- evidence: {evidence} -> intent: {answer_intent(answer)}; entities: {entities}"
    try:
        quoted = [resolver.resolve(item_id).rendered for item_id in evidence_ids]
    except tx.ItemTextError as error:
        raise DistillError(str(error)) from error
    return "\n".join(["- worked example"] + (quoted or ["(no supplied item was used)"]) + [f"answer: intent: {answer_intent(answer)}; entities: {entities}"])


def render_block(scenario: str, lines: Sequence[str]) -> str:
    return "\n".join([f"Worked examples for {scenario} requests:"] + list(lines))


def _fill_scenario(
    scenario: str, ranked: Sequence[Mapping[str, Any]], *, measure: tx.TokenMeasure, token_cap: int, max_per_scenario: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Walk ranked candidate entries and keep them while the scenario block stays within the count and token caps.

    Returns ``(kept, fate)``: the kept entries (each with its ``exemplar_id`` and ``tokens``) and, per source trajectory id, ``banked`` or the
    reason it was dropped (``duplicate-text``, ``scenario-count-cap``, ``token-cap``).
    """

    kept: list[dict[str, Any]] = []
    fate: dict[str, str] = {}
    seen_text: set[str] = set()
    for entry in ranked:
        tid = entry["source_trajectory_id"]
        if entry["text"] in seen_text:
            fate[tid] = "duplicate-text"
            continue
        if len(kept) >= max_per_scenario:
            fate[tid] = "scenario-count-cap"
            continue
        trial = render_block(scenario, [e["text"] for e in kept] + [entry["text"]])
        if measure.count(trial) > token_cap:
            fate[tid] = "token-cap"
            continue
        entry = dict(entry, exemplar_id="ex-" + cm.content_sha256({k: entry[k] for k in ("scenario", "text", "source_row_id")})[:12],
                     tokens=measure.count(entry["text"]))
        kept.append(entry)
        seen_text.add(entry["text"])
        fate[tid] = "banked"
    return kept, fate


def _rank_key(entry: Mapping[str, Any]) -> tuple[float, float, str]:
    return (-entry["reward"], -entry["row_f1"], entry["source_row_id"])


def build_exemplar_bank(
    credit: Sequence[Mapping[str, Any]], answers: Mapping[tuple[str, str], Mapping[str, Any]], *, iteration: int, source_half: str,
    token_cap: int = cm.EXEMPLAR_BLOCK_TOKEN_CAP, max_per_scenario: int = cm.MAX_EXEMPLARS_PER_SCENARIO,
    max_evidence: int = cm.MAX_EVIDENCE_ITEMS_PER_EXEMPLAR, resolver: tx.ItemTextResolver | None = None, measure: tx.TokenMeasure | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """``(bank, fate)``: the scenario-keyed bank and, per trajectory id of every positive row's best trajectory, what became of it.

    ``resolver`` (when given) puts the exact supplied-item text into every exemplar and a content hash per item into the entry
    (``evidence_items``); ``measure`` is the token measure of the caps (default: the offline estimate). The bank records both.
    """

    measure = measure or tx.default_measure()
    candidates: dict[str, list[dict[str, Any]]] = {}
    fate: dict[str, str] = {}
    for row in credit:
        if not row["positive"]:
            continue
        best = row["trajectories"][row["best_k"]]
        contact = answers[(row["row_id"], best["arm"])]
        answer = contact["answer"]
        if not labels_safe(answer):
            fate[best["trajectory_id"]] = "unsafe-label"
            continue
        used = tx.used_items(contact["supply_items"], answer, resolver, max_evidence)
        ids = [u["item_id"] for u in used]
        entry = {
            "scenario": answer["scenario"], "action": answer["action"], "entities": [{"type": e["type"], "filler": _clean(e["filler"])} for e in answer["entities"]],
            "evidence_item_ids": ids, "reward": best["reward"], "row_f1": best["row_f1"], "source_row_id": row["row_id"],
            "source_trajectory_id": best["trajectory_id"], "text": render_exemplar(answer, ids, resolver),
        }
        if resolver is not None:
            entry["evidence_items"] = [{"item_id": u["item_id"], "content_sha256": resolver.resolve(u["item_id"]).content_sha256, "used_basis": u["basis"]}
                                       for u in used]
        candidates.setdefault(answer["scenario"], []).append(entry)
    by_scenario: dict[str, Any] = {}
    for scenario in sorted(candidates):
        kept, scenario_fate = _fill_scenario(scenario, sorted(candidates[scenario], key=_rank_key), measure=measure, token_cap=token_cap,
                                             max_per_scenario=max_per_scenario)
        fate.update(scenario_fate)
        if kept:
            block = render_block(scenario, [e["text"] for e in kept])
            by_scenario[scenario] = {"entries": kept, "block_text": block, "block_tokens": measure.count(block)}
    bank = {"schema": EXEMPLAR_SCHEMA, "iteration": iteration, "source_half": source_half, "block_token_cap": token_cap,
            "max_per_scenario": max_per_scenario, "token_measure": measure.name,
            "exemplar_text": tx.EXEMPLAR_TEXT_RESOLVED if resolver is not None else tx.EXEMPLAR_TEXT_IDS_ONLY, "by_scenario": by_scenario}
    return bank, fate


# ----------------------------------------------------------------------------------------------------------------------------------------
# rules
# ----------------------------------------------------------------------------------------------------------------------------------------


def build_contrast_pairs(credit: Sequence[Mapping[str, Any]], answers: Mapping[tuple[str, str], Mapping[str, Any]], iteration: int) -> list[dict[str, Any]]:
    """One pair per positive row whose best F1 is strictly above its worst F1 (best vs worst trajectory of the same row)."""

    pairs: list[dict[str, Any]] = []
    for row in credit:
        if not row["positive"] or row["worst_k"] is None:
            continue
        best, worst = row["trajectories"][row["best_k"]], row["trajectories"][row["worst_k"]]
        gap = best["row_f1"] - worst["row_f1"]
        if gap <= sp.GAIN_EPSILON:
            continue
        best_answer = answers[(row["row_id"], best["arm"])]["answer"]
        worst_answer = answers[(row["row_id"], worst["arm"])]["answer"]
        if not (labels_safe(best_answer) and labels_safe(worst_answer)):
            continue
        pairs.append({
            "schema": PAIR_SCHEMA, "pair_id": f"{row['row_id']}|k{best['k_index']}>k{worst['k_index']}", "iteration": iteration, "row_id": row["row_id"],
            "best_trajectory_id": best["trajectory_id"], "worst_trajectory_id": worst["trajectory_id"], "best_f1": best["row_f1"],
            "worst_f1": worst["row_f1"], "gap": gap, "diff": answer_diff(best_answer, worst_answer),
        })
    return pairs


def build_rule_bank(
    pairs: Sequence[Mapping[str, Any]], *, iteration: int, source_half: str, vote_min: int = cm.VOTE_MIN_ROWS,
    token_cap: int = cm.RULE_BANK_TOKEN_CAP, measure: tx.TokenMeasure | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``(rule_bank, refused)``. A candidate needs ``vote_min`` distinct supporting rows; survivors fill the token cap in rank order."""

    measure = measure or tx.default_measure()
    support: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        for candidate in pair["diff"]:
            key = candidate_key(candidate)
            slot = support.setdefault(key, {"candidate": candidate, "rows": {}, "pair_ids": [], "gaps": []})
            if pair["row_id"] not in slot["rows"]:           # one vote per row
                slot["rows"][pair["row_id"]] = pair["gap"]
                slot["pair_ids"].append(pair["pair_id"])
                slot["gaps"].append(pair["gap"])
    refused: list[dict[str, Any]] = []
    survivors: list[dict[str, Any]] = []
    for key in sorted(support):
        slot = support[key]
        cand = slot["candidate"]
        if len(slot["rows"]) < vote_min:
            refused.append({"candidate": key, "reason": "below-vote", "support_rows": len(slot["rows"]), "vote_min": vote_min})
            continue
        survivors.append({
            "rule_id": "r-" + cm.content_sha256(key)[:12], "kind": cand["kind"], "text": rule_text(cand), "scenario": cand["scenario"],
            "slot_type": cand.get("slot_type"), "best_intent": cand.get("best_intent"), "worst_intent": cand.get("worst_intent"),
            "candidate_key": key, "support_count": len(slot["rows"]), "support_rows": sorted(slot["rows"]), "pair_ids": sorted(slot["pair_ids"]),
            "mean_gap": round(sum(slot["gaps"]) / len(slot["gaps"]), 9),
        })
    survivors.sort(key=lambda r: (-r["support_count"], -r["mean_gap"], r["rule_id"]))
    kept: list[dict[str, Any]] = []
    for rule in survivors:
        if measure.count(render_rules([r["text"] for r in kept] + [rule["text"]])) > token_cap:
            refused.append({"candidate": rule["candidate_key"], "reason": "token-cap", "support_rows": rule["support_count"], "vote_min": vote_min})
            continue
        kept.append(rule)
    rendered = render_rules([r["text"] for r in kept]) if kept else ""
    bank = {"schema": RULE_SCHEMA, "iteration": iteration, "source_half": source_half, "vote_min": vote_min, "token_cap": token_cap,
            "token_measure": measure.name, "rules": kept, "rendered_text": rendered, "rendered_tokens": measure.count(rendered)}
    return bank, refused


def render_rules(texts: Sequence[str]) -> str:
    return "\n".join(["Consumption rules:"] + [f"- {t}" for t in texts])


# ----------------------------------------------------------------------------------------------------------------------------------------
# trace lines
# ----------------------------------------------------------------------------------------------------------------------------------------


def build_tfrl_lines(
    credit: Sequence[Mapping[str, Any]], answers: Mapping[tuple[str, str], Mapping[str, Any]], fate: Mapping[str, str],
    pairs: Sequence[Mapping[str, Any]], *, iteration: int,
) -> list[dict[str, Any]]:
    """One ``tfrl`` line per sampled trajectory. ``selected`` = the trajectory became an exemplar of the bank."""

    negative_ids = {p["worst_trajectory_id"] for p in pairs}
    lines: list[dict[str, Any]] = []
    for row in credit:
        for traj in row["trajectories"]:
            contact = answers[(row["row_id"], traj["arm"])]
            tid = traj["trajectory_id"]
            if row["positive"] and traj["k_index"] == row["best_k"]:
                state = fate.get(tid, "unbanked")
                selected, reason = state == "banked", ("exemplar-source" if state == "banked" else f"positive-best-not-banked:{state}")
            elif tid in negative_ids:
                selected, reason = False, "contrast-negative"
            elif row["positive"]:
                selected, reason = False, "positive-row-other-trajectory"
            else:
                selected, reason = False, f"non-positive-row:{row['reason']}"
            lines.append({
                "schema": rt.SCHEMA, "kind": rt.KIND_TFRL, "ts": contact["ts"], "run_id": contact["run_id"], "attempt_id": contact["attempt_id"],
                "row_id": row["row_id"], "arm": traj["arm"], "iteration": iteration, "trajectory_id": tid, "reward": traj["reward"],
                "selected": selected, "reason": reason,
            })
    for line in lines:
        rt.validate_line(line)
    return lines


# ----------------------------------------------------------------------------------------------------------------------------------------
# the whole distillation
# ----------------------------------------------------------------------------------------------------------------------------------------


def distil(
    half_roster: Mapping[str, Any], source_half: str, iteration: int, contact_lines: Sequence[Mapping[str, Any]],
    score_lines: Sequence[Mapping[str, Any]], *, k: int = cm.DEFAULT_K, scorer_id: str | None = None, with_twin: bool = True,
    allow_missing_rows: bool = False, roster_file_sha256: str | None = None, trace_sha256s: Sequence[str] = (),
    resolver: tx.ItemTextResolver | None = None, measure: tx.TokenMeasure | None = None,
) -> dict[str, Any]:
    """Distil one half. Returns the documents (nothing is written): ``credit``, ``pairs``, ``exemplar_bank``, ``rule_bank``, ``refused``,
    ``tfrl_lines``, ``receipt_core``. Rows of other halves in the trace are ignored (counted); a roster row without all K trajectories and
    a baseline score is a refusal unless ``allow_missing_rows`` (then the row is skipped and counted).

    ``resolver`` puts the exact supplied-item text into the exemplars (an unknown id is refused); ``measure`` is the token measure of the caps. The
    defaults (no resolver, the offline estimate) give a bank the sampler's plan builder refuses: a flight-bound bank needs both."""

    measure = measure or tx.default_measure()
    items = sp._half_rows(half_roster, source_half)
    row_ids = [str(i["sample_id"]) for i in items]
    in_half = set(row_ids)
    arms = sp.sampler_arms(k)
    own_contacts = [c for c in contact_lines if c["row_id"] in in_half]
    answers = sp.solver_contacts(own_contacts, arms)
    scores = sp.scores_from_lines([s for s in score_lines if s["row_id"] in in_half], scorer_id=scorer_id)
    usable: list[str] = []
    skipped: list[str] = []
    twin = sp.TWIN_ARM if with_twin else None
    for row_id in row_ids:
        complete = all((row_id, arm) in answers and (row_id, arm) in scores for arm in arms) and (row_id, sp.BASELINE_ARM) in scores \
            and (twin is None or (row_id, twin) in scores)
        (usable if complete else skipped).append(row_id)
    if skipped and not allow_missing_rows:
        raise DistillError(f"{len(skipped)} roster rows lack K={k} delivered Solver answers with scores (first: {skipped[0]}); pass allow_missing_rows to skip them")
    if not usable:
        raise DistillError("no row has a complete set of trajectories")
    credit = sp.credit_rows(usable, iteration, k, scores, twin_arm=twin)
    exemplar_bank, fate = build_exemplar_bank(credit, answers, iteration=iteration, source_half=source_half, resolver=resolver, measure=measure)
    pairs = build_contrast_pairs(credit, answers, iteration)
    rule_bank, refused = build_rule_bank(pairs, iteration=iteration, source_half=source_half, measure=measure)
    tfrl_lines = build_tfrl_lines(credit, answers, fate, pairs, iteration=iteration)

    source_rows = sorted({e["source_row_id"] for s in exemplar_bank["by_scenario"].values() for e in s["entries"]}
                         | {r for rule in rule_bank["rules"] for r in rule["support_rows"]})
    receipt_core = {
        "schema": RECEIPT_SCHEMA, "loop_id": cm.LOOP_ID, "iteration": iteration, "source_half": source_half,
        "kind": "half", "measure": measure.name, "measure_detail": dict(measure.detail()),
        "exemplar_text": exemplar_bank["exemplar_text"], "exemplar_text_source": resolver.identity() if resolver is not None else None,
        "source_halves": [source_half], "consuming_half": cm.other_half(source_half),
        "source_split_identities": {source_half: half_roster["split_identity_hash"]}, "source_roster_file_sha256": roster_file_sha256,
        "source_half_rows": len(row_ids), "source_row_ids": source_rows,
        "exemplar_bank": {"content_sha256": cm.content_sha256(exemplar_bank), "scenarios": len(exemplar_bank["by_scenario"]),
                          "entries": sum(len(s["entries"]) for s in exemplar_bank["by_scenario"].values()),
                          "max_block_tokens": max((s["block_tokens"] for s in exemplar_bank["by_scenario"].values()), default=0),
                          "block_token_cap": cm.EXEMPLAR_BLOCK_TOKEN_CAP, "measure": measure.name},
        "rule_bank": {"content_sha256": cm.content_sha256(rule_bank), "rules": len(rule_bank["rules"]), "rendered_tokens": rule_bank["rendered_tokens"],
                      "token_cap": cm.RULE_BANK_TOKEN_CAP, "vote_min": cm.VOTE_MIN_ROWS, "measure": measure.name},
        "credit_summary": sp.credit_summary(credit),
        "counts": {"rows_in_half": len(row_ids), "rows_used": len(usable), "rows_skipped": len(skipped), "rows_in_trace_outside_half": len(
            {c["row_id"] for c in contact_lines} - in_half), "contrast_pairs": len(pairs), "rules_refused": len(refused),
            "tfrl_lines": len(tfrl_lines)},
        "inputs": {"trace_sha256s": sorted(trace_sha256s), "k": k, "baseline_arm": sp.BASELINE_ARM, "twin_arm": twin, "scorer_id": scorer_id},
        "method": "deterministic template over the answer difference of the best and worst trajectory of a row; no model call",
        "model_calls": 0,
    }
    return {"credit": credit, "pairs": pairs, "exemplar_bank": exemplar_bank, "rule_bank": rule_bank, "refused": refused, "tfrl_lines": tfrl_lines,
            "receipt_core": receipt_core}


def write_outputs(result: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    """Write every output (write-once) and the bank receipt last. Returns the receipt."""

    out = Path(out_dir)
    files = {
        "credit.jsonl": cm.jsonl_bytes(result["credit"]),
        "contrast-pairs.jsonl": cm.jsonl_bytes(result["pairs"]),
        "exemplar-bank.json": cm.file_bytes(result["exemplar_bank"]),
        "rule-bank.json": cm.file_bytes(result["rule_bank"]),
        "refused-rules.jsonl": cm.jsonl_bytes(result["refused"]),
        "tfrl-trace.jsonl": cm.jsonl_bytes(result["tfrl_lines"]),
    }
    receipt = copy.deepcopy(dict(result["receipt_core"]))
    receipt["files"] = {name: {"sha256": cm.sha256_bytes(data), "bytes": len(data)} for name, data in files.items()}
    receipt["exemplar_bank"]["file"] = "exemplar-bank.json"
    receipt["rule_bank"]["file"] = "rule-bank.json"
    targets = {out / name: data for name, data in files.items()}
    targets[out / "bank-receipt.json"] = cm.file_bytes(receipt)
    for path, data in targets.items():
        if path.exists() and path.read_bytes() != data:
            raise DistillError(f"refusing to overwrite {path}: it exists with different bytes")
    for path, data in targets.items():
        cm.write_once(path, data)
    return receipt


def load_bank(receipt_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Read a bank receipt and its two banks, verifying every content hash. Returns ``(receipt, exemplar_bank, rule_bank)``."""

    path = Path(receipt_path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    banks = []
    for key in ("exemplar_bank", "rule_bank"):
        bank_path = path.parent / receipt[key]["file"]
        raw = bank_path.read_bytes()
        if cm.sha256_bytes(raw) != receipt["files"][receipt[key]["file"]]["sha256"]:
            raise DistillError(f"{bank_path}: file hash differs from the receipt")
        bank = json.loads(raw.decode("utf-8"))
        if cm.content_sha256(bank) != receipt[key]["content_sha256"]:
            raise DistillError(f"{bank_path}: content hash differs from the receipt")
        banks.append(bank)
    return receipt, banks[0], banks[1]


# ----------------------------------------------------------------------------------------------------------------------------------------
# the union bank (the bank the Fold B flight consumes)
# ----------------------------------------------------------------------------------------------------------------------------------------

LoadedBank = tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]


def union_banks(h1: LoadedBank, h2: LoadedBank, *, measure: tx.TokenMeasure | None = None) -> dict[str, Any]:
    """Merge the two halves' banks of one iteration into the union bank for the Fold B flight (nothing is written; see :func:`write_union_outputs`).

    Inputs are ``load_bank`` results (receipt, exemplar bank, rule bank; hashes verified). Refused: a receipt that is not a single-half receipt of
    its half, different iterations, a measure other than the one the receipts name, differing exemplar-text sources, overlapping source rows.

    * exemplars: the entries of both halves per scenario, re-ranked ``(reward desc, row F1 desc, row id)``, filled again under the per-scenario
      count cap (8) and the 2,000-token block cap with ``measure`` (a merged block is a new text and is re-measured), duplicate texts dropped;
    * rules: one rule per candidate key (the halves' supporting rows are disjoint, so supports add and the mean gap is support-weighted), ranked
      ``(support desc, mean gap desc, rule id)`` and filled under the 400-token cap. A candidate that missed the vote in BOTH halves is not
      re-pooled (the halves' banks carry only the survivors);
    * the receipt names both halves as sources, both split identities, the union of the contributing source rows, and the receipt and content
      hashes of its two inputs. :func:`tfrl_v2_common.check_union_for_fold_b` accepts it for Fold B rows and :func:`tfrl_v2_common.check_cross_fit`
      refuses it for either half.
    """

    measure = measure or tx.default_measure()
    inputs = {cm.HALF_H1: h1, cm.HALF_H2: h2}
    receipts = {half: inputs[half][0] for half in cm.HALVES}
    for half, (receipt, exemplars, rules) in inputs.items():
        if receipt.get("kind", "half") != "half" or receipt.get("source_halves") != [half]:
            raise DistillError(f"the {half} input is not the single-half receipt of {half} (source_halves {receipt.get('source_halves')!r})")
        if exemplars.get("source_half") != half or rules.get("source_half") != half:
            raise DistillError(f"the {half} banks name another source half")
    first, second = receipts[cm.HALF_H1], receipts[cm.HALF_H2]
    if first["iteration"] != second["iteration"]:
        raise DistillError("the two halves' banks are of different iterations")
    for key in ("measure", "exemplar_text", "exemplar_text_source"):
        if first.get(key) != second.get(key):
            raise DistillError(f"the two halves' banks differ in {key}")
    if first.get("measure") != measure.name:
        raise DistillError(f"the halves' counts were produced by measure {first.get('measure')!r}; the union must be re-measured with the same one, got {measure.name!r}")
    if set(first["source_row_ids"]) & set(second["source_row_ids"]):
        raise DistillError("the two halves share source rows")
    if first["source_split_identities"][cm.HALF_H1] == second["source_split_identities"][cm.HALF_H2]:
        raise DistillError("the two halves carry the same split identity")

    iteration = first["iteration"]
    candidates: dict[str, list[dict[str, Any]]] = {}
    for half in cm.HALVES:
        for scenario, block in inputs[half][1]["by_scenario"].items():
            candidates.setdefault(scenario, []).extend(dict(e, source_half=half) for e in block["entries"])
    by_scenario: dict[str, Any] = {}
    dropped: dict[str, int] = {}
    for scenario in sorted(candidates):
        kept, fate = _fill_scenario(scenario, sorted(candidates[scenario], key=_rank_key), measure=measure, token_cap=cm.EXEMPLAR_BLOCK_TOKEN_CAP,
                                    max_per_scenario=cm.MAX_EXEMPLARS_PER_SCENARIO)
        for state in fate.values():
            if state != "banked":
                dropped[state] = dropped.get(state, 0) + 1
        if kept:
            block_text = render_block(scenario, [e["text"] for e in kept])
            by_scenario[scenario] = {"entries": kept, "block_text": block_text, "block_tokens": measure.count(block_text)}
    exemplar_bank = {"schema": EXEMPLAR_SCHEMA, "iteration": iteration, "source_halves": list(cm.HALVES), "block_token_cap": cm.EXEMPLAR_BLOCK_TOKEN_CAP,
                     "max_per_scenario": cm.MAX_EXEMPLARS_PER_SCENARIO, "token_measure": measure.name, "exemplar_text": first["exemplar_text"],
                     "by_scenario": by_scenario}

    merged: dict[str, dict[str, Any]] = {}
    for half in cm.HALVES:
        for rule in inputs[half][2]["rules"]:
            slot = merged.setdefault(rule["candidate_key"], {"rule": rule, "rows": [], "pair_ids": [], "weighted_gap": 0.0})
            slot["rows"].extend(rule["support_rows"])
            slot["pair_ids"].extend(rule["pair_ids"])
            slot["weighted_gap"] += rule["mean_gap"] * rule["support_count"]
    survivors = []
    for key in sorted(merged):
        slot = merged[key]
        rows = sorted(slot["rows"])
        survivors.append(dict(slot["rule"], support_count=len(rows), support_rows=rows, pair_ids=sorted(slot["pair_ids"]),
                              mean_gap=round(slot["weighted_gap"] / len(rows), 9)))
    survivors.sort(key=lambda r: (-r["support_count"], -r["mean_gap"], r["rule_id"]))
    kept_rules: list[dict[str, Any]] = []
    for rule in survivors:
        if measure.count(render_rules([r["text"] for r in kept_rules] + [rule["text"]])) > cm.RULE_BANK_TOKEN_CAP:
            dropped["rule-token-cap"] = dropped.get("rule-token-cap", 0) + 1
            continue
        kept_rules.append(rule)
    rendered = render_rules([r["text"] for r in kept_rules]) if kept_rules else ""
    rule_bank = {"schema": RULE_SCHEMA, "iteration": iteration, "source_halves": list(cm.HALVES), "vote_min": cm.VOTE_MIN_ROWS,
                 "token_cap": cm.RULE_BANK_TOKEN_CAP, "token_measure": measure.name, "rules": kept_rules, "rendered_text": rendered,
                 "rendered_tokens": measure.count(rendered)}

    source_rows = sorted({e["source_row_id"] for s in by_scenario.values() for e in s["entries"]} | {r for rule in kept_rules for r in rule["support_rows"]})
    receipt_core = {
        "schema": RECEIPT_SCHEMA, "loop_id": cm.LOOP_ID, "iteration": iteration, "kind": "union", "measure": measure.name,
        "measure_detail": dict(measure.detail()), "exemplar_text": first["exemplar_text"], "exemplar_text_source": first["exemplar_text_source"],
        "source_halves": list(cm.HALVES), "consuming_half": None, "consumers": "the Fold B flight (and Fold A confirmation); never a construction half",
        "source_split_identities": {half: receipts[half]["source_split_identities"][half] for half in cm.HALVES},
        "source_roster_file_sha256": {half: receipts[half].get("source_roster_file_sha256") for half in cm.HALVES},
        "source_row_ids": source_rows,
        "sources": [{"half": half, "receipt_content_sha256": cm.content_sha256(receipts[half]),
                     "exemplar_bank_content_sha256": receipts[half]["exemplar_bank"]["content_sha256"],
                     "rule_bank_content_sha256": receipts[half]["rule_bank"]["content_sha256"]} for half in cm.HALVES],
        "order": {"exemplars": "scenario sorted; entries by (reward desc, row F1 desc, row id)", "rules": "by (support desc, mean gap desc, rule id)"},
        "exemplar_bank": {"content_sha256": cm.content_sha256(exemplar_bank), "scenarios": len(by_scenario),
                          "entries": sum(len(s["entries"]) for s in by_scenario.values()),
                          "max_block_tokens": max((s["block_tokens"] for s in by_scenario.values()), default=0),
                          "block_token_cap": cm.EXEMPLAR_BLOCK_TOKEN_CAP, "per_scenario_cap": cm.MAX_EXEMPLARS_PER_SCENARIO, "measure": measure.name},
        "rule_bank": {"content_sha256": cm.content_sha256(rule_bank), "rules": len(kept_rules), "rendered_tokens": rule_bank["rendered_tokens"],
                      "token_cap": cm.RULE_BANK_TOKEN_CAP, "vote_min": cm.VOTE_MIN_ROWS, "measure": measure.name},
        "counts": {"exemplar_entries_in": sum(len(c) for c in candidates.values()), "dropped": dict(sorted(dropped.items())),
                   "rules_in": sum(len(inputs[h][2]["rules"]) for h in cm.HALVES)},
        "method": "deterministic merge of the two halves' banks; no model call", "model_calls": 0,
    }
    return {"exemplar_bank": exemplar_bank, "rule_bank": rule_bank, "receipt_core": receipt_core}


def write_union_outputs(result: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    """Write the union bank, its rule bank and its receipt (write-once; the receipt last). Returns the receipt; ``load_bank`` reads it back."""

    out = Path(out_dir)
    files = {"exemplar-bank.json": cm.file_bytes(result["exemplar_bank"]), "rule-bank.json": cm.file_bytes(result["rule_bank"])}
    receipt = copy.deepcopy(dict(result["receipt_core"]))
    receipt["files"] = {name: {"sha256": cm.sha256_bytes(data), "bytes": len(data)} for name, data in files.items()}
    receipt["exemplar_bank"]["file"] = "exemplar-bank.json"
    receipt["rule_bank"]["file"] = "rule-bank.json"
    targets = {out / name: data for name, data in files.items()}
    targets[out / "bank-receipt.json"] = cm.file_bytes(receipt)
    for path, data in targets.items():
        if path.exists() and path.read_bytes() != data:
            raise DistillError(f"refusing to overwrite {path}: it exists with different bytes")
    for path, data in targets.items():
        cm.write_once(path, data)
    return receipt


def _estimate_only(name: str) -> tx.TokenMeasure:
    if name != tx.MEASURE_ESTIMATE:
        raise DistillError(f"the command line supports only the offline estimate measure; the {name!r} measure needs an injected tokenizer client (module API)")
    return tx.EstimateMeasure()


def union_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="tfrl_v2_distiller.py union", description="merge the two halves' banks into the Fold B union bank")
    parser.add_argument("--h1-receipt", required=True)
    parser.add_argument("--h2-receipt", required=True)
    parser.add_argument("--measure", default=tx.MEASURE_ESTIMATE)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(list(argv))
    try:
        result = union_banks(load_bank(Path(args.h1_receipt)), load_bank(Path(args.h2_receipt)), measure=_estimate_only(args.measure))
        receipt = write_union_outputs(result, Path(args.out_dir))
    except (DistillError, cm.CommonError, tx.ItemTextError, tx.MeasureError, OSError, KeyError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    print(json.dumps({k: receipt[k] for k in ("kind", "source_halves", "measure", "exemplar_bank", "rule_bank", "counts")}, indent=1, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "union":
        return union_main(arguments[1:])
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    parser.add_argument("--half", choices=cm.HALVES, required=True, help="the SOURCE half whose trajectories are distilled")
    parser.add_argument("--roster", help="source half roster receipt (default: the committed path)")
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--trace", nargs="+", required=True, help="rollout-trace JSONL file(s) of the half's construction flight")
    parser.add_argument("--k", type=int, default=cm.DEFAULT_K)
    parser.add_argument("--scorer-id")
    parser.add_argument("--no-twin", action="store_true")
    parser.add_argument("--allow-missing-rows", action="store_true")
    parser.add_argument("--measure", default=tx.MEASURE_ESTIMATE, help="only 'estimate' on the command line (see the module API for vllm-tokenize)")
    parser.add_argument("--data-root", help="resolve the supplied-item text from the registered knowledge stores under this data root (default: ids only)")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(arguments)
    try:
        roster_path = Path(args.roster) if args.roster else REPO_ROOT / cm.HALF_ROSTER_RELPATH[args.half]
        raw = roster_path.read_bytes()
        document = json.loads(raw.decode("utf-8"))
        paths = [Path(p) for p in args.trace]
        resolver = tx.ItemTextResolver.from_pins(Path(args.data_root)) if args.data_root else None
        contacts, scores, _ = sp.read_trace(paths)
        result = distil(document, args.half, args.iteration, contacts, scores, k=args.k, scorer_id=args.scorer_id, with_twin=not args.no_twin,
                        allow_missing_rows=args.allow_missing_rows, roster_file_sha256=cm.sha256_bytes(raw),
                        trace_sha256s=[cm.sha256_file(p) for p in paths], measure=_estimate_only(args.measure), resolver=resolver)
        receipt = write_outputs(result, Path(args.out_dir))
    except (DistillError, sp.SamplerError, cm.CommonError, tx.ItemTextError, tx.MeasureError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    print(json.dumps({k: receipt[k] for k in ("source_half", "consuming_half", "exemplar_bank", "rule_bank", "counts", "credit_summary")}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
