#!/usr/bin/env python3
"""slurp_l338_port_column_read.py -- the READ of the NPU-port twin of L338 (the lever column on the BF16 / vLLM-Ascend Thinking core, TRAIN Fold B).

Port mode of ``slurp_l338_column_read.py``. That reader binds the historical cell ids and the historical analysis root (``l338_family``); this one takes the cell ids from
``l338_port_family`` (the port run and attempt ids, the port READ run id, the port analysis root) and the reading rules from the port recipe
(``docs/readiness/2026-10-28-slurp-l338-lever-column-fold-b-npu-port-v1/analysis-recipe-v1.json``, ``port_extensions``), never from an edited copy of the historical arm table. Every scoring
function is the historical reader's, imported unchanged (``read_stage``, ``smoke_stage``, ``arm_words``, ``npc.paired_contrasts``: arm minus lock, paired per row,
utterance-cluster bootstrap, 10,000 draws, seed 20261018).

Stages (both write-once):

  smoke  the four-row smoke decode-cost table, ``analysis/<port READ id>/smoke-decode-cost.json`` (schema ``saea-l338-lever-column-smoke-v1``), the input of
         ``scripts/flights/batch/smoke_signoff.py`` and of the port package's ``signoff-conditions-v1.json``. Nothing is scored.
  read   the historical read (``read.json`` and ``README.md``, same format as the historical read) plus ``port-verdicts.json``: the port's three cross-run verdicts.
         Run ``stage3_rollout_score.py --mode perception-branches`` first (``step_score.sh``) so the per-arm column score files exist.

Port verdicts (every number carries ``source`` and ``locator`` naming the input file in ``inputs`` and the place in it; ``inputs`` lists path and sha256 of each)

  PARITY       per arm in ``port_extensions.parity_band.applies_to`` (lock, aa): official pooled SLU-F1 in [71.61, 72.75] AND intent accuracy in [84.91, 85.74], both
               inclusive, both read on the scorer's two-decimal percent; otherwise OUTSIDE-BAND with the metric, the side (below / above) and the distance in pp.
               S0 is PARITY only when lock AND aa are PARITY AND the run is ADMISSIBLE.
  ADMISSIBLE   |SLU-F1(aa) - SLU-F1(lock)| <= 1.20 pp (the point delta of the two official column scorings, from the paired contrast, rounded to 6 decimals before the
               comparison so that a delta of exactly 1.20 is not lost to float error); otherwise NOT-ADMISSIBLE. A run without the twin arm is NOT-ADMISSIBLE.
  IMPROVEMENT  per lever arm (l1-support, l2-union): IMPROVEMENT-ADMISSIBLE only when ALL of (1) the run is ADMISSIBLE, (2) the arm's SLU-F1 (two-decimal) is >= band top
               + cross-run bar (72.75 + 2.5 = 75.25), (3) the paired SLU delta against the port lock has CI95 lower bound > 0 (strict), (4) aa is PARITY (inside the
               band). Otherwise NO-CROSS-RUN-STATEMENT (the recipe: "no cross-run improvement statement is made; the within-run words stand on their own"). The within-run
               word (NOMINATED / HARM / NEUTRAL, ``arm_words``) is reported beside it and is independent of it.

Model-free. Every sealed byte is opened read-only; no utterance text, gold label or model output is printed or persisted.

    python -B scripts/analysis/slurp_l338_port_column_read.py --stage smoke --data-root $SPEECHRL_DATA_DIR
    python -B scripts/analysis/slurp_l338_port_column_read.py --stage read  --data-root $SPEECHRL_DATA_DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _path in (str(_REPO / "src"), str(_HERE), str(_REPO / "scripts" / "flights" / "nx")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import l338_family as fam  # noqa: E402
import l338_port_family as pf  # noqa: E402
import slurp_l338_column_read as hist  # noqa: E402  (the historical reader: every scoring function is used unchanged)

base, npc = hist.base, hist.npc
ReadError = hist.ReadError
VERDICTS_SCHEMA = "saea-l338-port-verdicts-v1"
RECIPE_SCHEMA = "saea-stage3-rollout-analysis-recipe-v1"
WORD_PARITY, WORD_OUTSIDE = "PARITY", "OUTSIDE-BAND"
WORD_ADMISSIBLE, WORD_NOT_ADMISSIBLE = "ADMISSIBLE", "NOT-ADMISSIBLE"
WORD_IMPROVEMENT, WORD_NO_STATEMENT = "IMPROVEMENT-ADMISSIBLE", "NO-CROSS-RUN-STATEMENT"
METRICS = (("slu_f1", "slu_f1_pct", "SLU-F1"), ("intent", "intent_pct", "intent accuracy"))
ADMISSIBILITY_DECIMALS = 6


def two_dp(value: float) -> float:
    """The scorer's two-decimal percent, formatted exactly as the reader's table prints it (``{:.2f}``)."""

    return float(f"{value:.2f}")


def cite(value: Any, source: str, locator: str) -> dict:
    """A number with the input file it came from (``source`` is a key of the verdict file's ``inputs``) and the place in that file (or the computation over it)."""

    return {"value": value, "source": source, "locator": locator}


# --------------------------------------------------------------------------- the rules, read from the port recipe


def load_recipe(path: Path) -> dict:
    """The port recipe, refused by name unless it binds the port ids and the historical scoring constants."""

    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    problems: list[str] = []
    if recipe.get("schema") != RECIPE_SCHEMA:
        problems.append(f"schema {recipe.get('schema')!r} != {RECIPE_SCHEMA!r}")
    ids = recipe.get("run_ids") or {}
    for key, expected in (("read", pf.READ_RUN_ID), ("arm", pf.CELLS["arm"]["run"]), ("smoke", pf.CELLS["smoke"]["run"])):
        if ids.get(key) != expected:
            problems.append(f"run_ids.{key} {ids.get(key)!r} != {expected!r}")
    if recipe.get("read_run_id") != pf.READ_RUN_ID:
        problems.append(f"read_run_id {recipe.get('read_run_id')!r} != {pf.READ_RUN_ID!r}")
    if recipe.get("delivering") != fam.DELIVERING or recipe.get("twin") != fam.TWIN:
        problems.append("delivering / twin are not lock / aa")
    if sorted(recipe.get("arms") or []) != sorted(fam.ARM_ORDER):
        problems.append(f"arms {recipe.get('arms')!r} != {sorted(fam.ARM_ORDER)!r}")
    primary = recipe.get("primary") or {}
    if primary.get("draws") != hist.DRAWS or primary.get("seed") != hist.SEED:
        problems.append(f"primary draws/seed {primary.get('draws')!r}/{primary.get('seed')!r} != {hist.DRAWS}/{hist.SEED}")
    if (recipe.get("guard") or {}).get("floors_pp") != {"intent": fam.FLOOR_INTENT_PP, "slu": fam.FLOOR_SLU_PP}:
        problems.append("guard.floors_pp is not the historical floors")
    if not isinstance(recipe.get("port_extensions"), Mapping):
        problems.append("no port_extensions object")
    if problems:
        raise ReadError(f"the port recipe {path} is refused: " + "; ".join(problems))
    return recipe


def rules_from_recipe(recipe: Mapping) -> dict:
    """The numeric rules of the port verdicts with the locator of each in the recipe (``source`` ``recipe``)."""

    ext = recipe["port_extensions"]
    band, cmp0, bar = ext["parity_band"], ext["registered_comparisons"][0], ext["cross_run_bar"]
    if cmp0.get("arm") != fam.TWIN or cmp0.get("against") != fam.DELIVERING or band.get("applies_to") != [fam.DELIVERING, fam.TWIN]:
        raise ReadError("the recipe's registered comparison / band do not bind aa against lock and the band to lock and aa")
    if bar.get("twin_arm") != fam.TWIN or not bar.get("requires_twin_arm_in_the_same_run"):
        raise ReadError("the recipe's cross-run bar does not require the twin aa in the same run")
    floor = round(band["slu_f1"]["high"] + bar["bar_slu_pp"], 2)
    p = "port_extensions"
    return {
        "band": {"slu_f1": (band["slu_f1"]["low"], band["slu_f1"]["high"]), "intent": (band["intent"]["low"], band["intent"]["high"])},
        "applies_to": list(band["applies_to"]), "twin_threshold_slu_pp": cmp0["threshold_slu_pp"], "bar_slu_pp": bar["bar_slu_pp"], "improvement_floor_slu": floor,
        "cites": {
            "band.slu_f1.low": cite(band["slu_f1"]["low"], "recipe", f"{p}.parity_band.slu_f1.low"),
            "band.slu_f1.high": cite(band["slu_f1"]["high"], "recipe", f"{p}.parity_band.slu_f1.high"),
            "band.intent.low": cite(band["intent"]["low"], "recipe", f"{p}.parity_band.intent.low"),
            "band.intent.high": cite(band["intent"]["high"], "recipe", f"{p}.parity_band.intent.high"),
            "twin_threshold_slu_pp": cite(cmp0["threshold_slu_pp"], "recipe", f"{p}.registered_comparisons[0].threshold_slu_pp"),
            "bar_slu_pp": cite(bar["bar_slu_pp"], "recipe", f"{p}.cross_run_bar.bar_slu_pp"),
            "improvement_floor_slu": cite(floor, "recipe", f"{p}.parity_band.slu_f1.high + {p}.cross_run_bar.bar_slu_pp (rounded to 2 decimals)")}}


# --------------------------------------------------------------------------- the verdicts (pure functions over the paired contrast)


def parity_of_arm(arm: str, levels: Mapping, rules: Mapping) -> dict:
    """PARITY / OUTSIDE-BAND of one arm: both metrics inside their band, inclusive, on the two-decimal reading."""

    out: dict[str, Any] = {}
    outside: list[dict] = []
    for metric, key, label in METRICS:
        raw = levels[arm][key]
        value = two_dp(raw)
        low, high = rules["band"][metric]
        side = "below" if value < low else "above" if value > high else "inside"
        distance = round(low - value, 2) if side == "below" else round(value - high, 2) if side == "above" else 0.0
        out[metric] = {"label": label, "value_pct": cite(raw, "read_json", f"contrast.levels.{arm}.{key}"), "value_pct_2dp": value,
                       "band": [rules["cites"][f"band.{metric}.low"], rules["cites"][f"band.{metric}.high"]], "side": side, "distance_pp": distance}
        if side != "inside":
            outside.append({"metric": metric, "side": side, "distance_pp": distance})
    return {"arm": arm, "word": WORD_PARITY if not outside else WORD_OUTSIDE, "outside": outside, **out}


def admissibility(contrast: Mapping, rules: Mapping) -> dict:
    """ADMISSIBLE when |SLU-F1(aa) - SLU-F1(lock)| <= the registered threshold; a run without the twin supports no cross-run statement."""

    twin = (contrast.get("versus_delivering") or {}).get(fam.TWIN)
    threshold = rules["twin_threshold_slu_pp"]
    if twin is None:
        return {"word": WORD_NOT_ADMISSIBLE, "reason": "cross-flight-no-replicate: the run has no twin arm aa", "threshold_slu_pp": rules["cites"]["twin_threshold_slu_pp"]}
    point = twin["slu_f1"]["point_pp"]
    magnitude = round(abs(point), ADMISSIBILITY_DECIMALS)
    levels = contrast["levels"]
    return {"word": WORD_ADMISSIBLE if magnitude <= threshold else WORD_NOT_ADMISSIBLE, "rule": f"|aa - lock| <= {threshold} pp on official pooled SLU-F1 (point delta)",
            "aa_minus_lock_slu_pp": cite(point, "read_json", "contrast.versus_delivering.aa.slu_f1.point_pp"), "abs_rounded_6dp": magnitude,
            "aa_minus_lock_slu_pp_from_2dp_readings": round(two_dp(levels[fam.TWIN]["slu_f1_pct"]) - two_dp(levels[fam.DELIVERING]["slu_f1_pct"]), 2),
            "aa_minus_lock_intent_pp": cite(twin["intent"]["point_pp"], "read_json", "contrast.versus_delivering.aa.intent.point_pp"),
            "threshold_slu_pp": rules["cites"]["twin_threshold_slu_pp"]}


def improvement_of_arm(arm: str, contrast: Mapping, rules: Mapping, parity: Mapping, admissible: Mapping, within_run_word: str | None) -> dict:
    """IMPROVEMENT-ADMISSIBLE / NO-CROSS-RUN-STATEMENT for one lever arm (see the module docstring for the four conditions)."""

    level = contrast["levels"][arm]["slu_f1_pct"]
    delta = contrast["versus_delivering"][arm]["slu_f1"]
    level2 = two_dp(level)
    floor = rules["improvement_floor_slu"]
    conditions = {
        "run_admissible": admissible["word"] == WORD_ADMISSIBLE,
        "slu_at_or_above_improvement_floor": level2 >= floor,
        "slu_delta_ci95_lower_above_zero": delta["ci95_lo_pp"] > 0,
        "twin_aa_inside_band": parity[fam.TWIN]["word"] == WORD_PARITY,
    }
    return {
        "arm": arm, "word": WORD_IMPROVEMENT if all(conditions.values()) else WORD_NO_STATEMENT, "conditions": conditions,
        "slu_f1_pct": cite(level, "read_json", f"contrast.levels.{arm}.slu_f1_pct"), "slu_f1_pct_2dp": level2,
        "improvement_floor_slu": rules["cites"]["improvement_floor_slu"],
        "slu_delta_vs_lock_pp": cite(delta["point_pp"], "read_json", f"contrast.versus_delivering.{arm}.slu_f1.point_pp"),
        "slu_delta_ci95_pp": [cite(delta["ci95_lo_pp"], "read_json", f"contrast.versus_delivering.{arm}.slu_f1.ci95_lo_pp"),
                              cite(delta["ci95_hi_pp"], "read_json", f"contrast.versus_delivering.{arm}.slu_f1.ci95_hi_pp")],
        "intent_pct": cite(contrast["levels"][arm]["intent_pct"], "read_json", f"contrast.levels.{arm}.intent_pct"),
        "intent_delta_vs_lock_pp": cite(contrast["versus_delivering"][arm]["intent"]["point_pp"], "read_json", f"contrast.versus_delivering.{arm}.intent.point_pp"),
        "intent_delta_ci95_pp": [cite(contrast["versus_delivering"][arm]["intent"]["ci95_lo_pp"], "read_json", f"contrast.versus_delivering.{arm}.intent.ci95_lo_pp"),
                                 cite(contrast["versus_delivering"][arm]["intent"]["ci95_hi_pp"], "read_json", f"contrast.versus_delivering.{arm}.intent.ci95_hi_pp")],
        "within_run_word": cite(within_run_word, "read_json", f"arm_words.words.{arm}.word")}


def port_verdicts(contrast: Mapping, arm_words: Mapping, rules: Mapping, *, inputs: Mapping, interval: Mapping, recipe_status: str | None = None) -> dict:
    """The ``port-verdicts.json`` document from a paired-contrast record (``npc.paired_contrasts``), the historical ``arm_words`` and the recipe rules."""

    levels = contrast["levels"]
    missing = [arm for arm in rules["applies_to"] if arm not in levels]
    if missing:
        raise ReadError(f"the parity arms {missing} have no column in the contrast")
    parity = {arm: parity_of_arm(arm, levels, rules) for arm in rules["applies_to"]}
    admissible = admissibility(contrast, rules)
    words = (arm_words or {}).get("words") or {}
    improvement = {arm: improvement_of_arm(arm, contrast, rules, parity, admissible, (words.get(arm) or {}).get("word"))
                   for arm in hist.CANDIDATES if arm in contrast["versus_delivering"]}
    reasons = [f"{arm} is {parity[arm]['word']}" for arm in rules["applies_to"] if parity[arm]["word"] != WORD_PARITY]
    if admissible["word"] != WORD_ADMISSIBLE:
        reasons.append(f"the run is {admissible['word']}")
    s0 = WORD_PARITY if not reasons else WORD_OUTSIDE
    return {
        "schema": VERDICTS_SCHEMA, "status": "PORT-VERDICTS (development numbers; claims_benefit false; a reading, never an admission)",
        "read_run_id": pf.READ_RUN_ID, "run_id": pf.CELLS["arm"]["run"], "recipe_status": recipe_status, "inputs": dict(inputs), "interval": dict(interval),
        "rules": {k: v for k, v in rules["cites"].items()},
        "rule_text": {
            "parity": "arm's SLU-F1 in [band.slu_f1.low, band.slu_f1.high] AND intent in [band.intent.low, band.intent.high], inclusive, two-decimal percent",
            "admissible": "|SLU-F1(aa) - SLU-F1(lock)| <= twin_threshold_slu_pp (point delta; rounded to 6 decimals before the comparison)",
            "improvement": ("IMPROVEMENT-ADMISSIBLE iff run_admissible AND SLU-F1 (2dp) >= improvement_floor_slu AND SLU delta vs lock CI95 lower bound > 0 AND aa PARITY; "
                            "otherwise NO-CROSS-RUN-STATEMENT")},
        "parity": parity, "twin_admissibility": admissible, "improvement": improvement,
        "summary": {"s0": {"word": s0, "reasons": reasons, "needs": "lock PARITY and aa PARITY and the run ADMISSIBLE"},
                    "s2_may_start": parity[fam.DELIVERING]["word"] == WORD_PARITY,
                    "s2_basis": "recipe port_extensions.parity_band.consequence: OUTSIDE-BAND on the port lock arm -> S2 does not start (a reading; the owner decides)",
                    "improvement": {arm: v["word"] for arm, v in improvement.items()}}}


# --------------------------------------------------------------------------- the sealed-run adapter and the CLI


def _input(path: Path, root: Path | None = None) -> dict:
    p = Path(path)
    return {"path": str(p.relative_to(root)) if root and p.is_relative_to(root) else str(p), "sha256": base.sha256_file(p)}


def _smoke(args: argparse.Namespace, root: Path) -> tuple[dict, Path]:
    cell = pf.CELLS["smoke"]
    paths = base.run_paths(args.data_root, cell["run"], cell["attempt"])
    for name in ("trace", "outputs"):
        if not paths[name].exists():
            raise ReadError(f"missing sealed file {paths[name]}")
    outputs_rows = hist.load_outputs_rows(paths["outputs"])
    contacts = npc.trace_contacts(paths["trace"], fam.THINKING_KNOBS["budget_message"])
    by_arm = npc.attribute(contacts, hist.arm_contact_map(outputs_rows))
    result = hist.smoke_stage(by_arm, missing=hist.missing_columns(outputs_rows, contacts))
    result["sealed_hashes"] = {n: base.sha256_file(paths[n]) for n in ("trace", "outputs")}
    result["port"] = {"read_run_id": pf.READ_RUN_ID, "run_id": cell["run"], "attempt_id": cell["attempt"],
                      "note": "slot-normalised GPU hours in `throughput` use the historical 4 slots; the port card-hour bound is PROVISIONAL (l338_port_family.CARD_HOUR_FORMULA)"}
    result["throughput"]["registered_gpu_hours_ceiling"] = None
    return result, root / "smoke-decode-cost.json"


def _read(args: argparse.Namespace, root: Path) -> tuple[dict, Path, dict, Path]:
    recipe_path = args.recipe if args.recipe.is_absolute() else args.repo_root / args.recipe
    recipe = load_recipe(recipe_path)
    rules = rules_from_recipe(recipe)
    cell = pf.CELLS["arm"]
    paths = base.run_paths(args.data_root, cell["run"], cell["attempt"])
    for name in ("trace", "outputs", "manifest"):
        if not paths[name].exists():
            raise ReadError(f"missing sealed file {paths[name]}")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if str(manifest.get("status", "")).upper() != "COMPLETED":
        raise ReadError(f"manifest status is {manifest.get('status')!r}, not COMPLETED")
    arms = hist.load_column_rows(root / "columns")
    roster_path = args.repo_root / fam.ROSTER[cell["roster"]]
    ids, cluster_of, scenario_of = base.roster_maps(roster_path)
    for arm, rows in arms.items():
        if set(rows) != set(ids):
            raise ReadError(f"{arm}: the per-sample id set is not the frozen Fold B roster's")
    outputs_rows = hist.load_outputs_rows(paths["outputs"])
    contacts = npc.trace_contacts(paths["trace"], fam.THINKING_KNOBS["budget_message"])
    by_arm = npc.attribute(contacts, hist.arm_contact_map(outputs_rows))
    wanted = {request for row in outputs_rows for request in row["requests"].values()}
    streams = hist.solver_stream_keys(outputs_rows, hist.trace_response_keys(paths["trace"], wanted))
    sampling_path = root / "columns" / "sampling-rows.json"
    sampling_rows = json.loads(sampling_path.read_text(encoding="utf-8"))["rows"] if sampling_path.exists() else None
    gold, n1 = hist._load_json(args.gold_intents), hist._load_json(args.n1_json)
    result = hist.read_stage(arms, ids, cluster_of, scenario_of, by_arm, hist.contact_rows(outputs_rows, contacts), seed=args.seed, draws=args.draws,
                             missing=hist.missing_columns(outputs_rows, contacts), stream_disc=hist.stream_discordance(streams, ids),
                             perception_disc=hist.stream_discordance(hist.perception_stream_keys(outputs_rows), ids), sampling_rows=sampling_rows, gold=gold,
                             outputs_rows=outputs_rows, forms=hist.l1_forms_from_trace(outputs_rows, paths["trace"], n1))
    result["sealed_hashes"] = {n: base.sha256_file(paths[n]) for n in ("trace", "outputs", "manifest")}
    result["throughput"]["registered_gpu_hours_ceiling"] = None
    result["port"] = {"read_run_id": pf.READ_RUN_ID, "run_id": cell["run"], "attempt_id": cell["attempt"], "recipe": str(args.recipe),
                      "note": "slot-normalised GPU hours in `throughput` use the historical 4 slots; the port card-hour bound is PROVISIONAL (l338_port_family.CARD_HOUR_FORMULA)"}
    inputs = {"recipe": _input(recipe_path, args.repo_root), "roster": _input(roster_path, args.repo_root), "manifest": _input(paths["manifest"], args.data_root),
              "trace": _input(paths["trace"], args.data_root), "outputs": _input(paths["outputs"], args.data_root)}
    for arm in hist.ARMS:
        inputs[f"column_scores:{arm}"] = _input(root / "columns" / hist.COLUMN_SCORE_NAME.format(arm=arm, scorer=hist.SCORER), args.data_root)
    meta = {"recipe": recipe, "rules": rules, "inputs": inputs}
    return result, root / "read.json", meta, root / pf.VERDICTS_NAME


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "read"), required=True)
    parser.add_argument("--data-root", type=Path, default=Path(hist._data_root_env()))
    parser.add_argument("--repo-root", type=Path, default=_REPO)
    parser.add_argument("--analysis-root", type=Path, default=None, help=f"default <data root>/{pf.ANALYSIS_ROOT}")
    parser.add_argument("--recipe", type=Path, default=Path(pf.RECIPE_RELPATH), help="the port analysis recipe (repo-relative or absolute)")
    parser.add_argument("--seed", type=int, default=hist.SEED)
    parser.add_argument("--draws", type=int, default=hist.DRAWS)
    parser.add_argument("--gold-intents", type=Path, default=None, help="optional JSON {sample_id: gold intent id} for the L2 recall tables")
    parser.add_argument("--n1-json", type=Path, default=None, help="optional JSON {sample_id: N1} for the N1-gated L1 forms")
    args = parser.parse_args(argv)
    root = args.analysis_root or (args.data_root / pf.ANALYSIS_ROOT)
    try:
        if args.stage == "smoke":
            outs = [root / "smoke-decode-cost.json"]
        else:
            outs = [root / "read.json", root / pf.VERDICTS_NAME]
        for out in outs:
            if out.exists():
                print(f"REFUSED: {out} exists (write-once)", file=sys.stderr)
                return 6
        root.mkdir(parents=True, exist_ok=True)
        if args.stage == "smoke":
            result, out = _smoke(args, root)
            out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            out.with_name("smoke-decode-cost.md").write_text(hist.render_markdown({"arms_nodes": result["arms"], "throughput": result.get("throughput")}), encoding="utf-8")
            print(json.dumps({"written": str(out)}))
            return 0
        result, out, meta, verdict_out = _read(args, root)
        read_text = json.dumps(result, indent=2, sort_keys=True) + "\n"
        inputs = dict(meta["inputs"], read_json={"path": str(out), "sha256": hashlib.sha256(read_text.encode("utf-8")).hexdigest()})
        interval = dict(result["interval"], registered=bool(result["interval"]["draws"] == hist.DRAWS and result["interval"]["seed"] == hist.SEED))
        verdicts = port_verdicts(result["contrast"], result["arm_words"], meta["rules"], inputs=inputs, interval=interval,
                                 recipe_status=(meta["recipe"].get("port_extensions") or {}).get("status"))
        out.write_text(read_text, encoding="utf-8")
        out.with_name("README.md").write_text(hist.render_markdown(result), encoding="utf-8")
        verdict_out.write_text(json.dumps(verdicts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"written": [str(out), str(verdict_out)], "arm_words": {a: w["word"] for a, w in result["arm_words"]["words"].items()},
                          "parity": {a: v["word"] for a, v in verdicts["parity"].items()}, "twin": verdicts["twin_admissibility"]["word"],
                          "improvement": verdicts["summary"]["improvement"]}))
        return 0
    except ReadError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
