"""The NPU-port READ: ``scripts/analysis/slurp_l338_port_column_read.py`` (port mode of the L338 column reader).

Model-free, synthetic per-arm result fixtures. The proofs: (1) the ids, the analysis root and the sign-off table path come from ``l338_port_family`` and the port
recipe, not from the historical family; (2) PARITY / OUTSIDE-BAND per arm against the band, inclusive at the four edges, with side and distance; (3) the twin rule at
1.2 pp; (4) the improvement verdict (an improving arm against a neutral arm, and each of the four conditions failing alone); (5) the smoke table is accepted by
``smoke_signoff.py`` under the port package's committed ``signoff-conditions-v1.json``; (6) the read stage writes ``read.json`` / ``README.md`` /
``port-verdicts.json`` write-once, every number of the verdict file cites an input that exists and whose sha256 matches.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _extra in (REPO / "scripts" / "flights" / "nx", REPO / "scripts" / "flights" / "direct", REPO / "scripts" / "flights" / "tfrl", REPO / "scripts" / "flights" / "batch",
               REPO / "scripts" / "analysis", REPO / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import l338_family as fam  # noqa: E402
import l338_port_family as pf  # noqa: E402
import slurp_l338_column_read as hist  # noqa: E402
import slurp_l338_port_column_read as port  # noqa: E402
import smoke_signoff  # noqa: E402
import test_l338_registration as fx  # noqa: E402  (the historical synthetic sealed-run fixtures)

PKG = REPO / pf.PACKAGE
RECIPE = REPO / pf.RECIPE_RELPATH
HIST_READ_ID = fam.READ_RUN_ID


# --------------------------------------------------------------------------- the PORT arm table fixtures (the historical rows plus the replicate lock-rep2)


def port_outputs_rows(n=4, lists=(2, 3)):
    """``fx._outputs_rows`` (the historical four-arm sealed rows) with the replicate branch after the twin: a copy of the twin's branch under the replicate's own ids."""

    rows = fx._outputs_rows(n, lists=lists)
    for row in rows:
        branches = []
        for branch in row["branches"]:
            branches.append(branch)
            if branch["branch_id"] == fam.TWIN:
                for arm in pf.REPLICATE_ARMS:
                    p, s = f"{row['sample_id']}:{arm}:p0", f"{row['sample_id']}:{arm}:s0"
                    row["requests"][s] = f"req-{s}"
                    branches.append(dict(branch, branch_id=arm, contact_id=p, solver_contact_id=s))
        row["branches"] = branches
        assert [b["branch_id"] for b in branches] == list(pf.ARM_ORDER)
    return rows


def port_arms_scores(ids, *, gain=0.0, intent=1.0):
    """``fx._arms_scores`` with the replicate column: an exact repeat of lock (a zero replicate delta)."""

    scores = fx._arms_scores(ids, gain=gain, intent=intent)
    for arm in pf.REPLICATE_ARMS:
        scores[arm] = dict(scores[fam.DELIVERING])
    return scores


# --------------------------------------------------------------------------- synthetic contrasts with controlled levels and intervals


def iv(point, lo=None, hi=None):
    return {"point_pp": point, "ci95_lo_pp": point - 1.0 if lo is None else lo, "ci95_hi_pp": point + 1.0 if hi is None else hi}


def contrast(lock=(72.0, 85.0), aa=(72.1, 85.1), arms=None, twin_point=None, *, with_twin=True):
    """A paired-contrast record: ``lock`` / ``aa`` are (SLU-F1 %, intent %); ``arms`` maps a lever arm to ``(slu %, lo, hi, intent %)`` (delta vs lock from the levels)."""

    levels = {"lock": {"slu_f1_pct": lock[0], "intent_pct": lock[1]}}
    versus = {}
    if with_twin:
        levels["aa"] = {"slu_f1_pct": aa[0], "intent_pct": aa[1]}
        point = aa[0] - lock[0] if twin_point is None else twin_point
        versus["aa"] = {"slu_f1": iv(point), "intent": iv(aa[1] - lock[1], aa[1] - lock[1] - 0.5, aa[1] - lock[1] + 0.5)}
    for arm, (slu, lo, hi, intent) in (arms or {}).items():
        levels[arm] = {"slu_f1_pct": slu, "intent_pct": intent}
        versus[arm] = {"slu_f1": iv(slu - lock[0], lo, hi), "intent": iv(intent - lock[1], intent - lock[1] - 0.5, intent - lock[1] + 0.5)}
    return {"n": 1080, "delivering": "lock", "levels": levels, "versus_delivering": versus}


@pytest.fixture(scope="module")
def rules():
    return port.rules_from_recipe(port.load_recipe(RECIPE))


def verdicts(rules, c, words=None):
    return port.port_verdicts(c, {"words": {a: {"word": w} for a, w in (words or {}).items()}}, rules, inputs={}, interval={"draws": 10000, "seed": 20261018})


# --------------------------------------------------------------------------- (1) ids and rules come from the port family and the port recipe


def test_the_reader_binds_the_port_ids_not_the_historical_ones(rules):
    assert pf.READ_RUN_ID == "SAEA-SLURP-L338-LEVER-COLUMN-FOLD-B-READ-NPU-PORT-V1" and pf.READ_RUN_ID != HIST_READ_ID
    assert pf.ANALYSIS_ROOT == f"analysis/{pf.READ_RUN_ID}" and pf.SMOKE_TABLE_RELPATH == f"{pf.ANALYSIS_ROOT}/smoke-decode-cost.json"
    assert pf.CELLS["arm"]["run"] == "SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-NPU-PORT-V1" and pf.CELLS["arm"]["run"] != fam.CELLS["arm"]["run"]
    source = (REPO / pf.PORT_READER).read_text(encoding="utf-8")
    assert "l338_arms" not in source and "fam.CELLS" not in source and "fam.ANALYSIS_ROOT" not in source and "fam.READ_RUN_ID" not in source
    assert port.hist is hist and port.WORD_PARITY == "PARITY"
    assert rules["band"] == {"slu_f1": (71.61, 72.75), "intent": (84.91, 85.74)} and rules["twin_threshold_slu_pp"] == 1.2
    assert rules["bar_slu_pp"] == 2.5 and rules["improvement_floor_slu"] == 75.25 and rules["applies_to"] == ["lock", "aa"]


def test_the_recipe_is_refused_by_name_when_it_binds_other_ids(tmp_path):
    doc = json.loads(RECIPE.read_text(encoding="utf-8"))
    doc["run_ids"]["read"] = HIST_READ_ID
    bad = tmp_path / "recipe.json"
    bad.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(port.ReadError, match="run_ids.read"):
        port.load_recipe(bad)
    doc = json.loads(RECIPE.read_text(encoding="utf-8"))
    del doc["port_extensions"]
    bad.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(port.ReadError, match="port_extensions"):
        port.load_recipe(bad)


# --------------------------------------------------------------------------- (2) parity


@pytest.mark.parametrize("slu,intent", [(71.61, 84.91), (72.75, 85.74), (72.0, 85.0), (71.61, 85.74), (72.75, 84.91)])
def test_parity_inside_the_band_edges_are_inclusive(rules, slu, intent):
    v = verdicts(rules, contrast(lock=(slu, intent), aa=(slu, intent)))
    assert v["parity"]["lock"]["word"] == "PARITY" and v["parity"]["lock"]["outside"] == []
    assert v["summary"]["s0"]["word"] == "PARITY" and v["summary"]["s2_may_start"] is True


@pytest.mark.parametrize("slu,intent,metric,side,distance", [
    (71.60, 85.0, "slu_f1", "below", 0.01), (72.76, 85.0, "slu_f1", "above", 0.01), (72.0, 84.90, "intent", "below", 0.01), (72.0, 85.75, "intent", "above", 0.01),
    (70.0, 85.0, "slu_f1", "below", 1.61), (72.0, 87.0, "intent", "above", 1.26)])
def test_parity_outside_the_band_reports_metric_side_and_distance(rules, slu, intent, metric, side, distance):
    v = verdicts(rules, contrast(lock=(slu, intent), aa=(72.0, 85.0)))
    lock = v["parity"]["lock"]
    assert lock["word"] == "OUTSIDE-BAND" and lock["outside"] == [{"metric": metric, "side": side, "distance_pp": distance}]
    assert v["summary"]["s0"]["word"] == "OUTSIDE-BAND" and v["summary"]["s2_may_start"] is False and "lock is OUTSIDE-BAND" in v["summary"]["s0"]["reasons"]
    assert v["parity"]["aa"]["word"] == "PARITY"


def test_parity_reads_the_two_decimal_percent_and_reports_both_metrics_when_both_are_out(rules):
    v = verdicts(rules, contrast(lock=(72.754, 85.7449), aa=(72.7551, 85.0)))
    assert v["parity"]["lock"]["word"] == "PARITY" and v["parity"]["lock"]["slu_f1"]["value_pct_2dp"] == 72.75         # 72.754 reads as 72.75: inside
    assert v["parity"]["aa"]["word"] == "OUTSIDE-BAND" and v["parity"]["aa"]["slu_f1"]["value_pct_2dp"] == 72.76 and v["parity"]["aa"]["slu_f1"]["side"] == "above"
    both = verdicts(rules, contrast(lock=(70.0, 80.0), aa=(72.0, 85.0)))["parity"]["lock"]
    assert [o["metric"] for o in both["outside"]] == ["slu_f1", "intent"] and both["outside"][1]["distance_pp"] == 4.91


def test_aa_outside_the_band_blocks_s0_even_when_lock_is_inside(rules):
    v = verdicts(rules, contrast(lock=(72.0, 85.0), aa=(72.9, 85.0)))                 # |aa - lock| = 0.9 pp: admissible, but aa is above the band
    assert v["twin_admissibility"]["word"] == "ADMISSIBLE" and v["parity"]["aa"]["word"] == "OUTSIDE-BAND"
    assert v["summary"]["s0"]["word"] == "OUTSIDE-BAND" and v["summary"]["s0"]["reasons"] == ["aa is OUTSIDE-BAND"] and v["summary"]["s2_may_start"] is True


# --------------------------------------------------------------------------- (3) the twin rule at 1.2 pp


@pytest.mark.parametrize("point,word", [(0.0, "ADMISSIBLE"), (1.19, "ADMISSIBLE"), (1.2, "ADMISSIBLE"), (-1.2, "ADMISSIBLE"), (1.2000000000000028, "ADMISSIBLE"),
                                        (1.21, "NOT-ADMISSIBLE"), (-1.21, "NOT-ADMISSIBLE"), (3.0, "NOT-ADMISSIBLE")])
def test_twin_rule_passes_at_1_2_and_fails_above(rules, point, word):
    c = contrast(lock=(72.0, 85.0), aa=(72.0, 85.0), twin_point=point)
    out = port.admissibility(c, rules)
    assert out["word"] == word and out["aa_minus_lock_slu_pp"]["source"] == "read_json" and out["threshold_slu_pp"]["value"] == 1.2


def test_the_historical_twin_offsets_pass_and_a_wide_pair_fails(rules):
    for offset in (1.15, -0.12, 0.45):                          # the three offsets the recipe registers as passing
        assert port.admissibility(contrast(aa=(72.0, 85.0), twin_point=offset), rules)["word"] == "ADMISSIBLE"
    wide = verdicts(rules, contrast(lock=(71.7, 85.0), aa=(72.9, 85.0)))              # both inside the band, 1.2 apart only by rounding of the band
    assert wide["twin_admissibility"]["word"] == "ADMISSIBLE"
    far = verdicts(rules, contrast(lock=(71.0, 85.0), aa=(72.5, 85.0)))
    assert far["twin_admissibility"]["word"] == "NOT-ADMISSIBLE" and far["summary"]["s0"]["word"] == "OUTSIDE-BAND"
    assert "the run is NOT-ADMISSIBLE" in far["summary"]["s0"]["reasons"]


def test_a_run_without_the_twin_is_not_admissible(rules):
    c = contrast(with_twin=False, arms={"l1-support": (76.0, 0.5, 2.0, 85.0)})
    out = port.admissibility(c, rules)
    assert out["word"] == "NOT-ADMISSIBLE" and "cross-flight-no-replicate" in out["reason"]
    with pytest.raises(port.ReadError, match="parity arms"):
        verdicts(rules, c)                                       # aa has no column: the verdict file refuses rather than guess


# --------------------------------------------------------------------------- (4) improvement and the verdict words


def test_an_improving_arm_is_admissible_and_a_neutral_arm_is_not(rules):
    c = contrast(lock=(72.0, 85.0), aa=(72.3, 85.2), arms={"l1-support": (75.6, 1.6, 4.9, 85.1), "l2-union": (72.4, -1.0, 1.8, 86.0)})
    v = verdicts(rules, c, {"l1-support": "NOMINATED", "l2-union": "NEUTRAL"})
    good, neutral = v["improvement"]["l1-support"], v["improvement"]["l2-union"]
    assert good["word"] == "IMPROVEMENT-ADMISSIBLE" and all(good["conditions"].values()) and good["within_run_word"]["value"] == "NOMINATED"
    assert neutral["word"] == "NO-CROSS-RUN-STATEMENT" and neutral["conditions"]["slu_at_or_above_improvement_floor"] is False
    assert neutral["conditions"]["slu_delta_ci95_lower_above_zero"] is False and neutral["within_run_word"]["value"] == "NEUTRAL"
    assert v["summary"]["improvement"] == {"l1-support": "IMPROVEMENT-ADMISSIBLE", "l2-union": "NO-CROSS-RUN-STATEMENT"}
    assert good["slu_delta_ci95_pp"][0] == {"value": 1.6, "source": "read_json", "locator": "contrast.versus_delivering.l1-support.slu_f1.ci95_lo_pp"}


@pytest.mark.parametrize("slu,lo,aa,twin_point,failing", [
    (75.25, 0.1, (72.3, 85.2), None, None),                      # exactly at the floor: admissible
    (75.24, 0.1, (72.3, 85.2), None, "slu_at_or_above_improvement_floor"),
    (76.0, 0.0, (72.3, 85.2), None, "slu_delta_ci95_lower_above_zero"),          # lower bound exactly zero: not above zero
    (76.0, -0.2, (72.3, 85.2), None, "slu_delta_ci95_lower_above_zero"),
    (76.0, 0.5, (72.9, 87.0), None, "twin_aa_inside_band"),                      # aa intent above the band
    (76.0, 0.5, (72.3, 85.2), 1.25, "run_admissible")])
def test_each_improvement_condition_fails_alone(rules, slu, lo, aa, twin_point, failing):
    c = contrast(lock=(72.0, 85.0), aa=aa, twin_point=twin_point, arms={"l1-support": (slu, lo, slu - 72.0 + 1.0, 85.0)})
    out = verdicts(rules, c)["improvement"]["l1-support"]
    if failing is None:
        assert out["word"] == "IMPROVEMENT-ADMISSIBLE"
    else:
        assert out["word"] == "NO-CROSS-RUN-STATEMENT" and [k for k, ok in out["conditions"].items() if not ok] == [failing]


def test_the_verdict_words_are_the_recipes_and_the_within_run_words_stay_the_historical_ones(rules):
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    ext = recipe["port_extensions"]
    assert ext["parity_band"]["words"] == [port.WORD_PARITY, port.WORD_OUTSIDE]
    assert ext["registered_comparisons"][0]["words"] == [port.WORD_ADMISSIBLE, port.WORD_NOT_ADMISSIBLE]
    assert "IMPROVEMENT-ADMISSIBLE" in ext["cross_run_bar"]["improvement_rule"] and port.WORD_IMPROVEMENT == "IMPROVEMENT-ADMISSIBLE"
    assert recipe["verdict_words"] == sorted(hist.WORDS) and list(hist.WORDS) == ["NOMINATED", "HARM", "NEUTRAL"]
    # an arm that is NOMINATED within the run but below the cross-run floor makes no cross-run statement; the within-run word stands on its own
    c = contrast(arms={"l1-support": (74.0, 1.5, 3.0, 85.0)})
    out = verdicts(rules, c, {"l1-support": "NOMINATED"})["improvement"]["l1-support"]
    assert out["word"] == "NO-CROSS-RUN-STATEMENT" and out["within_run_word"]["value"] == "NOMINATED"


# --------------------------------------------------------------------------- (5) the smoke table and the sign-off conditions


def _smoke_run(tmp_path):
    rows = port_outputs_rows(4)
    for row in rows:
        for b in row["branches"]:
            for sample in (b["solver_samples"] or {}).get("samples", []) + (b["perception_samples"] or {}).get("samples", []):
                row["requests"].setdefault(sample["contact_id"], f"req-{sample['contact_id']}")
    cell = pf.CELLS["smoke"]
    fx._write_sealed(tmp_path, rows, cell["run"], cell["attempt"])
    return rows


def test_the_port_signoff_conditions_point_at_the_port_read_id_and_validate():
    committed = json.loads((PKG / "signoff-conditions-v1.json").read_text(encoding="utf-8"))
    assert committed == pf.signoff_conditions()
    smoke_signoff.validate_conditions(committed)
    assert committed["table_relpath"] == f"analysis/{pf.READ_RUN_ID}/smoke-decode-cost.json" and HIST_READ_ID not in committed["table_relpath"]
    assert committed["package"] == pf.PKG_NAME and committed["table_schema"] == fam.SMOKE_SCHEMA == hist.SCHEMA_SMOKE
    assert committed["expected_arms"] == sorted(pf.ARM_ORDER) and "lock-rep2" in committed["expected_arms"] and pf.PORT_READER in committed["_note"]
    # only the package name, the table path, the expected arms (the port arm table adds the replicate) and the note differ from the historical conditions
    historical = fam.signoff_conditions()
    assert {k for k in committed if committed[k] != historical.get(k)} == {"package", "table_relpath", "expected_arms", "_note"}


def test_the_smoke_reader_writes_the_table_the_signoff_accepts(tmp_path):
    _smoke_run(tmp_path)
    out = tmp_path / pf.ANALYSIS_ROOT
    assert port.main(["--stage", "smoke", "--data-root", str(tmp_path)]) == 0
    table_path = out / "smoke-decode-cost.json"
    table = json.loads(table_path.read_text(encoding="utf-8"))
    assert table["schema"] == fam.SMOKE_SCHEMA and set(table["arms"]) == set(pf.ARM_ORDER) and table["absent_columns"] == 0 and table["unattributed_contacts"] == 0
    assert table["port"]["read_run_id"] == pf.READ_RUN_ID and table["port"]["run_id"] == pf.CELLS["smoke"]["run"] and "A-SECRET-TRANSCRIPT" not in table_path.read_text()
    assert port.main(["--stage", "smoke", "--data-root", str(tmp_path)]) == 6                       # write-once
    # the committed port conditions, resolved against the data root exactly as step_signoff.sh / smoke_signoff.py do (no --table)
    marker = tmp_path / "state" / "signoff.marker"
    rc = smoke_signoff.main(["--conditions", str(PKG / "signoff-conditions-v1.json"), "--data-root", str(tmp_path), "--marker", str(marker), "--utc", "2026-10-30T00:00:00Z"])
    assert rc == 0 and marker.is_file() and "PASS" in marker.read_text().splitlines()[0]
    checks = smoke_signoff.evaluate(pf.signoff_conditions(), table)
    assert checks and [c for c in checks if not c["ok"]] == []


def test_the_smoke_signoff_fails_on_a_capped_port_table(tmp_path):
    rows = port_outputs_rows(4)
    mapping = {}
    for row in rows:
        mapping.update(hist.arm_contact_map([row]))
    contacts = fx._synthetic_contacts(rows, cap_hit=True)
    table = hist.smoke_stage(hist.npc.attribute(contacts, mapping), missing=hist.missing_columns(rows, contacts, pf.ARM_ORDER))
    assert any(not c["ok"] for c in smoke_signoff.evaluate(pf.signoff_conditions(), table))


def test_the_smoke_reader_refuses_a_missing_sealed_smoke(tmp_path, capsys):
    assert port.main(["--stage", "smoke", "--data-root", str(tmp_path)]) == 2
    assert "missing sealed file" in capsys.readouterr().err


# --------------------------------------------------------------------------- (6) the read stage end to end


def _read_run(tmp_path, n=40, *, gain=1.0):
    ids = [f"s{i}" for i in range(n)]
    rows = port_outputs_rows(n)
    for row in rows:
        for b in row["branches"]:
            for sample in (b["solver_samples"] or {}).get("samples", []) + (b["perception_samples"] or {}).get("samples", []):
                row["requests"].setdefault(sample["contact_id"], f"req-{sample['contact_id']}")
    cell = pf.CELLS["arm"]
    run_dir = fx._write_sealed(tmp_path, rows, cell["run"], cell["attempt"])
    (run_dir / f"{cell['run']}.run-manifest.json").write_text(json.dumps({"status": "COMPLETED"}), encoding="utf-8")
    roster = tmp_path / "repo" / fam.ROSTER[cell["roster"]]
    roster.parent.mkdir(parents=True)
    roster.write_text(json.dumps({"items": [{"sample_id": i, "utterance_id": i, "stratum": "alarm" if int(i[1:]) % 2 else "play"} for i in ids]}), encoding="utf-8")
    columns = tmp_path / pf.ANALYSIS_ROOT / "columns"
    columns.mkdir(parents=True)
    scores = port_arms_scores(ids, gain=gain, intent=1.0)
    for arm, per_row in scores.items():
        doc = {hist.SCORER: {"per_sample": [dict(sample_id=i, **per_row[i]) for i in ids]}}
        (columns / hist.COLUMN_SCORE_NAME.format(arm=arm, scorer=hist.SCORER)).write_text(json.dumps(doc), encoding="utf-8")
    return ids


def _leaves_with_source(node, found):
    if isinstance(node, dict):
        if set(node) == {"value", "source", "locator"}:
            found.append(node)
        else:
            for v in node.values():
                _leaves_with_source(v, found)
    elif isinstance(node, list):
        for v in node:
            _leaves_with_source(v, found)
    return found


def _resolve(doc, dotted):
    node = doc
    for part in dotted.replace("[", ".").replace("]", "").split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def test_the_read_stage_writes_the_table_the_verdicts_and_cites_every_number(tmp_path):
    import hashlib

    _read_run(tmp_path)
    argv = ["--stage", "read", "--data-root", str(tmp_path), "--repo-root", str(tmp_path / "repo"), "--recipe", str(RECIPE), "--draws", "200"]
    assert port.main(argv) == 0
    out = tmp_path / pf.ANALYSIS_ROOT
    assert sorted(p.name for p in out.iterdir() if p.is_file()) == ["README.md", "port-verdicts.json", "read.json"]
    read = json.loads((out / "read.json").read_text(encoding="utf-8"))
    verdicts_doc = json.loads((out / "port-verdicts.json").read_text(encoding="utf-8"))
    assert read["schema"] == hist.SCHEMA_READ and set(read["arm_words"]["words"]) == set(fam.CANDIDATE_ARMS) and read["contrast"]["delivering"] == "lock"
    assert read["port"]["read_run_id"] == pf.READ_RUN_ID and (out / "README.md").read_text(encoding="utf-8").startswith("# ")
    assert verdicts_doc["schema"] == port.VERDICTS_SCHEMA and verdicts_doc["read_run_id"] == pf.READ_RUN_ID and verdicts_doc["interval"]["registered"] is False
    # the fixture lock is 75.0 SLU / 100 intent: far above the band; aa is its exact repeat: admissible but also outside the band
    assert verdicts_doc["parity"]["lock"]["word"] == "OUTSIDE-BAND" and verdicts_doc["twin_admissibility"]["word"] == "ADMISSIBLE"
    assert verdicts_doc["twin_admissibility"]["aa_minus_lock_slu_pp"]["value"] == 0.0
    assert set(verdicts_doc["improvement"]) == set(fam.CANDIDATE_ARMS) and verdicts_doc["summary"]["s2_may_start"] is False
    # every cited number names an input that exists, matches its sha256 and (for read_json) the place it was read from
    inputs = verdicts_doc["inputs"]
    assert {"recipe", "read_json", "roster", "manifest", "column_scores:lock", "column_scores:aa", "column_scores:lock-rep2", "column_scores:l1-support", "column_scores:l2-union"} <= set(inputs)
    assert inputs["read_json"]["sha256"] == hashlib.sha256((out / "read.json").read_bytes()).hexdigest()
    assert inputs["recipe"]["sha256"] == hashlib.sha256(RECIPE.read_bytes()).hexdigest()
    for name, entry in inputs.items():
        path = Path(entry["path"])
        path = path if path.is_absolute() else (tmp_path / "repo" / path if name in ("recipe", "roster") else tmp_path / path)
        if name == "recipe":
            path = RECIPE
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], name
    cites = _leaves_with_source(verdicts_doc, [])
    assert len(cites) > 40
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    for cited in cites:
        assert cited["source"] in inputs, cited
        if cited["source"] == "read_json" and not cited["locator"].startswith("arm_words"):
            assert _resolve(read, cited["locator"]) == cited["value"], cited
        if cited["source"] == "recipe" and "+" not in cited["locator"]:
            assert _resolve(recipe, cited["locator"]) == cited["value"], cited
    assert port.main(argv) == 6                                                                     # write-once


def test_the_read_stage_refuses_an_incomplete_manifest_and_a_missing_column(tmp_path, capsys):
    _read_run(tmp_path)
    cell = pf.CELLS["arm"]
    manifest = tmp_path / "runs" / cell["run"] / cell["attempt"] / f"{cell['run']}.run-manifest.json"
    manifest.write_text(json.dumps({"status": "RUNNING"}), encoding="utf-8")
    argv = ["--stage", "read", "--data-root", str(tmp_path), "--repo-root", str(tmp_path / "repo"), "--recipe", str(RECIPE), "--draws", "50"]
    assert port.main(argv) == 2 and "not COMPLETED" in capsys.readouterr().err
    manifest.write_text(json.dumps({"status": "COMPLETED"}), encoding="utf-8")
    (tmp_path / pf.ANALYSIS_ROOT / "columns" / hist.COLUMN_SCORE_NAME.format(arm="aa", scorer=hist.SCORER)).unlink()
    assert port.main(argv) == 2 and "missing column score file" in capsys.readouterr().err
    assert not (tmp_path / pf.ANALYSIS_ROOT / "port-verdicts.json").exists()                      # nothing is written on a refusal


# --------------------------------------------------------------------------- (7) noise model M1: the replicate, the widened CI95, per-row F1 and the words


def _versus(point, lo, hi, *, per_row=None):
    entry = {"slu_f1": iv(point, lo, hi), "intent": iv(0.0, -0.5, 0.5), "intent_saves_harms_ties_vs_delivering": {"saves": 0, "harms": 0, "ties": 0}}
    if per_row is not None:
        entry["per_row_slu_f1"] = iv(*per_row)
    return entry


def _l1_contrast(slu, per_row, intent=(0.0, -0.5, 0.5)):
    """A noise-adjusted-shaped record with one L1 arm: ``slu`` / ``per_row`` are (point, lo, hi) in pp."""

    versus = {"aa": _versus(0.1, -0.9, 1.1, per_row=(0.0, -0.5, 0.5)), "l1-support": _versus(*slu, per_row=per_row)}
    versus["l1-support"]["intent"] = iv(*intent)
    return {"n": 1080, "delivering": "lock", "levels": {"lock": {}, "aa": {}, "l1-support": {}}, "versus_delivering": versus}


def test_the_recipe_carries_the_noise_model_and_the_reader_refuses_a_recipe_that_disagrees(tmp_path):
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    model = recipe["port_extensions"]["noise_model"]
    assert model["copies"] == list(pf.REPLICATE_SET) == ["lock", "aa", "lock-rep2"] and model["max_replicates"] == 2 and model["replicate_arms"] == ["lock-rep2"]
    assert model["formula"] == port.paired_ci.FORMULA and recipe["arms"] == sorted(recipe["arms"], key=pf.ARM_ORDER.index) == list(pf.ARM_ORDER)
    for edit, message in ((lambda d: d["port_extensions"]["noise_model"].update(formula="x"), "noise_model.formula"),
                          (lambda d: d["port_extensions"].pop("noise_model"), "noise_model is absent"),
                          (lambda d: d["port_extensions"]["noise_model"].update(copies=["lock", "aa"]), "noise_model.copies"),
                          (lambda d: d["arms"].remove("lock-rep2"), "arms")):
        doc = json.loads(RECIPE.read_text(encoding="utf-8"))
        edit(doc)
        bad = tmp_path / "recipe.json"
        bad.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(port.ReadError, match=message):
            port.load_recipe(bad)


def test_nominated_needs_both_the_pooled_and_the_per_row_lower_bound_above_zero():
    ok = port.port_arm_words(_l1_contrast((3.0, 1.6, 4.4), (1.0, 0.2, 1.8)))["words"]["l1-support"]
    assert ok["word"] == "NOMINATED" and ok["hist_word"] == "NOMINATED" and ok["nominated_withheld_by"] == [] and ok["next_step"] == "335 B Fold A column"
    # pooled lower bound clears the 1.49 floor but the per-row lower bound is below zero: the historical word is NOMINATED, the port word is withheld
    withheld = port.port_arm_words(_l1_contrast((3.0, 1.6, 4.4), (-0.3, -1.1, 0.5)))["words"]["l1-support"]
    assert withheld["hist_word"] == "NOMINATED" and withheld["word"] == "NEUTRAL" and withheld["nominated_withheld_by"] == ["per_row_primary_ci_lo_above_zero"]
    assert withheld["next_step"] is None and withheld["per_row_slu_delta_pp"] == -0.3
    # a per-row lower bound of exactly zero is not above zero
    assert port.port_arm_words(_l1_contrast((3.0, 1.6, 4.4), (0.5, 0.0, 1.0)))["words"]["l1-support"]["word"] == "NEUTRAL"
    # the per-row loss is resolved (upper bound below minus the floor): still withheld, and the pooled rule alone would have said NOMINATED
    loss = port.port_arm_words(_l1_contrast((3.0, 1.6, 4.4), (-3.0, -4.0, -2.0)))["words"]["l1-support"]
    assert loss["hist_word"] == "NOMINATED" and loss["word"] == "NEUTRAL" and set(loss["nominated_withheld_by"]) >= {"per_row_primary_ci_lo_above_zero"}
    # HARM is the historical pooled rule and is never softened
    harm = port.port_arm_words(_l1_contrast((-3.0, -4.4, -1.6), (0.5, 0.1, 0.9)))["words"]["l1-support"]
    assert harm["word"] == "HARM" and harm["hist_word"] == "HARM"
    with pytest.raises(port.ReadError, match="per_row_slu_f1"):
        port.port_arm_words(contrast(arms={"l1-support": (75.0, 1.6, 4.9, 85.0)}))


def test_the_l2_lever_is_judged_on_intent_with_the_per_row_slu_guard():
    versus = {"aa": _versus(0.0, -1.0, 1.0, per_row=(0.0, -0.5, 0.5)),
              "l2-union": dict(_versus(0.2, -1.0, 1.4, per_row=(0.1, -0.4, 0.6)), intent=iv(1.2, 0.8, 1.6))}
    c = {"n": 1080, "delivering": "lock", "levels": {}, "versus_delivering": versus}
    assert port.port_arm_words(c)["words"]["l2-union"]["word"] == "NOMINATED"
    versus["l2-union"]["per_row_slu_f1"] = iv(-3.0, -4.5, -2.0)                       # pooled SLU is fine, the typical row lost 3 pp: the guard fails at the per-row level
    out = port.port_arm_words(c)["words"]["l2-union"]
    assert out["word"] == "NEUTRAL" and "per_row_slu_no_resolved_loss" in out["nominated_withheld_by"]


def test_the_improvement_verdict_adds_the_per_row_condition_only_when_the_record_carries_it(rules):
    c = contrast(lock=(72.0, 85.0), aa=(72.3, 85.2), arms={"l1-support": (76.0, 0.5, 6.0, 85.0)})
    assert "per_row_slu_delta_ci95_lower_above_zero" not in verdicts(rules, c)["improvement"]["l1-support"]["conditions"]       # a pre-M1 record
    c["versus_delivering"]["l1-support"]["per_row_slu_f1"] = iv(0.8, -0.2, 1.8)
    c["levels"]["l1-support"]["per_row_slu_f1_pct"] = 70.0
    out = verdicts(rules, c)["improvement"]["l1-support"]
    assert out["conditions"]["per_row_slu_delta_ci95_lower_above_zero"] is False and out["word"] == "NO-CROSS-RUN-STATEMENT"
    c["versus_delivering"]["l1-support"]["per_row_slu_f1"] = iv(0.8, 0.1, 1.8)
    assert verdicts(rules, c)["improvement"]["l1-support"]["word"] == "IMPROVEMENT-ADMISSIBLE"


def test_the_read_stage_widens_every_ci_and_reports_per_row_f1(tmp_path):
    _read_run(tmp_path)
    argv = ["--stage", "read", "--data-root", str(tmp_path), "--repo-root", str(tmp_path / "repo"), "--recipe", str(RECIPE), "--draws", "200"]
    assert port.main(argv) == 0
    out = tmp_path / pf.ANALYSIS_ROOT
    read = json.loads((out / "read.json").read_text(encoding="utf-8"))
    verdicts_doc = json.loads((out / "port-verdicts.json").read_text(encoding="utf-8"))
    contrast_doc = read["contrast"]
    assert set(contrast_doc["versus_delivering"]) >= {"aa", "lock-rep2", "l1-support", "l2-union"} and set(read["contrast_bootstrap_only"]["versus_delivering"]) == set(contrast_doc["versus_delivering"])
    model = read["noise_model"]
    assert model["copies"] == ["lock", "aa", "lock-rep2"] and [m["n_pairs"] for m in model["metrics"].values()] == [3, 3, 3]
    # the fixture's copies are exact repeats of lock: zero replicate variance, zero widening, the reported CI95 IS the bootstrap interval
    for metric in ("slu_f1", "intent", "per_row_slu_f1"):
        assert model["metrics"][metric]["variance_pp2"] == 0.0
    for arm in ("l1-support", "l2-union"):
        entry = contrast_doc["versus_delivering"][arm]
        assert entry["slu_f1"]["widening_pp"] == 0.0 and entry["slu_f1"]["ci95_lo_pp"] == entry["slu_f1"]["bootstrap_ci95_lo_pp"] == read["contrast_bootstrap_only"]["versus_delivering"][arm]["slu_f1"]["ci95_lo_pp"]
        assert set(entry["per_row_slu_f1"]) >= {"point_pp", "ci95_lo_pp", "ci95_hi_pp", "widening_pp"} and "per_row_slu_f1_pct" in contrast_doc["levels"][arm]
    assert read["interval"]["bootstrap"]["draws"] == 200 and "widened" in read["interval"]["kind"] and read["arm_words"]["ci_definition"] == contrast_doc["ci_definition"]
    assert {"hist_word", "nominated_withheld_by"} <= set(read["arm_words"]["words"]["l1-support"])
    assert read["arm_words_bootstrap_only"]["words"]["l1-support"]["word"] in ("NOMINATED", "NEUTRAL", "HARM")
    text = (out / "README.md").read_text(encoding="utf-8")
    assert "Noise model (M1)" in text and "lock-rep2" in text
    # the verdict file carries the noise model and the per-row fields, every number cited
    assert verdicts_doc["schema"] == "saea-l338-port-verdicts-v2" and verdicts_doc["noise_model"]["copies"] == ["lock", "aa", "lock-rep2"]
    assert verdicts_doc["noise_model"]["metrics"]["slu_f1"]["pairs"][0]["pair"] == ["aa", "lock"] and verdicts_doc["noise_model"]["metrics"]["slu_f1"]["n_pairs"] == 3
    improvement = verdicts_doc["improvement"]["l1-support"]
    assert "per_row_slu_delta_ci95_lower_above_zero" in improvement["conditions"] and improvement["per_row_slu_delta_vs_lock_pp"]["source"] == "read_json"
    assert verdicts_doc["rule_text"]["ci95"].startswith("every CI95 is the row-paired bootstrap interval widened")


def test_the_read_stage_refuses_a_missing_replicate_column(tmp_path, capsys):
    _read_run(tmp_path)
    (tmp_path / pf.ANALYSIS_ROOT / "columns" / hist.COLUMN_SCORE_NAME.format(arm="lock-rep2", scorer=hist.SCORER)).unlink()
    argv = ["--stage", "read", "--data-root", str(tmp_path), "--repo-root", str(tmp_path / "repo"), "--recipe", str(RECIPE), "--draws", "50"]
    assert port.main(argv) == 2 and "lock-rep2" in capsys.readouterr().err
    assert not (tmp_path / pf.ANALYSIS_ROOT / "port-verdicts.json").exists()
