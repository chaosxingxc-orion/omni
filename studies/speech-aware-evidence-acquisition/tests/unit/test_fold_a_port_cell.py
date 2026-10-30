"""The Fold A confirmation twin of the L338 port flight: ids, caps, derive, writer, package.

The cell is the port Fold B flight moved onto TRAIN Fold A (``SLURP-EVO-FOLD-A-V2``) by one declared token transform (``scripts/flights/nx/fold_a_port_derive_cells.py``) and written up
by ``fold_a_port_write_registration.py`` (reached through ``l338_port_write_registration.py --family fold-a-confirmation``). The Fold A preparations FA1-FA5 are not run: the cell pins
them by the zero sentinel and the facts say PENDING. Nothing here contacts a model, a server or an encoder; every derive and writer run goes into a throwaway root.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _extra in (REPO / "scripts" / "flights" / "nx", REPO / "scripts" / "flights" / "direct", REPO / "scripts" / "flights" / "tfrl", REPO / "tests" / "unit", REPO / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import fold_a_port_derive_cells as fdc  # noqa: E402
import fold_a_port_family as fa  # noqa: E402
import fold_a_port_write_registration as fw  # noqa: E402
import l338_family as fam  # noqa: E402
import l338_port_family as pf  # noqa: E402
import l338_port_write_registration as w  # noqa: E402
from test_l338_port_family_switch import DATA_ROOT, FOLD_A_PKG_NAME, MINT_COMMIT, make_root, needs_data  # noqa: E402

from speech_aware_evidence_acquisition.core.config import compose, protocol_hash  # noqa: E402

CJK = re.compile("[\\u4e00-\\u9fff]")
ZERO = "0" * 64
COMMIT = "28bf3176" + "0" * 32
AT = "2026-10-30T00:00:00+00:00"
#: composed-config hash / protocol hash of the Fold A cell by fragment choice (the arm table, caps and pins are the port family's; only the identities and the zero-sentinel pins differ).
FOLD_A_HASHES = {"default": ("f26ec1315303783a2ab5b6bd5335a3d60512c5d976dc056f544f62622702afd5", "86a2d5e20f3a3d5568a671e04a8cdc8818157916ef24096cb1b38df697e4817c")}
FOLD_A_FILES = ["README.md", "countersign-row-texts-v1.json", "draft-ledger-rows.md", "fit-plan-registered-v1.json", "generation-plan-registered-v1.json", "protocol-registered-v1.json",
                "registered-facts-v1.json", "root-admission-draft-v1.json", "signoff-conditions-v1.json"]


@pytest.fixture(autouse=True)
def _restore_default_fragment():
    yield
    pf.select_model_fragment("default")


@pytest.fixture(scope="module")
def derived_root(tmp_path_factory):
    pf.select_model_fragment("default")
    root = make_root(tmp_path_factory.mktemp("fold_a") / "root", packages=("2026-10-28-slurp-l338-lever-column-fold-b-npu-port-v1",))
    fdc.derive(root)
    return root


def _load(root: Path, kind: str, name: str) -> dict:
    return json.loads((root / "configs" / kind / f"{name}.json").read_text(encoding="utf-8"))


def _leaves(node, prefix=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaves(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _leaves(value, f"{prefix}[{index}]")
    else:
        yield prefix, node


# ---- ids and caps --------------------------------------------------------------------------------------------------------------------------


def test_ids_are_the_fold_b_port_ids_with_the_fold_token_swapped():
    spec, hist = fa.CELLS["arm"], pf.CELLS["arm"]
    assert (spec["cell"], spec["baseline"], spec["run"], spec["fit_run"]) == (
        "slurp-native-stage3-l338-lever-column-fold-a-npu-port-v1", "slurp-native-stage3-l338-lever-column-fold-a-baseline-raw-port-v1",
        "SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-A-NPU-PORT-V1", "SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-A-FIT-NPU-PORT-V1")
    assert spec["attempt"] == "attempt-slurp-native-stage3-l338-lever-column-fold-a-generation-npu-port-v1"
    assert spec["fit_attempt"] == "attempt-slurp-native-stage3-l338-lever-column-fold-a-fit-npu-port-v1"
    assert spec["historical_run"] == hist["run"] and fa.ALL_KEYS == ("arm",)
    assert fa.DECISION_ID == "slurp-l338-lever-column-fold-a-npu-port-admission-v1"
    assert fa.DECISION_RELPATH == "coordination/slurp-l338-lever-column-fold-a-npu-port-v1/root-admission-v1.json"
    assert fa.PACKAGE == "docs/readiness/2026-10-30-slurp-l338-fold-a-confirmation-npu-port-v1"
    assert fa.eligible_run_ids() == [spec["run"], spec["fit_run"]] and fa.eligible_attempt_ids() == [spec["attempt"], spec["fit_attempt"]]
    assert fa.READ_RUN_ID == "SAEA-SLURP-L338-LEVER-COLUMN-FOLD-A-READ-NPU-PORT-V1"


def test_the_swap_protects_the_model_fragment_and_the_session():
    assert fa.swap(pf.DEFAULT_MODEL_FRAGMENT) == pf.DEFAULT_MODEL_FRAGMENT and fa.swap(pf.BUDGET6144_MODEL_FRAGMENT) == pf.BUDGET6144_MODEL_FRAGMENT
    assert fa.swap(pf.SERVICE_SESSION) == pf.SERVICE_SESSION and not fa.has_fold_b(pf.SERVICE_SESSION)
    assert fa.swap("TRAIN Fold B fold-b FOLD-B") == "TRAIN Fold A fold-a FOLD-A" and fa.has_fold_b("x FOLD-B y")


def test_caps_follow_the_port_formulas_on_the_fold_a_roster():
    assert (fa.ROSTER_PATH, fa.SPLIT_ID, len(fa.ROSTER["items"])) == ("docs/receipts/rosters/slurp-evo-fold-a-v2.json", "SLURP-EVO-FOLD-A-V2", 1080)
    assert fa.SPLIT_IDENTITY_HASH == "61b0f06254975873bea6afb287ab54823dc71b57e5398697b9bef817e25398b6"
    assert math.isclose(fa.ROSTER_AUDIO_SECONDS, 3084.5284375, abs_tol=1e-6)
    assert pf.CONTACTS_PER_ROW == 17 and fa.AUDIO_FACTOR == 34
    assert (pf.generation_calls("arm"), pf.fit_calls("arm"), fa.audio_seconds()) == (36720, 18360, 104874)
    assert fa.audio_seconds() == math.ceil(fa.ROSTER_AUDIO_SECONDS * (pf.CONTACTS_PER_ROW + pf.FALLBACK_CONTACTS_PER_ROW_CEILING) - 1e-9)
    # the FIT cap is the gate's own requirement (raw arm: rows x contacts x 1), not the packet-arm factor
    from speech_aware_evidence_acquisition.core.preparation_fit import native_contact_required_cap  # noqa: PLC0415

    assert native_contact_required_cap(1080 * pf.CONTACTS_PER_ROW, feature_packets=False) == 18360


# ---- the derive ---------------------------------------------------------------------------------------------------------------------------------


def test_derive_writes_the_cell_and_the_comparator_pair_and_nothing_else(derived_root):
    for kind, name in (("baseline", fa.CELLS["arm"]["baseline"]), ("experiment", fa.CELLS["arm"]["cell"]), ("baseline", fa.COMPARATOR_CELL), ("experiment", fa.COMPARATOR_CELL)):
        path = derived_root / "configs" / kind / f"{name}.json"
        assert path.is_file() and not path.is_symlink(), name
        assert not CJK.search(path.read_text(encoding="utf-8"))


def test_baseline_is_the_raw_arm_c17_module_with_fold_a_pins(derived_root):
    base = _load(derived_root, "baseline", fa.CELLS["arm"]["baseline"])
    fold_b = _load(derived_root, "baseline", pf.CELLS["arm"]["baseline"])
    assert base["arm"] == "raw-speech-agent-native-tools" and "speech_packet_promotion_receipt" not in base and base["policy"] == pf.PROGRAM == "phase-native-module-v1-c17"
    assert base["native_contact_fit"] == {**fold_b["native_contact_fit"], "run_id": fa.CELLS["arm"]["fit_run"], "attempt_id": fa.CELLS["arm"]["fit_attempt"]}
    assert base["native_contact_fit"]["contact_cap"] == 18360
    arms = [a["arm_id"] for a in base["native_episode"]["roles"]["perception"]["sampling_variants"]["arms"]]
    assert tuple(arms) == pf.ARM_ORDER
    # the roster receipt is pinned for real, every Fold A artefact of FA1-FA5 by the zero sentinel
    receipts = {r["relpath"]: r["sha256"] for r in base["preparation_source_receipts"]}
    roster_sha = hashlib.sha256((REPO / fa.ROSTER_PATH).read_bytes()).hexdigest()
    assert receipts[fa.ROSTER_PATH] == roster_sha == "1fe7908c31396774d7535fdf37bd791762215c324999d66ed7beb9e7b2f9033d"
    pending = {rel for rel, sha in receipts.items() if sha == ZERO}
    assert {fw.PREPARATION_ARTIFACTS[k] for k in ("encode", "alias", "content", "agreement")} == pending
    assert base["prepared_phase_supply"] == {"relpath": fw.PREPARATION_ARTIFACTS["supply"], "sha256": ZERO}
    store = base["knowledge_stores"]["intent-boundaries"]["phase_supply"]["perception"]
    assert store["agreement_source"]["sha256"] == ZERO and store["query_source"]["sha256"] == ZERO and "fold-a" in store["query_source"]["relpath"]
    assert base["native_episode"]["consumption_module"]["components"][3]["params"]["agreement_source"]["sha256"] == ZERO
    # shared artefacts keep their real Fold B port pins
    assert store["facet_source"] == fold_b["knowledge_stores"]["intent-boundaries"]["phase_supply"]["perception"]["facet_source"]
    assert base["knowledge_stores"]["train-exemplars"] == {**fold_b["knowledge_stores"]["train-exemplars"], "phase_supply": base["knowledge_stores"]["train-exemplars"]["phase_supply"]}


def test_no_fold_b_token_survives_outside_the_declared_prose_leaves(derived_root):
    for kind, name in (("baseline", fa.CELLS["arm"]["baseline"]), ("experiment", fa.CELLS["arm"]["cell"]), ("baseline", fa.COMPARATOR_CELL), ("experiment", fa.COMPARATOR_CELL)):
        doc = _load(derived_root, kind, name)
        assert fdc.scan_fold_b(doc, kind) == [], (kind, name)
    base = _load(derived_root, "baseline", fa.CELLS["arm"]["baseline"])
    assert base["what_this_is"].startswith("The Fold A CONFIRMATION twin of ") and "The Fold B port text, carried over unchanged:" in base["what_this_is"]
    exp = _load(derived_root, "experiment", fa.CELLS["arm"]["cell"])
    assert exp["purpose"].startswith(fa.CELLS["arm"]["run"]) and exp["split_id"] == "SLURP-EVO-FOLD-A-V2"
    assert exp["service_session"] == pf.SERVICE_SESSION and exp["cells"][0]["model_fragment"] == pf.MODEL_FRAGMENT


def test_experiment_caps_population_and_readiness(derived_root):
    exp = _load(derived_root, "experiment", fa.CELLS["arm"]["cell"])
    assert (exp["execution_plan"]["planned_model_calls"], exp["execution_plan"]["planned_speech_audio_seconds"]) == (36720, 104874)
    assert (exp["family_budget"]["generation_calls_cap"], exp["family_budget"]["fit_endpoint_calls_cap"], exp["family_budget"]["repeated_speech_audio_seconds_cap"]) == (36720, 18360, 104874)
    pop = exp["population"]
    assert (pop["count"], pop["roster_receipt"], pop["split_identity_hash"], pop["dataset_fragment"]) == (1080, fa.ROSTER_PATH, fa.SPLIT_IDENTITY_HASH, fa.DATASET_FRAGMENT)
    assert exp["populations"][0]["ids"] == [item["sample_id"] for item in fa.ROSTER["items"]] and exp["populations"][0]["dataset_fragment"] == fa.DATASET_FRAGMENT
    cell = exp["cells"][0]
    assert cell["readiness"]["state"] == "held" and cell["dataset_fragment"] == fa.DATASET_FRAGMENT
    reqs = cell["readiness"]["requirements"]
    assert reqs[0]["relpath"] == fa.DECISION_RELPATH and reqs[0]["sha256"] == ZERO and reqs[0]["expected_decision_id"] == fa.DECISION_ID
    assert reqs[1]["sha256"] == ZERO and reqs[2]["sha256"] == ZERO and reqs[2]["relpath"] == fa.CELLS["arm"]["supply_relpath"]
    assert exp["core_read_design"]["arms"] == {arm: fa.CELLS["arm"]["run"] for arm in pf.ARM_ORDER}


def test_cell_composes_and_the_hashes_are_pinned(derived_root):
    spec = fa.CELLS["arm"]
    exp = _load(derived_root, "experiment", spec["cell"])
    cell = exp["cells"][0]
    composed = compose(derived_root, cell["model_fragment"], cell["dataset_fragment"], cell["baseline_fragment"], spec["cell"])
    phash = protocol_hash(composed.config_hash, composed.values["arm"], exp["population"]["split_identity_hash"])
    assert (composed.config_hash, phash) == FOLD_A_HASHES["default"]
    assert composed.values["native_contact_fit"]["contact_cap"] == 18360
    # the comparator pair composes too (config only, never flown)
    comp = _load(derived_root, "experiment", fa.COMPARATOR_CELL)["cells"][0]
    compose(derived_root, comp["model_fragment"], comp["dataset_fragment"], comp["baseline_fragment"], fa.COMPARATOR_CELL)


def test_derive_refuses_an_undeclared_moved_leaf(tmp_path, monkeypatch):
    root = make_root(tmp_path / "root", packages=())
    monkeypatch.setitem(fdc.FOLD_A_MOVED_LEAVES, "baseline", ("preparation_source_receipts[",))        # the knowledge-store pins now move undeclared
    with pytest.raises(ValueError, match="undeclared leaves"):
        fdc.derive(root, dry_run=True)


def test_a_fold_b_token_in_a_non_prose_leaf_is_refused(tmp_path, monkeypatch):
    root = make_root(tmp_path / "root", packages=())
    monkeypatch.setattr(fa, "swap", lambda text: text)                    # a transform that forgets the tokens must not pass
    with pytest.raises(ValueError):
        fdc.derive(root, dry_run=True)


def test_pins_file_value_pins_an_artefact_and_empty_pins_leave_the_sentinel(tmp_path, monkeypatch):
    root = make_root(tmp_path / "root", packages=())
    agreement = fw.PREPARATION_ARTIFACTS["agreement"]
    monkeypatch.setattr(fa, "load_pins", lambda: {agreement: "ab" * 32})
    base = fdc.fold_a_baseline(root)
    receipts = {r["relpath"]: r["sha256"] for r in base["preparation_source_receipts"]}
    assert receipts[agreement] == "ab" * 32 and receipts[fw.PREPARATION_ARTIFACTS["alias"]] == ZERO
    assert fa.load_pins() is not None


def test_the_committed_pins_file_is_empty_until_the_preparations_are_run():
    doc = json.loads((REPO / "scripts/flights/nx/fold_a_port_pins.json").read_text(encoding="utf-8"))
    assert doc["pins"] == {} and doc["schema"] == "saea-fold-a-port-pins-v1"


def test_the_filled_pins_file_leaves_no_zero_sentinel_for_a_preparation(tmp_path):
    """Once the Fold A preparations are run and pinned, no pin of the pins file or of a preparation artefact may be the zero sentinel.

    While the pins file is still empty (preparations not run) the check is skipped, never passed: the empty state is covered by the test above.
    """

    doc = json.loads((REPO / "scripts/flights/nx/fold_a_port_pins.json").read_text(encoding="utf-8"))
    if not doc["pins"]:
        pytest.skip("PENDING: fold_a_port_pins.json is empty; the Fold A preparations FA1-FA5 are not run")
    assert all(re.fullmatch("[0-9a-f]{64}", sha) and sha != ZERO for sha in doc["pins"].values())
    assert set(fw.PREPARATION_ARTIFACTS.values()) <= set(doc["pins"]), "every preparation artefact needs a real pin"
    root = make_root(tmp_path / "root", packages=("2026-10-28-slurp-l338-lever-column-fold-b-npu-port-v1",))
    report = fdc.derive(root, dry_run=True)
    assert not set(report["pending_pins"]) & set(fw.PREPARATION_ARTIFACTS.values())


# ---- the writer ---------------------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    if not (DATA_ROOT / "prepared").is_dir():
        pytest.skip("data root not mounted")
    pf.select_model_fragment("default")
    root = make_root(tmp_path_factory.mktemp("fold_a_writer") / "root")
    result = fw.write_package(root, COMMIT, AT, DATA_ROOT)
    return root, result


@needs_data
def test_package_file_set_and_determinism(written, tmp_path):
    root, result = written
    pkg = root / fa.PACKAGE
    assert sorted(p.name for p in pkg.iterdir()) == sorted(FOLD_A_FILES)
    assert (root / fa.PREREG).is_file() and result["ledger_appended"] is None
    second = fw.write_package(make_root(tmp_path / "root2"), COMMIT, AT, DATA_ROOT)
    assert {k: v for k, v in second["shas"].items()} == result["shas"]


@needs_data
def test_facts_identity_caps_and_pending_preparations(written):
    root, _ = written
    facts = json.loads((root / fa.PACKAGE / "registered-facts-v1.json").read_text(encoding="utf-8"))
    ident = facts["identity"]["arm"]
    assert (ident["composed_config_hash"], ident["protocol_hash"]) == FOLD_A_HASHES["default"]
    assert (ident["family_budget"]["generation_calls_cap"], ident["family_budget"]["fit_endpoint_calls_cap"], ident["family_budget"]["repeated_speech_audio_seconds_cap"]) == (36720, 18360, 104874)
    assert facts["accounting"]["caps"] == {"arm": {"generation_calls": 36720, "fit_endpoint_calls": 18360, "speech_audio_seconds": 104874}}
    assert set(facts["pins"]["preparations"]) == {"encode", "alias", "content", "agreement", "supply"}
    assert all(p["status"] == "PENDING" and p["receipt_sha256"] == ZERO for p in facts["pins"]["preparations"].values())
    assert facts["status"] == "DRAFT-HELD" and facts["preparations_status"].startswith("PENDING")
    assert facts["roster"] == {"arm": fa.ROSTER_PATH, "roster_id": "SLURP-EVO-FOLD-A-V2", "split_identity_hash": fa.SPLIT_IDENTITY_HASH}
    block = facts["stage3_measurement"][fa.CELLS["arm"]["run"]]
    assert block["read_roster"] == "SLURP-EVO-FOLD-A-V2" and block["claims_benefit"] is False and len(facts["stage3_measurement"]) == 1
    signoff = json.loads((root / fa.PACKAGE / "signoff-conditions-v1.json").read_text(encoding="utf-8"))
    assert signoff["applicable"] is False and "no smoke" in signoff["_note"].lower()
    draft = json.loads((root / fa.PACKAGE / "root-admission-draft-v1.json").read_text(encoding="utf-8"))
    assert draft["status"] == "DRAFT-NOT-MINTED" and draft["eligible_run_ids"] == fa.eligible_run_ids() and draft["decision_id"] == fa.DECISION_ID
    for name in ("generation-plan-registered-v1.json", "fit-plan-registered-v1.json"):
        plan = json.loads((root / fa.PACKAGE / name).read_text(encoding="utf-8"))
        assert plan["status"].startswith("DRAFT-HELD") and plan["composed_config_hash"] == ident["composed_config_hash"]
        assert plan["prepared_phase_supply"]["relpath"] == fa.CELLS["arm"]["supply_relpath"] and "none exists yet" in plan["prepared_phase_supply"]["note"]
        assert plan["execution_plan"]["split_identity_hash"] == fa.SPLIT_IDENTITY_HASH


@needs_data
def test_ledger_rows_are_unconsumed_provisional_and_fold_a(written):
    root, _ = written
    text = (root / fa.PACKAGE / "draft-ledger-rows.md").read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line.startswith("| 20")]
    assert len(rows) == 7                                                   # five preparation rows, then the two cell rows
    cells = [[c.strip() for c in line.strip().strip("|").split(" | ")] for line in rows]
    assert all(len(c) == 15 for c in cells) and all(c[8] == "no" for c in cells)
    by_run = {c[1]: c for c in cells}
    arm = by_run[fa.CELLS["arm"]["run"]]
    assert (arm[9], arm[10]) == ("36720", "104874") and arm[11].startswith("PROVISIONAL") and arm[5] == fa.CARRIER_TEXT and arm[7] == fa.SPLIT_IDENTITY_HASH
    fit = by_run[fa.CELLS["arm"]["fit_run"]]
    assert (fit[9], fit[10]) == ("18360", "104874") and fit[3].endswith("-- FIT spend identity")
    assert by_run[fw.PREPARATION_RUNS["encode"]][9] == "1080" and by_run[fw.PREPARATION_RUNS["encode"]][2] == "bounded-discovery-probe"
    assert all(by_run[fw.PREPARATION_RUNS[k]][9:12] == ["0", "0", "0"] for k in ("alias", "content", "agreement", "supply"))
    assert "Fold B" not in arm[3].split("Fold A confirmation twin of")[0]
    assert text.count("PENDING") >= 1 and "NOT YET RUN" in text
    ledger = (root / "docs" / "exposure-ledger.md").read_bytes()
    assert ledger == (REPO / "docs" / "exposure-ledger.md").read_bytes()    # nothing appended


@needs_data
def test_readme_and_every_file_are_english_draft_and_held(written):
    root, _ = written
    for path in [*(root / fa.PACKAGE).iterdir(), root / fa.PREREG]:
        text = path.read_text(encoding="utf-8")
        assert not CJK.search(text), path.name
    readme = (root / fa.PACKAGE / "README.md").read_text(encoding="utf-8")
    assert "DRAFT / HELD" in readme and "104,874" in readme and "no smoke" in readme.lower() and "zero sentinel" in readme
    assert "fold_a_port_pins.json" in readme


@needs_data
def test_budget_switch_flows_through_the_fold_a_cell(tmp_path):
    root = make_root(tmp_path / "root")
    pf.select_model_fragment("budget6144-max7168")
    result = fw.write_package(root, COMMIT, AT, DATA_ROOT)
    facts = json.loads((root / fa.PACKAGE / "registered-facts-v1.json").read_text(encoding="utf-8"))
    assert facts["identity"]["arm"]["model_fragment"] == pf.BUDGET6144_MODEL_FRAGMENT and facts["model_fragment_switch"]["max_supplied_items"] == 51
    assert facts["identity"]["arm"]["composed_config_hash"] != FOLD_A_HASHES["default"][0]
    base = _load(root, "baseline", fa.CELLS["arm"]["baseline"])
    assert base["native_episode"]["consumption_module"]["sampling_frame"]["max_supplied_items"] == 51
    assert "budget 6144, cap 7168" in base["what_this_is"]
    assert result["package"] == fa.PACKAGE


@needs_data
def test_dispatcher_reaches_the_fold_a_writer(tmp_path, capsys):
    root = make_root(tmp_path / "root")
    assert w.main(["--family", "fold-a-confirmation", "--study-commit", COMMIT, "--registered-at", AT, "--data-root", str(DATA_ROOT), "--root", str(root)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["package"] == fa.PACKAGE and set(FOLD_A_FILES) <= set(printed["shas"])


def test_sources_are_english():
    for rel in ("scripts/flights/nx/fold_a_port_family.py", "scripts/flights/nx/fold_a_port_derive_cells.py", "scripts/flights/nx/fold_a_port_write_registration.py",
                "scripts/flights/nx/fold_a_port_pins.json", "tests/unit/test_fold_a_port_cell.py"):
        assert not CJK.search((REPO / rel).read_text(encoding="utf-8")), rel
