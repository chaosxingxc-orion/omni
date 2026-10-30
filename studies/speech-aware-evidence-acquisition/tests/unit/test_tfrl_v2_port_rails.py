"""The rails declarations of the T cell: the analysis recipe, the READ row draft, the bank-release registration and the writer check that flips ``rails_pending`` to ``rails_declared``.

Proved here: (1) the recipe is byte-deterministic and its sha256 is pinned in the READ row draft (and re-pinned when a rule moves); (2) the bank-release registration is well formed, carries
the T0 rule, requires the vLLM tokenize measure and names the consuming cells, and every consumption frame it records parses; each structural defect is reported; (3) the marker is
``rails_declared`` only when all three deliverables exist on disk and check out, and ``rails_pending`` when any one is missing or moved; (4) the writer emits the marker, the recipe sha and the
files, deterministically, and only the intended package files carry them.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _extra in (REPO / "scripts" / "flights" / "nx", REPO / "scripts" / "flights" / "direct", REPO / "scripts" / "flights" / "tfrl", REPO / "scripts" / "analysis", REPO / "tests" / "unit", REPO / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import l338_port_family as pf  # noqa: E402
import tfrl_v2_common as cm  # noqa: E402
import tfrl_v2_port_family as tf  # noqa: E402
import tfrl_v2_port_rails as rails  # noqa: E402
import tfrl_v2_port_write_registration as tw  # noqa: E402
from test_l338_port_family_switch import DATA_ROOT, make_root, needs_data  # noqa: E402
from test_tfrl_v2_port_cell import AT, COMMIT, PKG_DIR, PREREG_NAME, _t_config_names  # noqa: E402

from speech_aware_evidence_acquisition.core import stage3_algorithms as s3  # noqa: E402
from speech_aware_evidence_acquisition.rails import tfrl_v2_consumption as rail  # noqa: E402

reg = tw.reg
CJK = re.compile("[\\u4e00-\\u9fff]")
#: sha256 of the committed-form recipe bytes: a moved rule moves this pin (and the READ row draft that carries it).
RECIPE_SHA = "303d238a1fb0979862607b2ccb931ce492c7650c6f5629fd6092aa5fc8cce3ab"


@pytest.fixture(autouse=True)
def _default_fragment():
    pf.select_model_fragment("default")
    tf.select_model_fragment()
    yield
    pf.select_model_fragment("default")
    tf.select_model_fragment()


def make_t_root(dst: Path) -> Path:
    root = make_root(dst, packages=(), copy_config_files=_t_config_names())
    for child in (root / "docs" / "readiness").iterdir():
        if child.name in (PKG_DIR, PREREG_NAME) and child.is_symlink():
            child.unlink()
    return root


@pytest.fixture()
def rails_root(tmp_path):
    return make_t_root(tmp_path / "root")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- (1) the recipe and the READ row draft ----------------------------------------------------------------------------------------------------------


def test_the_recipe_is_deterministic_and_its_sha_is_pinned():
    assert rails.recipe_bytes() == rails.recipe_bytes() and rails.recipe_sha256() == hashlib.sha256(rails.recipe_bytes()).hexdigest()
    assert rails.recipe_sha256() == RECIPE_SHA
    doc = rails.recipe_document()
    assert doc["schema"] == rails.RECIPE_SCHEMA and doc["read_run_id"] == tf.READ_RUN_ID and doc["reader"] == rails.READER
    assert doc["stop_rule"]["construction_gain_pp"] == 1.0 and doc["stop_rule"]["max_iterations"] == 3
    assert doc["bank_consuming_arms"] == [tf.T_ARM] and doc["delivering"] == "lock" and doc["replicates"] == ["lock-rep2"]
    assert doc["fold_b_read"]["noise_model"]["copies"] == ["lock", "aa", "lock-rep2"]
    assert doc["contacts"]["model"] == 0 and doc["contacts"]["server"] == 0 and "timestamp" not in json.dumps(doc) and "registered_at" not in json.dumps(doc)
    assert {"positive_rate", "mean_best_of_k_gain_pp", "rule_survivors", "exemplar_coverage_per_scenario"} <= set(doc["construction_read"]["per_half"])
    assert "never_from" in doc["construction_read"]["construction_gain"]


def test_the_read_row_draft_pins_the_recipe_sha_and_a_moved_rule_moves_the_pin():
    text = rails.read_row_text(rails.recipe_sha256())
    assert f"`{RECIPE_SHA}`" in text and rails.RECIPE_RELPATH in text and rails.READER in text and tf.READ_RUN_ID in text and "DRAFT" in text
    assert rails.read_row_text("f" * 64) != text
    moved = rails.recipe_document()
    moved["stop_rule"]["max_iterations"] = 4
    assert hashlib.sha256(rails._json_bytes(moved)).hexdigest() != RECIPE_SHA


# ---- (2) the bank-release registration -------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def release():
    return rails.bank_release_document(REPO)


def test_the_bank_release_registration_is_well_formed(release):
    assert rails.validate_bank_release(release) == []
    entry = release["iterations"][0]
    assert entry["iteration"] == 1 and sorted(entry["releases"]) == ["half-R01C-H1", "half-R01C-H2", "union"]
    union = entry["releases"]["union"]
    assert union["source_halves"] == ["R01C-H1", "R01C-H2"] and union["consuming_roster"] == "FOLD-B"
    assert union["consuming"]["cell_ids"] == [tf.CELLS["fsmoke"]["cell"], tf.CELLS["fb"]["cell"]] and union["union_hash"]["sha256"] == "0" * 64
    assert union["measure"]["required"] == "vllm-tokenize" and "estimate" in union["measure"]["refused"]
    assert union["token_counts"]["exemplar_block_token_cap"] == 2000 and union["token_counts"]["rule_bank_token_cap"] == 400
    half = entry["releases"]["half-R01C-H1"]
    assert half["source_halves"] == ["R01C-H1"] and half["consuming_roster"] == "R01C-H2" and half["source_split_identities"] == {"R01C-H1": tf.CELLS["ch1"]["split_identity_hash"]}
    assert half["consuming"]["planned_run_id"] == "SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-I2-H2-NPU-PORT-V1"
    assert release["measure"]["required"] == "vllm-tokenize" and release["caps"]["exemplars_replace_supplied_items"] == 8 and release["max_iterations"] == 3


def test_the_release_names_the_t0_rule_and_proves_the_forbidden_rosters_disjoint(release):
    rule = release["t0_never_a_bank_source"]
    assert "T0, the official SLURP test split, is never a source" in rule["rule"] and rule["allowed_source_halves"] == ["R01C-H1", "R01C-H2"]
    forbidden = rule["forbidden_rosters"]
    assert sorted(forbidden) == ["DEV-1000", "Fold A v2", "Fold B v2", "T0 official SLURP test split"]
    assert all(forbidden[name]["rows_shared_with_r01c"] == 0 for name in ("DEV-1000", "Fold A v2", "Fold B v2")) and forbidden["T0 official SLURP test split"]["roster"] is None
    assert rule["proofs"] == {"r01c_rows": 625, "halves_union_equals_r01c": True, "fold_b_rows_shared_with_r01c": 0}
    assert rule["allowed_source_identities"] == {"R01C-H1": tf.CELLS["ch1"]["split_identity_hash"], "R01C-H2": tf.CELLS["ch2"]["split_identity_hash"]}


def test_every_recorded_consumption_frame_parses_and_is_pending(release):
    frames = release["iterations"][0]["tfrl_v2_bank_frames"]
    assert sorted(frames) == sorted([tf.CELLS["fsmoke"]["cell"], tf.CELLS["fb"]["cell"]])
    for frame in frames.values():
        decl = rail.parse_bank_declaration(frame)
        assert decl.pending and decl.consumer == "FOLD-B" and decl.bank_kind == "union" and decl.measure == "vllm-tokenize" and decl.scope == rule_identities(release)


def rule_identities(release):
    return release["t0_never_a_bank_source"]["allowed_source_identities"]


@pytest.mark.parametrize("mutate,expected", [
    (lambda d: d.update(schema="other"), "schema is not"),
    (lambda d: d["t0_never_a_bank_source"].update(rule="no rule"), "the T0 rule is missing"),
    (lambda d: d["t0_never_a_bank_source"].update(allowed_source_halves=["R01C-H1", "FOLD-B"]), "the T0 rule is missing"),
    (lambda d: d["t0_never_a_bank_source"]["forbidden_rosters"].pop("T0 official SLURP test split"), "do not name the T0 official SLURP test split"),
    (lambda d: d["t0_never_a_bank_source"]["forbidden_rosters"]["DEV-1000"].update(rows_shared_with_r01c=3), "shares rows with the construction column"),
    (lambda d: d["measure"].update(required="estimate"), "the measure must require vllm-tokenize and refuse estimate"),
    (lambda d: d["measure"].update(refused=[]), "the measure must require vllm-tokenize and refuse estimate"),
    (lambda d: d["iterations"][0]["releases"]["union"]["measure"].update(required="estimate"), "the measure is not vllm-tokenize"),
    (lambda d: d["iterations"][0]["releases"]["half-R01C-H1"].update(consuming_roster="R01C-H1"), "a source half is the consuming roster"),
    (lambda d: d["iterations"][0]["releases"]["union"]["bank_receipt"].update(sha256="short"), "the bank receipt hash is not a sha256"),
    (lambda d: d["iterations"][0]["releases"]["union"]["token_counts"].update(exemplar_block_token_cap=2500), "the token caps are not 2,000 / 400"),
    (lambda d: d["iterations"][0]["releases"]["union"]["consuming"].update(cell_ids=[]), "names no consuming cell"),
    (lambda d: d["iterations"][0]["releases"]["union"].update(source_halves=["R01C-H1"]), "the union bank must name both halves"),
    (lambda d: d["iterations"][0]["releases"].pop("union"), "the releases must be the two half banks and the union"),
    (lambda d: d["iterations"][0]["tfrl_v2_bank_frames"][tf.CELLS["fb"]["cell"]].update(measure="estimate"), "does not parse"),
    (lambda d: d.update(iterations=[]), "no iteration is registered"),
])
def test_each_structural_defect_of_a_release_is_reported(release, mutate, expected):
    doc = copy.deepcopy(release)
    mutate(doc)
    assert any(expected in problem for problem in rails.validate_bank_release(doc)), rails.validate_bank_release(doc)


def test_the_release_is_deterministic_and_english_only(release):
    assert json.dumps(rails.bank_release_document(REPO), sort_keys=True) == json.dumps(release, sort_keys=True)
    assert not CJK.search(json.dumps(release, ensure_ascii=False)) and not CJK.search(rails.recipe_bytes().decode("utf-8"))


# ---- (3) the marker ---------------------------------------------------------------------------------------------------------------------------------


def test_the_marker_is_pending_until_all_three_deliverables_exist_on_disk(rails_root):
    state = rails.rails_status(rails_root)
    assert state["marker"] == "rails_pending"
    assert state["deliverables"]["stage3_label_and_consumption_rail"]["ok"] is True                           # the code deliverables are in the tree
    assert state["deliverables"]["analysis_recipe_and_reader"]["ok"] is False and state["deliverables"]["bank_release_registration"]["problems"] == ["file is missing"]
    shas = rails.write_rails_files(rails_root, reg.wjson, reg.wr)
    assert shas[rails.RECIPE_NAME] == RECIPE_SHA and sha(rails_root / rails.RECIPE_RELPATH) == RECIPE_SHA
    state = rails.rails_status(rails_root)
    assert state["marker"] == "rails_declared" and all(item["ok"] for item in state["deliverables"].values())
    assert state["deliverables"]["analysis_recipe_and_reader"]["recipe_sha256"] == RECIPE_SHA and state["deliverables"]["analysis_recipe_and_reader"]["recipe_sha_pinned_in_read_row"] is True


def test_a_missing_or_moved_deliverable_keeps_the_marker_pending(rails_root, monkeypatch):
    rails.write_rails_files(rails_root, reg.wjson, reg.wr)
    assert rails.rails_status(rails_root)["marker"] == "rails_declared"
    release_path = rails_root / rails.BANK_RELEASE_RELPATH
    original = release_path.read_bytes()
    release_path.unlink()
    assert rails.rails_status(rails_root)["marker"] == "rails_pending"
    broken = json.loads(original)
    broken["measure"]["required"] = "estimate"
    release_path.write_text(json.dumps(broken), encoding="utf-8")
    assert rails.rails_status(rails_root)["marker"] == "rails_pending"
    release_path.write_bytes(original)
    assert rails.rails_status(rails_root)["marker"] == "rails_declared"
    recipe_path = rails_root / rails.RECIPE_RELPATH
    recipe_path.write_bytes(recipe_path.read_bytes() + b" ")                                                       # the recipe moved but the READ row still pins the old sha
    state = rails.rails_status(rails_root)
    assert state["marker"] == "rails_pending" and state["deliverables"]["analysis_recipe_and_reader"]["ok"] is False
    recipe_path.write_bytes(rails.recipe_bytes())
    row_path = rails_root / rails.READ_ROW_RELPATH
    row_path.write_text(row_path.read_text(encoding="utf-8").replace(RECIPE_SHA, "0" * 64), encoding="utf-8")       # the pin is not the recipe's
    assert rails.rails_status(rails_root)["marker"] == "rails_pending"
    row_path.unlink()
    assert rails.rails_status(rails_root)["marker"] == "rails_pending"
    rails.write_rails_files(rails_root, reg.wjson, reg.wr)
    assert rails.rails_status(rails_root)["marker"] == "rails_declared"
    monkeypatch.setattr(rails, "REPO_ROOT", rails_root / "elsewhere")                                               # no reader and no rail file in the code tree
    state = rails.rails_status(rails_root)
    assert state["marker"] == "rails_pending" and state["deliverables"]["analysis_recipe_and_reader"]["reader_file"] is False
    assert state["deliverables"]["stage3_label_and_consumption_rail"]["rail_file"] is False
    monkeypatch.undo()
    monkeypatch.setattr(s3, "resolve_stage3_algorithm", lambda name: (_ for _ in ()).throw(s3.Stage3AlgorithmError("unregistered")))
    state = rails.rails_status(rails_root)
    assert state["marker"] == "rails_pending" and state["deliverables"]["stage3_label_and_consumption_rail"]["label_registered"] is False


# ---- (4) the writer ------------------------------------------------------------------------------------------------------------------------------------


@pytest.fixture()
def written(tmp_path):
    if not (DATA_ROOT / "prepared").is_dir():
        pytest.skip("the data root with the port preparation receipts is not mounted")
    root = make_t_root(tmp_path / "root")
    result = tw.write_package(root, COMMIT, AT, DATA_ROOT)
    return root, result


@needs_data
def test_the_writer_emits_the_three_files_the_pinned_recipe_sha_and_the_declared_marker(written):
    root, result = written
    package = root / tf.PACKAGE
    for name in (rails.RECIPE_NAME, rails.READ_ROW_NAME, rails.BANK_RELEASE_NAME):
        assert (package / name).is_file() and result["shas"][name] == sha(package / name)
    assert result["shas"][rails.RECIPE_NAME] == RECIPE_SHA
    facts = json.loads((package / "registered-facts-v1.json").read_text(encoding="utf-8"))
    assert facts["rails_status"]["marker"] == "rails_declared" and all(i["ok"] for i in facts["rails_status"]["deliverables"].values())
    recipe = facts["read_rule"]["analysis_recipe"]
    assert recipe["sha256"] == RECIPE_SHA and recipe["path"] == rails.RECIPE_RELPATH and recipe["read_row_draft"]["sha256"] == result["shas"][rails.READ_ROW_NAME] and recipe["reader"] == rails.READER
    assert facts["bank_release"] == {"path": rails.BANK_RELEASE_RELPATH, "sha256": result["shas"][rails.BANK_RELEASE_NAME]}
    assert facts["read_rule"]["bootstrap"]["seed"] == 20261018 and facts["rails_pending"]["fold-b"]                       # the per-cell items stay listed: the cells have not adopted the rails
    assert f"`{RECIPE_SHA}`" in (package / rails.READ_ROW_NAME).read_text(encoding="utf-8")
    readme = (package / "README.md").read_text(encoding="utf-8")
    assert "## Rails status: `rails_declared`" in readme and rails.RECIPE_NAME in readme and rails.BANK_RELEASE_NAME in readme
    for path in package.iterdir():
        assert not CJK.search(path.read_text(encoding="utf-8")), path


@needs_data
def test_the_writer_keeps_the_marker_pending_when_a_deliverable_is_defective(tmp_path, monkeypatch):
    root = make_t_root(tmp_path / "root")
    monkeypatch.setattr(rails, "validate_bank_release", lambda doc: ["forced defect"])
    tw.write_package(root, COMMIT, AT, DATA_ROOT)
    facts = json.loads((root / tf.PACKAGE / "registered-facts-v1.json").read_text(encoding="utf-8"))
    assert facts["rails_status"]["marker"] == "rails_pending" and facts["rails_status"]["deliverables"]["bank_release_registration"]["problems"] == ["forced defect"]
    assert "## Rails status: `rails_pending`" in (root / tf.PACKAGE / "README.md").read_text(encoding="utf-8")


@needs_data
def test_the_writer_is_idempotent_with_the_rails_files(written):
    root, result = written
    before = {p.name: sha(p) for p in (root / tf.PACKAGE).iterdir()}
    again = tw.write_package(root, COMMIT, None, DATA_ROOT)
    assert again["shas"] == result["shas"] and {p.name: sha(p) for p in (root / tf.PACKAGE).iterdir()} == before
