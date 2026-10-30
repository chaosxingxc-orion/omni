"""The TFRL-v2 consumption rail (``rails/tfrl_v2_consumption.py``), the ``tfrl-v2`` stage-3 label and the additive module-block frame.

Model-free fixtures. The proofs: (1) the label resolves and validates, and a label cannot claim an operator the record does not define; (2) the ``tfrl_v2_bank`` declaration
parses fail-closed (caps, measure, consumer, scope); (3) the flight-start refusals, with their exact messages: a receipt hash that differs from the declaration, the PENDING zero
sentinel, a source half that is the consuming roster, a source outside the construction halves (a T0 / Fold A / Fold B row), an estimate measure, an overlap of rows or split
identity, a cap overrun; (4) a clean bank loads and every file is re-hashed; (5) a module block with no frame is byte for byte what it was (block hash, compiled blocks, loader
check) and the Solver messages an existing stage-3 config renders are a pinned snapshot that an empty bank leaves unchanged; (6) ``used_basis`` is read from the trace.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
for _extra in (REPO / "scripts" / "flights" / "nx", REPO / "scripts" / "analysis", REPO / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import tfrl_v2_common as cmn  # noqa: E402

from speech_aware_evidence_acquisition.core import stage3_algorithms as s3  # noqa: E402
from speech_aware_evidence_acquisition.rails import consumption_module as cm  # noqa: E402
from speech_aware_evidence_acquisition.rails import tfrl_v2_consumption as rail  # noqa: E402
from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks  # noqa: E402
from speech_aware_evidence_acquisition.rails.knowledge import QueryState  # noqa: E402
from speech_aware_evidence_acquisition.rails.runner import _agent_messages  # noqa: E402

H1, H2 = "R01C-H1", "R01C-H2"
ID1, ID2 = "1" * 64, "2" * 64
SCOPE = {H1: ID1, H2: ID2}
FOLD_B_ID = "b" * 64
EXISTING = "slurp-native-stage3-thinking-system-fold-b-lock-baseline-v1"          # an existing stage-3 baseline that declares a module block
#: Pinned bytes of the existing config and of what it renders (the label's "no existing label moves" proof).
EXISTING_FILE_SHA = "5f0d506f2eb906af54f34acfae4e0b9a1c328433b08c676f107fe25dc807215e"
EXISTING_BLOCK_SHA = "00fdfa368b32bc143e8ce9355f031e5010029a8d2a75527975a205555d988d6d"
EXISTING_COMPILED_SHA = "194e8ec89179f6c03f0eec9d9e6f1f36fb13a29a7994708d712ac844e8ee7c6f"
SOLVER_TEMPLATE_SHA = "18bff5c557cac7ac14be6b08aa0a1a6ad54c1b9820199c3b6749b2271d756fbd"
RENDERED_SNAPSHOT_SHA = "f6fed2336bb79628e2ee7469e78bf88b7f35e3300a5d9a49c024c7414fb1f8b9"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---- a clean bank on disk ----------------------------------------------------------------------------------------------------------------------------


def make_bank(root: Path, *, kind="union", sources=(H1, H2), measure="vllm-tokenize", exemplar_text="resolved", iteration=1, block_tokens=1800, rule_tokens=300,
              identities=None, source_rows=("r1", "r2")) -> tuple[Path, dict]:
    """Write a bank directory shaped like the distiller's; return (receipt path relative to ``root``, receipt)."""

    out = root / "banks" / kind
    out.mkdir(parents=True)
    exemplar = {"schema": "saea-tfrl-v2-exemplar-bank-v1", "iteration": iteration, "by_scenario": {"alarm": {"entries": [{"text": "x"}], "block_text": "Worked examples for alarm requests:\n- x", "block_tokens": block_tokens}}}
    rules = {"schema": "saea-tfrl-v2-rule-bank-v1", "iteration": iteration, "rules": [], "rendered_text": "Consumption rules:", "rendered_tokens": rule_tokens}
    files = {"exemplar-bank.json": cmn.file_bytes(exemplar), "rule-bank.json": cmn.file_bytes(rules)}
    for name, data in files.items():
        (out / name).write_bytes(data)
    receipt = {
        "schema": "saea-tfrl-v2-bank-receipt-v1", "iteration": iteration, "kind": kind, "measure": measure, "exemplar_text": exemplar_text,
        "source_halves": list(sources), "source_split_identities": identities or {h: SCOPE[h] for h in sources}, "source_row_ids": list(source_rows),
        "exemplar_bank": {"content_sha256": cmn.content_sha256(exemplar), "file": "exemplar-bank.json", "max_block_tokens": block_tokens},
        "rule_bank": {"content_sha256": cmn.content_sha256(rules), "file": "rule-bank.json", "rendered_tokens": rule_tokens},
        "files": {name: {"sha256": sha(data), "bytes": len(data)} for name, data in files.items()},
    }
    (out / "bank-receipt.json").write_bytes(cmn.file_bytes(receipt))
    return out / "bank-receipt.json", receipt


def declaration(receipt: dict, *, consumer="FOLD-B", kind="union", relpath="banks/union/bank-receipt.json", pin=None) -> rail.BankDeclaration:
    block = rail.declaration_block(iteration=1, consumer=consumer, bank_kind=kind, receipt_relpath=relpath, receipt_sha=pin or rail.receipt_sha256(receipt), source_scope=SCOPE)
    return rail.parse_bank_declaration(block)


# ---- (1) the label ---------------------------------------------------------------------------------------------------------------------------------


def test_the_tfrl_v2_label_resolves_and_cannot_claim_an_operator_it_does_not_define():
    record = s3.resolve_stage3_algorithm("tfrl-v2")
    assert record.family == "training-free-rl-v2" and record.status == "planned" and set(record.operators) <= set(s3.STAGE3_OPERATORS)
    assert "rails/tfrl_v2_consumption.py" in record.citation
    label = {"id": "tfrl-v2", "operators_used": ["rollout", "scoring", "context_application"], "note": "T cell"}
    assert s3.validate_stage3_algorithm_label(label, cell="t")[1] == ("rollout", "scoring", "context_application")
    with pytest.raises(s3.Stage3AlgorithmError, match="does not define"):
        s3.validate_stage3_algorithm_label(dict(label, operators_used=["bogus"]), cell="t")
    with pytest.raises(s3.Stage3AlgorithmError, match="states which of that algorithm's operators"):
        s3.validate_stage3_algorithm_label(dict(label, operators_used=[]), cell="t")


def test_no_existing_experiment_label_or_baseline_moved():
    """The new label is registered but no committed fragment wears it, and no committed baseline carries the new module-block key."""

    for path in (REPO / "configs" / "experiment").glob("*.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert (doc.get("stage3_algorithm") or {}).get("id") != "tfrl-v2", path.name
    for path in (REPO / "configs" / "baseline").glob("*.json"):
        assert "tfrl_v2_bank" not in path.read_text(encoding="utf-8"), path.name


# ---- (2) the declaration ---------------------------------------------------------------------------------------------------------------------------


def test_a_declaration_parses_and_carries_the_registered_caps():
    block = rail.declaration_block(iteration=2, consumer=H2, bank_kind="half", receipt_relpath="a/b/bank-receipt.json", source_scope=SCOPE)
    decl = rail.parse_bank_declaration(block)
    assert decl.pending and decl.exemplar_token_cap == 2000 and decl.rule_token_cap == 400 and decl.exemplar_replaces_items == 8 and decl.measure == "vllm-tokenize"
    assert block["used_basis"] == "trace"


@pytest.mark.parametrize("mutate,match", [
    (lambda b: b.update(measure="estimate"), "bank whose token counts are estimates cannot be pinned"),
    (lambda b: b["exemplar_block"].update(token_cap=2001), "exemplar_block.token_cap must be an integer in 1..2000"),
    (lambda b: b["rule_bank"].update(token_cap=401), "rule_bank.token_cap must be an integer in 1..400"),
    (lambda b: b["exemplar_block"].update(replaces_supplied_items=7), "must be 8"),
    (lambda b: b["exemplar_block"].update(text="ids-only"), "text must be 'resolved'"),
    (lambda b: b["rule_bank"].update(delivery="user-message"), "Solver instruction text"),
    (lambda b: b.update(consumer="R01C-H1"), "the union bank is consumed by"),
    (lambda b: b.update(used_basis="writer"), "read from the rollout trace"),
    (lambda b: b.update(iteration=4), "iteration must be an integer in 1..3"),
    (lambda b: b.update(extra=1), "exactly the keys"),
    (lambda b: b["bank_receipt"].update(sha256="xyz"), "lowercase 64-hex"),
    (lambda b: b["bank_receipt"].update(relpath="../x"), "data-root-relative"),
    (lambda b: b.update(source_scope={H1: ID1}), "must map exactly the construction halves"),
    (lambda b: b.update(algorithm_id="tfgrpo-semantic-advantage-v1"), "algorithm_id must be 'tfrl-v2'"),
])
def test_a_malformed_declaration_is_refused_by_name(mutate, match):
    block = rail.declaration_block(iteration=1, consumer="FOLD-B", bank_kind="union", receipt_relpath="banks/union/bank-receipt.json", source_scope=SCOPE)
    mutate(block)
    with pytest.raises(rail.TfrlV2BankError, match=match):
        rail.parse_bank_declaration(block)


def test_a_half_bank_is_consumed_only_by_a_construction_half():
    with pytest.raises(rail.TfrlV2BankError, match="a half bank is consumed by a construction half"):
        rail.parse_bank_declaration(rail.declaration_block(iteration=1, consumer="FOLD-B", bank_kind="half", receipt_relpath="x/bank-receipt.json", source_scope=SCOPE))


# ---- (3) the flight-start refusals -------------------------------------------------------------------------------------------------------------


def test_a_clean_union_bank_is_accepted_and_returns_its_receipt_hash(tmp_path):
    _, receipt = make_bank(tmp_path)
    decl = declaration(receipt)
    assert rail.verify_bank(decl, receipt, consuming_id="FOLD-B", consuming_row_ids=["f1", "f2"], consuming_split_identity=FOLD_B_ID) == rail.receipt_sha256(receipt)


def test_the_pending_zero_sentinel_is_refused(tmp_path):
    _, receipt = make_bank(tmp_path)
    block = rail.declaration_block(iteration=1, consumer="FOLD-B", bank_kind="union", receipt_relpath="banks/union/bank-receipt.json", source_scope=SCOPE)
    with pytest.raises(rail.TfrlV2BankRefusal, match="is the PENDING zero sentinel; release the bank and pin its receipt hash " + rail.receipt_sha256(receipt)):
        rail.verify_bank(rail.parse_bank_declaration(block), receipt, consuming_id="FOLD-B")


def test_a_receipt_hash_that_differs_from_the_declaration_is_refused(tmp_path):
    _, receipt = make_bank(tmp_path)
    decl = declaration(receipt, pin="a" * 64)
    with pytest.raises(rail.TfrlV2BankRefusal, match="the bank receipt hash " + rail.receipt_sha256(receipt) + " differs from the hash " + "a" * 64 + " pinned in the module declaration"):
        rail.verify_bank(decl, receipt, consuming_id="FOLD-B")
    tampered = copy.deepcopy(receipt)
    tampered["source_row_ids"].append("sneaky")
    with pytest.raises(rail.TfrlV2BankRefusal, match="differs from the hash"):
        rail.verify_bank(declaration(receipt), tampered, consuming_id="FOLD-B")


def test_a_bank_whose_sources_include_the_consuming_roster_is_refused(tmp_path):
    _, receipt = make_bank(tmp_path, kind="half", sources=(H2,))
    decl = declaration(receipt, consumer=H2, kind="half", relpath="banks/half/bank-receipt.json")
    with pytest.raises(rail.TfrlV2BankRefusal, match=r"the bank was distilled from \['R01C-H2'\], which includes the consuming roster R01C-H2"):
        rail.verify_bank(decl, receipt, consuming_id=H2)
    _, other = make_bank(tmp_path / "o", kind="half", sources=(H1,))
    assert rail.verify_bank(declaration(other, consumer=H2, kind="half", relpath="banks/half/bank-receipt.json"), other, consuming_id=H2) == rail.receipt_sha256(other)


def test_a_union_bank_cannot_be_consumed_by_either_half():
    block = rail.declaration_block(iteration=1, consumer="FOLD-B", bank_kind="union", receipt_relpath="x/bank-receipt.json", source_scope=SCOPE)
    block["consumer"] = H1
    with pytest.raises(rail.TfrlV2BankError):
        rail.parse_bank_declaration(block)


def test_a_source_outside_the_construction_halves_is_refused_t0_never_a_source(tmp_path):
    _, receipt = make_bank(tmp_path, sources=(H1, "T0-OFFICIAL-TEST"), identities={H1: ID1, "T0-OFFICIAL-TEST": "9" * 64})
    with pytest.raises(rail.TfrlV2BankRefusal, match=r"bank source 'T0-OFFICIAL-TEST' is outside the declared construction scope \['R01C-H1', 'R01C-H2'\]; test-split \(T0\) rows"):
        rail.verify_bank(declaration(receipt), receipt, consuming_id="FOLD-B")
    _, wrong_identity = make_bank(tmp_path / "w", identities={H1: ID1, H2: "3" * 64})
    with pytest.raises(rail.TfrlV2BankRefusal, match="the split identity of bank source R01C-H2 is not the pinned construction identity"):
        rail.verify_bank(declaration(wrong_identity), wrong_identity, consuming_id="FOLD-B")


def test_an_estimate_measured_bank_is_refused(tmp_path):
    _, receipt = make_bank(tmp_path, measure="estimate")
    with pytest.raises(rail.TfrlV2BankRefusal, match="measure 'estimate', not 'vllm-tokenize'"):
        rail.verify_bank(declaration(receipt), receipt, consuming_id="FOLD-B")
    _, ids_only = make_bank(tmp_path / "i", exemplar_text="ids-only")
    with pytest.raises(rail.TfrlV2BankRefusal, match="exemplar_text must be 'resolved'"):
        rail.verify_bank(declaration(ids_only), ids_only, consuming_id="FOLD-B")


def test_row_and_identity_overlap_with_the_consumer_and_cap_overruns_are_refused(tmp_path):
    _, receipt = make_bank(tmp_path)
    decl = declaration(receipt)
    with pytest.raises(rail.TfrlV2BankRefusal, match=r"1 consuming rows are source rows of the bank \(first: r2\)"):
        rail.verify_bank(decl, receipt, consuming_id="FOLD-B", consuming_row_ids=["r2", "f9"])
    with pytest.raises(rail.TfrlV2BankRefusal, match="consuming split identity equals a source split identity"):
        rail.verify_bank(decl, receipt, consuming_id="FOLD-B", consuming_split_identity=ID1)
    with pytest.raises(rail.TfrlV2BankRefusal, match="the consuming roster is 'FOLD-A', the declaration names consumer 'FOLD-B'"):
        rail.verify_bank(decl, receipt, consuming_id="FOLD-A")
    _, big = make_bank(tmp_path / "b", block_tokens=2001)
    with pytest.raises(rail.TfrlV2BankRefusal, match="largest exemplar block is 2001 tokens, over the declared cap 2000"):
        rail.verify_bank(declaration(big), big, consuming_id="FOLD-B")
    _, long_rules = make_bank(tmp_path / "r", rule_tokens=401)
    with pytest.raises(rail.TfrlV2BankRefusal, match="rendered rule bank is 401 tokens, over the declared cap 400"):
        rail.verify_bank(declaration(long_rules), long_rules, consuming_id="FOLD-B")


def test_iteration_kind_and_schema_mismatches_are_refused(tmp_path):
    _, receipt = make_bank(tmp_path, iteration=2)
    with pytest.raises(rail.TfrlV2BankRefusal, match="bank is of iteration 2, the declaration is of iteration 1"):
        rail.verify_bank(declaration(receipt), receipt, consuming_id="FOLD-B")
    _, half = make_bank(tmp_path / "h", kind="half", sources=(H1,))
    with pytest.raises(rail.TfrlV2BankRefusal, match="bank is a 'half' bank, the declaration consumes a 'union' bank"):
        rail.verify_bank(declaration(half), half, consuming_id="FOLD-B")
    with pytest.raises(rail.TfrlV2BankRefusal, match="receipt schema is not"):
        rail.verify_bank(declaration(half), {"schema": "other"}, consuming_id="FOLD-B")


# ---- (4) the bank on disk --------------------------------------------------------------------------------------------------------------------------


def test_verify_bank_dir_loads_a_clean_bank_and_rehashes_every_file(tmp_path):
    path, receipt = make_bank(tmp_path)
    decl = declaration(receipt)
    loaded = rail.verify_bank_dir(decl, tmp_path, consuming_id="FOLD-B", consuming_split_identity=FOLD_B_ID)
    assert loaded.receipt_sha256 == rail.receipt_sha256(receipt)
    assert loaded.exemplar_block_text("alarm").startswith("Worked examples for alarm") and loaded.exemplar_block_text("weather") == "" and loaded.rule_text() == "Consumption rules:"
    (path.parent / "rule-bank.json").write_bytes(b"{}\n")
    with pytest.raises(rail.TfrlV2BankRefusal, match="the file hash of rule-bank.json differs from the receipt"):
        rail.verify_bank_dir(decl, tmp_path, consuming_id="FOLD-B")
    missing = declaration(receipt, relpath="banks/gone/bank-receipt.json")
    with pytest.raises(rail.TfrlV2BankRefusal, match="cannot be read"):
        rail.verify_bank_dir(missing, tmp_path, consuming_id="FOLD-B")


def _values_with(frame):
    block = json.loads(json.dumps(json.loads((REPO / "configs" / "baseline" / f"{EXISTING}.json").read_text(encoding="utf-8"))["native_episode"]["consumption_module"]))
    if frame is not None:
        block["tfrl_v2_bank"] = frame
    return {"native_episode": {"consumption_module": block}}


def test_verify_for_flight_is_a_no_op_without_a_frame_and_checks_the_declared_consumer(tmp_path):
    assert rail.verify_for_flight(_values_with(None), tmp_path, split_identity_hash=FOLD_B_ID) is None
    _, receipt = make_bank(tmp_path)
    frame = rail.declaration_block(iteration=1, consumer="FOLD-B", bank_kind="union", receipt_relpath="banks/union/bank-receipt.json", receipt_sha=rail.receipt_sha256(receipt), source_scope=SCOPE)
    assert rail.verify_for_flight(_values_with(frame), tmp_path, split_identity_hash=FOLD_B_ID).receipt_sha256 == rail.receipt_sha256(receipt)
    with pytest.raises(rail.TfrlV2BankRefusal, match="consuming split identity equals a source split identity"):
        rail.verify_for_flight(_values_with(frame), tmp_path, split_identity_hash=ID2)
    _, half = make_bank(tmp_path / "h", kind="half", sources=(H1,))
    half_frame = rail.declaration_block(iteration=1, consumer=H2, bank_kind="half", receipt_relpath="banks/half/bank-receipt.json", receipt_sha=rail.receipt_sha256(half), source_scope=SCOPE)
    assert rail.verify_for_flight(_values_with(half_frame), tmp_path / "h", split_identity_hash=ID2).receipt_sha256 == rail.receipt_sha256(half)
    with pytest.raises(rail.TfrlV2BankRefusal, match="declares consumer R01C-H2 but its split identity is not the pinned identity of that half"):
        rail.verify_for_flight(_values_with(half_frame), tmp_path / "h", split_identity_hash=FOLD_B_ID)


# ---- (5) the module block: no frame is byte for byte what it was ---------------------------------------------------------------------------------


def test_the_module_block_without_the_frame_is_unchanged_and_with_the_frame_compiles_to_the_same_policy():
    path = REPO / "configs" / "baseline" / f"{EXISTING}.json"
    assert sha(path.read_bytes()) == EXISTING_FILE_SHA
    doc = json.loads(path.read_text(encoding="utf-8"))
    block = doc["native_episode"]["consumption_module"]
    parsed = cm.parse_consumption_module_block(block, repo_root=REPO)
    assert parsed.block_sha256 == EXISTING_BLOCK_SHA
    hooks = carrier_module_hooks(repo_root=REPO)
    assert sha(cm.compile_module(parsed, hooks, repo_root=REPO).blocks_json.encode()) == EXISTING_COMPILED_SHA
    assert cm.verify_declared_module(block, doc, repo_root=REPO) == parsed.block_json
    framed = dict(block, tfrl_v2_bank=rail.declaration_block(iteration=1, consumer="FOLD-B", bank_kind="union", receipt_relpath="x/bank-receipt.json", source_scope=SCOPE))
    parsed_framed = cm.parse_consumption_module_block(framed, repo_root=REPO)
    assert parsed_framed.block_sha256 != EXISTING_BLOCK_SHA                                    # the declaration is hashed into the block
    assert cm.compile_module(parsed_framed, hooks, repo_root=REPO).blocks_json == cm.compile_module(parsed, hooks, repo_root=REPO).blocks_json
    assert cm.verify_declared_module(framed, doc, repo_root=REPO) == parsed_framed.block_json


def test_a_malformed_frame_in_a_module_block_is_a_config_error():
    doc = json.loads((REPO / "configs" / "baseline" / f"{EXISTING}.json").read_text(encoding="utf-8"))
    block = dict(doc["native_episode"]["consumption_module"], tfrl_v2_bank={"schema": "wrong"})
    with pytest.raises(cm.ConsumptionModuleConfigError, match="tfrl_v2_bank must be an object with exactly the keys"):
        cm.parse_consumption_module_block(block, repo_root=REPO)


def _render_existing_solver_messages() -> tuple:
    doc = json.loads((REPO / "configs" / "baseline" / f"{EXISTING}.json").read_text(encoding="utf-8"))
    template = (REPO / "configs" / doc["phase_material"]["solver"]["instruction_template"]).read_text(encoding="utf-8")
    assert sha(template.encode("utf-8")) == SOLVER_TEMPLATE_SHA
    adapter = SimpleNamespace(assemble_prompt=lambda sample, _history: ({"role": "system", "content": template}, {"role": "user", "content": []}))
    snapshot = SimpleNamespace(facets=(), values=(SimpleNamespace(value_id="train-exemplars::value:1", content="wake me up at nine | alarm_set"),
                                                  SimpleNamespace(value_id="train-exemplars::value:2", content="what is the weather | weather_query")))
    query = QueryState(query_id="q", query_sample_id="s", speech_packet_id="packet", speech_packet_sha256="b" * 64, use_split_identity_hash="c" * 64,
                       retrieved_value_ids=("train-exemplars::value:1", "train-exemplars::value:2"), query_source="speech_audio")
    return _agent_messages(adapter=adapter, sample=SimpleNamespace(sample_id="s"), task_contract="CONTRACT", snapshot=snapshot, snapshot_sha256="d" * 64, query_state=query)


def test_the_rendered_solver_messages_of_an_existing_stage3_config_are_a_pinned_snapshot_and_an_empty_bank_leaves_them_unchanged():
    messages = _render_existing_solver_messages()
    digest = sha(json.dumps(messages, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    assert digest == RENDERED_SNAPSHOT_SHA, digest
    system, user = rail.compose_solver_messages(messages)
    assert (system, user) == tuple(messages) and system is not messages[0]


def test_the_bank_adds_a_system_suffix_and_one_unlabelled_text_part_and_never_mutates_its_input():
    messages = _render_existing_solver_messages()
    before = json.dumps(messages, sort_keys=True)
    system, user = rail.compose_solver_messages(messages, exemplar_block_text="Worked examples for alarm requests:\n- x", rule_text="Consumption rules:\n- r")
    assert json.dumps(messages, sort_keys=True) == before
    assert system["content"] == messages[0]["content"] + "\n\nConsumption rules:\n- r"
    assert user["content"][:-1] == list(messages[1]["content"]) and user["content"][-1] == {"type": "text", "text": "Worked examples for alarm requests:\n- x", "evidence_id": "tfrl-v2/exemplar-block"}
    assert "knowledge_role" not in user["content"][-1]                                          # SUPPLY accounting reads key / value parts only
    with pytest.raises(rail.TfrlV2BankError, match="exactly the system and user messages"):
        rail.compose_solver_messages(messages[:1])


# ---- (6) used_basis from the trace --------------------------------------------------------------------------------------------------------------


def _contact(node, items):
    return {"kind": "contact", "node": node, "supply_items": items}


def _item(i, used, basis):
    return {"item_id": f"s::{i}", "rank": i, "score": 0.5, "used": used, "used_basis": basis}


def test_used_basis_is_read_from_the_trace_and_a_missing_basis_is_refused():
    lines = [_contact("solver", [_item(1, True, "content-pipe-form"), _item(2, False, "content-pipe-form"), _item(3, None, "none")]),
             _contact("solver", [_item(1, True, "json-form")]), _contact("perception", [_item(1, None, "none")])]
    summary = rail.used_basis_summary(lines)
    assert summary["contacts"] == 2 and summary["supply_items"] == 4 and summary["by_basis"] == {"json-form": 1, "content-pipe-form": 2, "none": 1}
    assert summary["by_used"] == {"true": 2, "false": 1, "null": 1} and summary["judged_items"] == 3 and summary["judged_share"] == 0.75 and summary["source"] == "trace"
    with pytest.raises(rail.TfrlV2BankError, match="without a closed used_basis"):
        rail.used_basis_summary([_contact("solver", [{"item_id": "a", "rank": 1, "score": None, "used": True}])])
