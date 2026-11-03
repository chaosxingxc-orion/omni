"""Functional contracts for the opt-in CPU knowledge context node."""
from dataclasses import replace
import json

import pytest
from test_llama_server_transport import _spawn_live_matching_process  # noqa: F401

from speech_aware_evidence_acquisition.rails.knowledge import KeyFacet, KnowledgeSnapshot, ValueBundle
from speech_aware_evidence_acquisition.rails.knowledge_context import (
    SCHEMA, prepare_content, prepare_messages, validate_config,
)


def config(mode="strict-v1", observability="debug"):
    return {"schema": SCHEMA, "mode": mode, "observability": observability}


def snapshot(*items):
    return KnowledgeSnapshot("test-snapshot", "1" * 64, ("train:a",), "test-builder",
        (KeyFacet("intent:a", "label", "text", "intent a", ("train:a",)),), tuple(items))


def value(identifier="store::a", **kwargs):
    defaults = dict(content='{"rule":"preserve exact slot"}', associated_facet_ids=("intent:a",),
                    source_sample_ids=("train:a",), provenance_refs=("train://a",), task_target="intent:a", prior_score=0.5)
    defaults.update(kwargs)
    return ValueBundle(identifier, **defaults)


def part(item, **kwargs):
    output = {"type": "text", "text": item.content, "evidence_id": item.value_id,
              "source_uri": f"knowledge-snapshot://{'2' * 64}/{item.value_id}", "knowledge_role": "value"}
    output.update(kwargs)
    return output


def run(items, parts=None, **kwargs):
    return prepare_content(list(map(part, items)) if parts is None else parts,
        snapshot=snapshot(*items), snapshot_sha256="2" * 64,
        config_json=validate_config(config(**kwargs)))


def test_default_and_explicit_legacy_are_identity_and_no_audit():
    items = [part(value())]
    assert validate_config(None) is None
    assert validate_config(config(mode="legacy")) is None
    result, audit = prepare_content(items, snapshot=None, snapshot_sha256=None, config_json=None)
    assert result is items and audit is None


def test_exact_alias_duplicates_keep_first_order_and_complete_provenance():
    first, alias, third = value(), value("store::alias"), value("store::third", content="other")
    media = {"type": "input_audio", "media_relpath": "one.wav"}
    output, audit = run([first, alias, third], [media, part(first), part(alias), part(third)])
    assert output == [media, part(first), part(third)]
    assert (audit["input_count"], audit["output_count"], audit["removed_count"], audit["duplicate_group_count"]) == (3, 2, 1, 1)
    assert [row["evidence_id"] for row in audit["duplicate_groups"][0]] == ["store::a", "store::alias"]
    assert audit["mapping"][1]["representative_id"] == first.value_id
    assert audit["model_calls"] == 0 and audit["elapsed_seconds"] >= 0
    assert audit["after"]["body_utf8_bytes"] < audit["before"]["body_utf8_bytes"]
    assert audit["after"]["exact_text_tokens"] is None


@pytest.mark.parametrize("change", [
    {"task_target": "intent:b"}, {"associated_facet_ids": ("slot:other",)},
    {"source_sample_ids": ("train:b",)}, {"provenance_refs": ("train://b",)},
    {"prior_score": 0.4}, {"content": '{ "rule":"preserve exact slot"}'},
    {"content": '{"rule":"drop exact slot"}'},
])
def test_same_text_scope_source_weight_or_semantic_differences_are_retained(change):
    items = [value(), value("store::b", **change)]
    output, audit = run(items)
    assert output == list(map(part, items)) and audit["removed_count"] == 0


def test_same_body_in_other_store_is_not_merged():
    output, audit = run([value(), value("other::b")])
    assert len(output) == 2
    assert audit["conflicts"]["same_text_distinct_semantics_groups"] == 1


@pytest.mark.parametrize("extra", [{"valid_until": "2027"}, {"version": 2}, {"source_uri": "other://source"}])
def test_runtime_extensions_and_unbound_versions_are_preserved(extra):
    items = [value(), value("store::b")]
    output, audit = run(items, [part(items[0]), part(items[1], **extra)])
    assert len(output) == 2 and audit["removed_count"] == 0


def test_unknown_items_fail_open_and_conflicts_are_not_resolved():
    item = value()
    parts = [part(item, evidence_id="unknown"), part(item, evidence_id="unknown", text="contradiction")]
    output, audit = run([item], parts)
    assert output == parts and audit["unresolved_count"] == 2
    assert audit["conflicts"]["same_id_different_payload_groups"] == 1
    assert audit["conflicts"]["semantic_contradictions"] is None


def test_normal_is_compact_debug_keeps_mapping_but_same_model_content():
    items = [value(), value("store::b")]
    normal, normal_audit = run(items, observability="normal")
    debug, debug_audit = run(items, observability="debug")
    assert normal == debug and normal_audit["mapping_sha256"] == debug_audit["mapping_sha256"]
    assert "mapping" not in normal_audit and len(debug_audit["mapping"]) == 2
    assert "content" not in json.dumps(normal_audit["duplicate_groups"])


def test_stable_output_and_idempotence():
    items = [value(), value("store::b")]
    output, audit = run(items)
    again, second = run(items, output)
    assert again == output and second["removed_count"] == 0
    _, repeated = run(items)
    assert {k:v for k,v in repeated.items() if k != "elapsed_seconds"} == {k:v for k,v in audit.items() if k != "elapsed_seconds"}


def test_messages_preserve_system_and_media():
    items = [value(), value("store::b")]
    system = {"role": "system", "content": "system"}
    source = (system, {"role": "user", "content": list(map(part, items))})
    output, audit = prepare_messages(source, snapshot=snapshot(*items), snapshot_sha256="2" * 64,
                                     config_json=validate_config(config()))
    assert output[0] is system and len(output[1]["content"]) == 1 and audit["removed_count"] == 1


@pytest.mark.parametrize("bad", [{}, {**config(), "mode":"semantic"}, {**config(), "observability":"all"}, {**config(), "unexpected":1}])
def test_invalid_config_refused(bad):
    with pytest.raises(ValueError, match="knowledge_context"):
        validate_config(bad)


def test_native_config_loader_only_enables_explicit_new_node():
    from test_native_episode_config import config as native_config, program
    from speech_aware_evidence_acquisition.rails.native_episode_config import load_native_episode_config
    document = native_config()
    assert load_native_episode_config(document, program()).knowledge_context_json is None
    document["native_episode"]["knowledge_context"] = config()
    assert load_native_episode_config(document, program()).knowledge_context_json == validate_config(config())
    document["native_episode"]["knowledge_context"] = config(mode="legacy")
    assert load_native_episode_config(document, program()).knowledge_context_json is None


def test_real_native_runner_emits_context_node_and_preserves_contact_budget(tmp_path):
    from speech_aware_evidence_acquisition.reproduction.ojw import openjiuwen_available
    if not openjiuwen_available():
        pytest.skip("fixed-image OpenJiuwen required")
    from test_native_episode_runner import _ActualEpisodeHarness
    harness = _ActualEpisodeHarness(tmp_path, sample_count=1, max_revisions=0,
                                    max_contacts=4, planner_actions=("ACCEPT",))
    harness.config = replace(harness.config, knowledge_context_json=validate_config(config()))
    try:
        summary = harness.run()
    finally:
        closed = harness.close()
    receipt = summary.results[0].native_episode_receipt
    records = receipt["knowledge_context_compilations"]
    assert len(records) == 4 and closed["request_count"] == 4
    assert all(row["compilation"]["model_calls"] == 0 for row in records)
    assert all(row["request_sha256"] and row["response_sha256"] for row in records)
    org = [payload for channel, _, payload in harness.trace.records if "org_knowledge_context" in payload]
    assert len(org) == 4


def test_equal_text_keys_with_distinct_original_label_scope_never_merge():
    first = KeyFacet("store::intent:a", "label", "text", "same", ("train:a",))
    second = replace(first, facet_id="store::intent:b")
    knowledge = replace(snapshot(), facets=(first, second))
    parts = [{"type":"text", "text": item.representation,
              "evidence_id":"key:" + item.facet_id, "knowledge_role":"key",
              "source_uri":f"knowledge-snapshot://{'2' * 64}/{item.facet_id}"}
             for item in knowledge.facets]
    output, audit = prepare_content(parts, snapshot=knowledge, snapshot_sha256="2" * 64,
                                    config_json=validate_config(config()))
    assert output == parts and audit["removed_count"] == 0


def test_no_source_mutation_and_incomplete_nonknowledge_parts_retained():
    item = value()
    knowledge = snapshot(item)
    state_before = dict(vars(knowledge))
    text = {"type": "text", "text": "application note"}
    parts = [text, part(item)]
    original = json.dumps(parts, sort_keys=True)
    output, audit = prepare_content(parts, snapshot=knowledge, snapshot_sha256="2" * 64,
                                    config_json=validate_config(config()))
    assert output == parts and output[0] is text
    assert json.dumps(parts, sort_keys=True) == original
    assert vars(knowledge) == state_before
    assert audit["unresolved_count"] == 1


def test_alternate_release_never_uses_base_snapshot_metadata():
    items = [value(), value("store::b")]
    parts = [part(item, source_uri=f"knowledge-snapshot://{'3' * 64}/{item.value_id}") for item in items]
    output, audit = run(items, parts)
    assert output == parts and audit["removed_count"] == 0
    assert audit["unresolved_count"] == 2
    assert all(row["category"]["metadata_status"] == "unresolved" for row in audit["mapping"])
