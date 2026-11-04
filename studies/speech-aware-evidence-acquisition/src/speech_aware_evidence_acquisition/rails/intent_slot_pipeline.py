"""Opt-in candidate-conditioned pipeline contracts, independent of the carrier.

No model client lives here. The native runtime owns audio, evidence, contacts,
request byte identity, FIT, response custody and replay.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

SCHEMA = "saea-intent-slot-pipeline-v1"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def parse_config(raw):
    fields = {"schema", "slot_mode", "max_candidates", "candidate_outside_recovery"}
    if not isinstance(raw, Mapping) or set(raw) != fields:
        raise ValueError("intent_slot_pipeline requires schema, slot_mode, max_candidates and candidate_outside_recovery")
    if raw["schema"] != SCHEMA or raw["slot_mode"] not in {"batched", "per_candidate"}:
        raise ValueError("unsupported intent-slot graph/schema version or slot mode")
    if type(raw["max_candidates"]) is not int or not 1 <= raw["max_candidates"] <= 3:
        raise ValueError("intent-slot max_candidates must be 1..3")
    if raw["candidate_outside_recovery"] is not False:
        raise ValueError("candidate-outside recovery is not implemented in this graph")
    return dict(raw)


def document(text):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate JSON field")
            obj[key] = value
        return obj
    try:
        value = json.loads(text, object_pairs_hook=unique)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("pipeline response must be strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError("pipeline response must be an object")
    return value


def candidates(text, maximum, legal_ids=None):
    value = document(text)
    rows = value.get("candidates")
    if not isinstance(rows, list) or not 1 <= len(rows) <= maximum:
        raise ValueError("candidate count is outside the configured bound")
    result, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("candidate must be an object")
        cid = row.get("intent_id")
        if not isinstance(cid, str) or not cid.strip() or cid in seen:
            raise ValueError("candidate ids must be nonempty and unique")
        if legal_ids is not None and cid not in legal_ids:
            raise ValueError("candidate intent is not in the carrier inventory")
        seen.add(cid)
        result.append(dict(row, candidate_id=cid))
    return result


def slot_results(text, expected_ids, legal_types=None):
    value = document(text)
    if set(value) != {"candidates"} or not isinstance(value["candidates"], list):
        raise ValueError("Slot requires candidate-keyed results")
    result = {}
    for row in value["candidates"]:
        if not isinstance(row, dict) or set(row) != {"candidate_id", "entities", "evidence_refs", "unresolved_fields"}:
            raise ValueError("Slot candidate fields are invalid")
        cid = row["candidate_id"]
        if not isinstance(cid, str) or cid not in expected_ids or cid in result:
            raise ValueError("Slot candidate is duplicate or belongs to another branch")
        entities = row["entities"]
        if not isinstance(entities, list):
            raise ValueError("entities must be an array, including [] for no slots")
        for entity in entities:
            if not isinstance(entity, dict) or set(entity) != {"type", "filler"}:
                raise ValueError("entity requires type and filler")
            if any(not isinstance(entity[k], str) or not entity[k].strip() for k in entity):
                raise ValueError("entity type and filler must be nonempty text")
            if legal_types is not None and entity["type"] not in legal_types:
                raise ValueError("entity type is outside the global carrier inventory")
        for key in ("evidence_refs", "unresolved_fields"):
            if not isinstance(row[key], list) or any(not isinstance(v, str) for v in row[key]):
                raise ValueError("evidence and uncertainty must be text lists")
        result[cid] = row
    if set(result) != set(expected_ids):
        raise ValueError("Slot results do not cover exactly the requested candidates")
    return result


def decision(text, candidate_rows, slots, identity):
    value = document(text)
    cid = identity(value)
    if cid not in {row["candidate_id"] for row in candidate_rows} or cid not in slots:
        raise ValueError("Decision chose an intent outside the evaluated candidate set")
    # Decision selects among already extracted complete hypotheses. Changes to
    # fillers belong to the Slot node, making its optimization attributable.
    if value.get("entities") != slots[cid]["entities"]:
        raise ValueError("Decision entities differ from the selected Slot hypothesis")
    return cid


def prompt_context(config, state, role_kind):
    payload = {"schema": SCHEMA, "config_sha256": digest(config),
               "candidate_outside_recovery": False}
    if role_kind == "perception":
        payload["max_candidates"] = config["max_candidates"]
        payload["evidence_inventory"] = state["intent_slot_context"]["evidence_inventory"]
    else:
        payload.update(state.get("intent_slot_context", {}))
    return "\n\nCandidate-conditioned pipeline context: " + canonical(payload)


def validate_receipt(receipt, *, trace_records=None):
    """Bind candidate groups to the actual native visits and archived responses."""
    block = receipt.get("intent_slot_pipeline")
    if block is None:
        return
    if block.get("schema") != SCHEMA or block.get("slot_mode") not in {"batched", "per_candidate"}:
        raise ValueError("invalid intent-slot receipt identity")
    ids = block.get("candidate_ids")
    if not isinstance(ids, list) or not 1 <= len(ids) <= 3 or len(set(ids)) != len(ids):
        raise ValueError("invalid intent-slot candidate identity")
    visits = receipt["role_visits"]
    slots = block["slot_sources"]
    expected = [ids] if block["slot_mode"] == "batched" else [[cid] for cid in ids]
    if [s["candidate_ids"] for s in slots] != expected:
        raise ValueError("slot receipt groups differ from evaluated candidates")
    if [v["stage"] for v in visits] != ["perception"] + ["slot"] * len(slots) + ["solver"]:
        # Intermediate failures carry no completed pipeline receipt until the
        # Slot stage has finished. A Solver refusal retains that complete prefix.
        if not (receipt.get("terminal_disposition") and [v["stage"] for v in visits] == ["perception"] + ["slot"] * len(slots)):
            raise ValueError("intent-slot receipt differs from actual graph order")
    sources = [block["perception_source"]] + [s["source"] for s in slots]
    if sources != [v["execution_receipt"]["final_contact"] for v in visits[:len(sources)]]:
        raise ValueError("candidate sources differ from native execution receipts")
    if "selected_candidate_id" in block:
        if block["selected_candidate_id"] not in ids or block.get("decision_source") != visits[-1]["execution_receipt"]["final_contact"]:
            raise ValueError("Decision selection/source is not bound to the final native visit")
    if trace_records is not None:
        if receipt.get("accepted_solver_receipt") is not None and "selected_candidate_id" not in block:
            raise ValueError("accepted pipeline receipt omits its selected candidate")
        for source in sources:
            rid = source["request_id"]
            request = trace_records.get(rid + "/request", {})
            response = trace_records.get(rid + "/response", {})
            if request.get("payload_sha256") != source["request_sha256"]:
                raise ValueError("candidate source request is absent or mismatched")
            if response.get("response_sha256") != source["response_sha256"] or response.get("response_text_sha256") != source["response_text_sha256"]:
                raise ValueError("candidate source response is absent or mismatched")
        perception_text = trace_records[block["perception_source"]["request_id"] + "/response"].get("text")
        actual_candidates = candidates(perception_text, 3)
        if [r["candidate_id"] for r in actual_candidates] != ids:
            raise ValueError("candidate ids differ from archived Perception text")
        for slot in slots:
            text = trace_records[slot["source"]["request_id"] + "/response"].get("text")
            actual = slot_results(text, slot["candidate_ids"])
            if digest(actual) != slot["result_sha256"]:
                raise ValueError("Slot parsed result differs from archived response")


def evidence_inventory(*, sample_id, speech_packet_id, speech_sha256, messages, candidate_ids):
    """Names sources actually supplied to this contact, scoped to its row/branch."""
    sources = [{"source_id": speech_packet_id, "kind": "original-audio",
                "source_sha256": speech_sha256}]
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, Mapping) and isinstance(part.get("evidence_id"), str):
                sources.append({"source_id": part["evidence_id"], "kind": "knowledge",
                                "source_uri": part["source_uri"],
                                "source_sha256": hashlib.sha256(part["text"].encode()).hexdigest()})
    return {"sample_id": sample_id, "candidate_ids": list(candidate_ids), "sources": sources}


def validate_evidence_refs(rows, inventory, expected_ids):
    if not isinstance(inventory, Mapping) or not inventory.get("sample_id"):
        raise ValueError("missing current-sample evidence inventory")
    if inventory["candidate_ids"] and inventory["candidate_ids"] != list(expected_ids):
        raise ValueError("evidence inventory belongs to another candidate branch")
    available = {source["source_id"] for source in inventory["sources"]}
    for row in rows:
        refs = row.get("evidence_refs", [])
        if not isinstance(refs, list) or any(not isinstance(ref, str) or ref not in available for ref in refs):
            raise ValueError("evidence reference was not supplied to the current sample and candidate")
