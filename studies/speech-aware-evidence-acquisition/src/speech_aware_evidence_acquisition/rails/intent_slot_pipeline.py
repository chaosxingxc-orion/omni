"""Opt-in candidate-conditioned pipeline contracts, independent of the carrier.

No model client lives here. The native runtime owns audio, evidence, contacts,
request byte identity, FIT, response custody and replay.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

SCHEMA = "saea-intent-slot-pipeline-v1"
SCHEMA_V2 = "saea-intent-slot-pipeline-v2"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def parse_config(raw):
    fields = {"schema", "slot_mode", "max_candidates", "candidate_outside_recovery"}
    if not isinstance(raw, Mapping) or not fields <= set(raw) <= fields | {"evidence_policy", "duplicate_candidate_policy", "node_rails", "answer_only_fallback", "coverage_recovery"}:
        raise ValueError("intent_slot_pipeline requires schema, slot_mode, max_candidates and candidate_outside_recovery")
    if raw["schema"] not in {SCHEMA, SCHEMA_V2} or raw["slot_mode"] not in {"batched", "per_candidate"}:
        raise ValueError("unsupported intent-slot graph/schema version or slot mode")
    if type(raw["max_candidates"]) is not int or not 1 <= raw["max_candidates"] <= 3:
        raise ValueError("intent-slot max_candidates must be 1..3")
    if raw["candidate_outside_recovery"] is not False:
        raise ValueError("candidate-outside recovery is not implemented in this graph")
    if raw.get("evidence_policy", "strict") not in {"strict", "unique_source_alias_v1"}:
        raise ValueError("unsupported evidence policy")
    if raw.get("duplicate_candidate_policy", "strict") not in {"strict", "identical_v1", "group_v2"}:
        raise ValueError("unsupported duplicate candidate policy")
    if (raw.get("duplicate_candidate_policy") == "group_v2") != (raw["schema"] == SCHEMA_V2):
        raise ValueError("v2 requires group_v2 and v1 cannot use it")
    if "node_rails" in raw and (raw["schema"] != SCHEMA_V2 or raw["node_rails"] != "sdk-handoff-v1"):
        raise ValueError("unsupported candidate node rails")
    if raw.get("node_rails") == "sdk-handoff-v1":
        if raw.get("answer_only_fallback") not in {"budgeted-v1", "off"}:
            raise ValueError("node rails require explicit answer_only_fallback: budgeted-v1 or off")
    elif "answer_only_fallback" in raw:
        raise ValueError("answer_only_fallback requires node rails")
    if "coverage_recovery" in raw:
        recovery = raw["coverage_recovery"]
        if (raw.get("node_rails") != "sdk-handoff-v1" or raw["slot_mode"] != "batched" or raw["max_candidates"] != 3
                or raw.get("answer_only_fallback") not in {"off", "budgeted-v1"}
                or not isinstance(recovery, Mapping)
                or set(recovery) != {"strategy", "max_extra_contacts"}
                or recovery["strategy"] != "complete-original-candidate-set-v1"
                or type(recovery["max_extra_contacts"]) is not int
                or not 1 <= recovery["max_extra_contacts"] <= raw["max_candidates"]):
            raise ValueError("invalid bounded Slot coverage recovery policy")
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


def candidates(text, maximum, legal_ids=None, *, duplicate_policy="strict"):
    value = document(text)
    rows = value.get("candidates")
    if not isinstance(rows, list) or not 1 <= len(rows) <= maximum:
        raise ValueError("candidate count is outside the configured bound")
    result, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("candidate must be an object")
        cid = row.get("intent_id")
        if not isinstance(cid, str) or not cid.strip():
            raise ValueError("candidate ids must be nonempty")
        if legal_ids is not None and cid not in legal_ids:
            raise ValueError("candidate intent is not in the carrier inventory")
        if cid in seen:
            if duplicate_policy == "group_v2":
                previous = next(item for item in result if item["candidate_id"] == cid)
                previous["sources"].append(dict(row))
                continue
            if duplicate_policy != "identical_v1":
                raise ValueError("candidate ids must be nonempty and unique")
            previous = next(item for item in result if item["candidate_id"] == cid)
            if any(previous.get(k) != row.get(k) for k in ("distinction", "evidence_refs", "unresolved_fields")):
                raise ValueError("duplicate candidate has conflicting content")
            continue
        seen.add(cid)
        projected = dict(row, candidate_id=cid)
        if duplicate_policy == "group_v2":
            projected["sources"] = [dict(row)]
        result.append(projected)
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
    payload = {"schema": config["schema"], "config_sha256": digest(config),
               "candidate_outside_recovery": False}
    if config.get("node_rails") == "sdk-handoff-v1" and role_kind != "perception":
        context = state.get("intent_slot_context", {})
        payload["node_rails"] = "sdk-handoff-v1"
        payload["candidates"] = [{"candidate_id": r["candidate_id"], "kind": "sample_hypothesis",
            "unresolved_fields": list(dict.fromkeys(field for source in r.get("sources", [r]) for field in source.get("unresolved_fields", [])))} for r in context.get("candidates", [])]
        if role_kind == "solver":
            payload["slot_results"] = {cid: {"candidate_id": row["candidate_id"], "entities": row["entities"],
                "status": "unresolved" if row["unresolved_fields"] else "observed", "unresolved_fields": row["unresolved_fields"]}
                for cid, row in context.get("slot_results", {}).items()}
        if role_kind == "slot":
            from .candidate_handoff import normalized_response
            # Transport projection reads the normalized state artifact, not raw model prose.
            raw_grouped = {"candidates": [source for row in context["candidates"] for source in row.get("sources", [row])]}
            normalized = normalized_response("perception", canonical(raw_grouped), {}, config)
            # In batched mode this is the full P view; per-candidate branches use
            # the full source hash recorded separately in the stage context.
            payload["perception_source"] = {"response_text_sha256": context.get("normalized_perception_sha256", hashlib.sha256(normalized.encode()).hexdigest())}
        payload["evidence_inventory"] = context["evidence_inventory"]
        if role_kind == "slot" and context.get("coverage_repair"):
            payload["coverage_repair"] = context["coverage_repair"]
        return "\n\nCandidate-conditioned pipeline context: " + canonical(payload)
    if role_kind == "perception":
        payload["max_candidates"] = config["max_candidates"]
        payload["evidence_inventory"] = state["intent_slot_context"]["evidence_inventory"]
    else:
        payload.update(state.get("intent_slot_context", {}))
    return "\n\nCandidate-conditioned pipeline context: " + canonical(payload)


def validate_receipt(receipt, *, trace_records=None, snapshot_cache=None):
    """Bind candidate groups to the actual native visits and archived responses."""
    block = receipt.get("intent_slot_pipeline")
    if block is None:
        return
    if block.get("schema") not in {SCHEMA, SCHEMA_V2} or block.get("slot_mode") not in {"batched", "per_candidate"}:
        raise ValueError("invalid intent-slot receipt identity")
    ids = block.get("candidate_ids")
    if not isinstance(ids, list) or not 1 <= len(ids) <= 3 or len(set(ids)) != len(ids):
        raise ValueError("invalid intent-slot candidate identity")
    visits = receipt["role_visits"]
    slots = block["slot_sources"]
    recovery = block.get("coverage_recovery")
    if recovery:
        config = parse_config(block.get("recovery_config"))
        if config.get("coverage_recovery") != recovery or digest(config) != block.get("config_sha256"):
            raise ValueError("Slot recovery config identity differs")
    expected = ([ids] + [s["candidate_ids"] for s in slots[1:]] if recovery else
                [ids] if block["slot_mode"] == "batched" else [[cid] for cid in ids])
    if recovery and (not slots or len(slots) - 1 > recovery["max_extra_contacts"]
                     or any(len(s["candidate_ids"]) != 1 for s in slots[1:])):
        raise ValueError("invalid Slot recovery contact groups")
    if recovery:
        extras = len(slots) - 1
        for visit in visits:
            execution = visit.get("execution_receipt", {})
            contacts = execution.get("contacts", [])
            if not isinstance(contacts, list) or len(contacts) not in (1, 2):
                raise ValueError("invalid coverage recovery contact chain")
            if len(contacts) == 2:
                extras += 1
                parent, child = contacts
                if (config.get("answer_only_fallback") != "budgeted-v1"
                        or child.get("agent_contact_id") != parent.get("agent_contact_id", "") + ":answer-only"
                        or execution.get("final_contact") != child):
                    raise ValueError("invalid answer-only parent-child identity")
                if trace_records is not None:
                    for contact in contacts:
                        rid = contact["request_id"]
                        request = trace_records.get(rid + "/request", {})
                        response = trace_records.get(rid + "/response", {})
                        payload = request.get("payload", {})
                        history = payload.get("history", [])
                        if (request.get("payload_sha256") != contact.get("request_sha256")
                                or not history or history[-1].get("contact_id") != contact.get("agent_contact_id")
                                or response.get("response_sha256") != contact.get("response_sha256")
                                or response.get("response_text_sha256") != contact.get("response_text_sha256")
                                or not isinstance(response.get("text"), str)
                                or hashlib.sha256(response["text"].encode()).hexdigest() != contact.get("response_text_sha256")):
                            raise ValueError("answer-only contact differs from archived request or response")
                    from ..core.reasoning_knobs import answer_only_decoding, fallback_reserve
                    parent_payload = trace_records[parent["request_id"] + "/request"]["payload"]
                    child_payload = trace_records[child["request_id"] + "/request"]["payload"]
                    parent_decoding = parent_payload.get("decoding_params", {})
                    child_decoding = child_payload.get("decoding_params", {})
                    if fallback_reserve(parent_decoding) is None or child_decoding != answer_only_decoding(parent_decoding):
                        raise ValueError("answer-only decoding differs from parent")
                    parent_response = trace_records[parent["request_id"] + "/response"]
                    relation = trace_records.get(child["agent_contact_id"] + "/recovery-parent", {})
                    if (parent_response.get("text") != "" or relation != {
                            "org_recovery_parent_contact_id": parent["agent_contact_id"],
                            "org_recovery_parent_request_id": parent["request_id"],
                            "org_recovery_parent_response_sha256": parent["response_sha256"],
                            "org_recovery_child_contact_id": child["agent_contact_id"]}):
                        raise ValueError("answer-only parent relationship differs from trace")
        if extras > recovery["max_extra_contacts"]:
            raise ValueError("coverage recovery shared budget exceeded")
        retained = slots[0].get("valid_ids")
        if (not isinstance(retained, list) or len(set(retained)) != len(retained)
                or any(cid not in ids for cid in retained)):
            raise ValueError("invalid initial Slot recovery sources")
        missing = [cid for cid in ids if cid not in retained]
        outstanding = list(missing)
        successor = block.get('slot_successor')
        invalid_sources = [s for s in slots[1:] if not s.get('valid_ids')]
        if invalid_sources:
            if (not isinstance(successor, Mapping) or len(invalid_sources) != 1
                    or successor.get('rejected_contact') != invalid_sources[0]['source']
                    or successor.get('candidate_ids') != ids
                    or successor.get('remaining_candidate_ids') != missing
                    or successor.get('max_extra_contacts') != recovery['max_extra_contacts']):
                raise ValueError('Slot successor source or budget differs')
            prior_fallbacks = sum(len(v['execution_receipt']['contacts']) - 1
                                  for v in visits[:slots.index(invalid_sources[0]) + 2])
            if successor.get('extra_contacts') != slots.index(invalid_sources[0]) + prior_fallbacks:
                raise ValueError('Slot successor shared extra budget differs')
            following = slots[slots.index(invalid_sources[0]) + 1:]
            if (not following or ':successor:' + successor['source_attempt_sha256'][:16]
                    not in following[0]['source']['agent_contact_id']):
                raise ValueError('Slot successor contact generation differs')
        for source in slots[1:]:
            if not outstanding or source['candidate_ids'] != [outstanding[0]] or source.get('valid_ids') not in ([], source['candidate_ids']):
                raise ValueError("Slot recovery source is not the next missing branch")
            if source['valid_ids']:
                outstanding.pop(0)
        if outstanding:
            raise ValueError("Slot recovery sources do not fill exactly the missing branches")
    if (not recovery and [s["candidate_ids"] for s in slots] != expected) or (recovery and slots[0]["candidate_ids"] != ids):
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
        trace_receipts = [value.get("use_native_episode") for value in trace_records.values()
                          if isinstance(value, Mapping) and isinstance(value.get("use_native_episode"), Mapping)
                          and value["use_native_episode"].get("accepted_solver_receipt") == receipt.get("accepted_solver_receipt")
                          and receipt.get("accepted_solver_receipt") is not None]
        if block.get("node_rails") == "sdk-handoff-v1" and (len(trace_receipts) != 1 or trace_receipts[0] != receipt):
            raise ValueError("native episode receipt differs from raw trace")
        if receipt.get("accepted_solver_receipt") is not None and "selected_candidate_id" not in block:
            raise ValueError("accepted pipeline receipt omits its selected candidate")
        for source in sources:
            rid = source["request_id"]
            request = trace_records.get(rid + "/request", {})
            response = trace_records.get(rid + "/response", {})
            if request.get("payload_sha256") != source["request_sha256"]:
                raise ValueError("candidate source request is absent or mismatched")
            if (response.get("response_sha256") != source["response_sha256"] or response.get("response_text_sha256") != source["response_text_sha256"]
                or (block.get("schema") == SCHEMA_V2 and (not isinstance(response.get("text"), str) or hashlib.sha256(response["text"].encode()).hexdigest() != source["response_text_sha256"]))):
                raise ValueError("candidate source response is absent or mismatched")
        perception_text = trace_records[block["perception_source"]["request_id"] + "/response"].get("text")
        actual_candidates = candidates(perception_text, 3, duplicate_policy=block.get("duplicate_candidate_policy", "strict"))
        if [r["candidate_id"] for r in actual_candidates] != ids:
            raise ValueError("candidate ids differ from archived Perception text")
        if block.get("schema") == SCHEMA_V2:
            grouping = {"policy": "group_v2", "raw_count": len(document(perception_text)["candidates"]),
                        "groups": [{"candidate_id": r["candidate_id"], "sources": r["sources"]} for r in actual_candidates]}
            if block.get("grouping") != grouping:
                raise ValueError("candidate grouping differs from archived Perception")
        if block.get("evidence_policy") == "unique_source_alias_v1" or block.get("schema") == SCHEMA_V2:
            bindings = block.get("evidence_bindings")
            if not isinstance(bindings, list) or [b.get("stage") for b in bindings] != ["perception"] + ["slot"] * len(slots):
                raise ValueError("missing source-bound evidence mapping")
            for binding, source, expected in zip(bindings, sources, [ids] + [s["candidate_ids"] for s in slots]):
                response = trace_records[source["request_id"] + "/response"]
                if block.get("node_rails") == "sdk-handoff-v1":
                    from .candidate_handoff import validate_inventory_against_request
                    validate_inventory_against_request(binding["inventory"], trace_records[source["request_id"] + "/request"], expected)
                if binding.get("response_text_sha256") != response.get("response_text_sha256"):
                    raise ValueError("evidence mapping response identity differs")
                rows = ((document(response.get("text"))["candidates"] if block.get("schema") == SCHEMA_V2
                         else candidates(response.get("text"), 3, duplicate_policy=block.get("duplicate_candidate_policy", "strict")))
                        if binding["stage"] == "perception" else
                        ([r for r in document(response.get("text")).get("candidates", []) if r.get("candidate_id") in slots[sources.index(source)-1]["valid_ids"]]
                         if recovery else slot_results(response.get("text"), expected).values()))
                if validate_evidence_refs(rows, binding["inventory"], expected, policy=block.get("evidence_policy", "strict")) != binding.get("mappings"):
                    raise ValueError("evidence mapping differs from raw response")
        for slot in slots:
            text = trace_records[slot["source"]["request_id"] + "/response"].get("text")
            if recovery:
                from .adapters.slurp_intent_slot import classify_slot_coverage
                request = trace_records[slot["source"]["request_id"] + "/request"]
                response_format = request["payload"]["decoding_params"]["response_format"]
                binding = block["evidence_bindings"][slots.index(slot) + 1]
                try:
                    report = classify_slot_coverage(text=text, response_format=response_format,
                        context={"candidates": [{"candidate_id": cid} for cid in slot["candidate_ids"]],
                                 "evidence_inventory": binding["inventory"]}, config=block)
                except (ValueError, KeyError, TypeError):
                    if slot['valid_ids'] or slots.index(slot) == 0:
                        raise
                    report = {'valid': {}}
                if list(report["valid"]) != slot["valid_ids"]:
                    raise ValueError("Slot recovery valid branches differ from archived response")
                actual = report["valid"]
                if digest(actual) != slot["result_sha256"]:
                    raise ValueError("Slot recovery result differs from archived response")
                continue
            if block.get("node_rails") == "sdk-handoff-v1":
                from .candidate_handoff import normalized_response
                text = normalized_response("slot", text, {"candidates": [{"candidate_id": cid} for cid in slot["candidate_ids"]]}, {"max_candidates": 3})
            actual = slot_results(text, slot["candidate_ids"])
            if digest(actual) != slot["result_sha256"]:
                raise ValueError("Slot parsed result differs from archived response")
        if block.get("node_rails") == "sdk-handoff-v1":
            from .candidate_handoff import validate_trace_handoffs
            validate_trace_handoffs(receipt, trace_records, snapshot_cache=snapshot_cache)
        if block.get("schema") == SCHEMA_V2 and "selected_candidate_id" in block:
            source = block["decision_source"]
            request = trace_records[source["request_id"] + "/request"]
            response = trace_records[source["request_id"] + "/response"]
            text = response.get("text")
            if (request.get("payload_sha256") != source["request_sha256"]
                or response.get("response_sha256") != source["response_sha256"]
                or response.get("response_text_sha256") != source["response_text_sha256"]
                or hashlib.sha256(text.encode()).hexdigest() != source["response_text_sha256"]
                or document(text) != {"candidate_id": block["selected_candidate_id"]}):
                raise ValueError("Decision response/source differs from trace")
            combined = {}
            for slot in slots:
                raw = trace_records[slot["source"]["request_id"] + "/response"]["text"]
                if recovery:
                    rows = [row for row in document(raw).get("candidates", []) if row.get("candidate_id") in slot["valid_ids"]]
                    parsed = slot_results(canonical({"candidates": rows}), slot["valid_ids"])
                    if set(combined) & set(parsed):
                        raise ValueError("Slot repair overwrote an existing result")
                    combined.update(parsed)
                    continue
                if block.get("node_rails") == "sdk-handoff-v1":
                    from .candidate_handoff import normalized_response
                    raw = normalized_response("slot", raw, {"candidates": [{"candidate_id": cid} for cid in slot["candidate_ids"]]}, {"max_candidates": 3})
                combined.update(slot_results(raw, slot["candidate_ids"]))
            if set(combined) != set(ids):
                raise ValueError("Slot recovery does not cover original candidates")
            answer = derive_answer(block["selected_candidate_id"], combined)
            if hashlib.sha256(answer.encode()).hexdigest() != block.get("derived_answer_sha256"):
                raise ValueError("derived answer differs from Slot source")
            return answer



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


def validate_evidence_refs(rows, inventory, expected_ids, *, policy="strict"):
    if not isinstance(inventory, Mapping) or not inventory.get("sample_id"):
        raise ValueError("missing current-sample evidence inventory")
    if inventory["candidate_ids"] and inventory["candidate_ids"] != list(expected_ids):
        raise ValueError("evidence inventory belongs to another candidate branch")
    available = {source["source_id"] for source in inventory["sources"]}
    aliases = {}
    for source in inventory["sources"]:
        source_id = source["source_id"]
        if source.get("kind") == "knowledge" and "::" in source_id:
            aliases.setdefault(source_id.split("::", 1)[1], set()).add(source_id)
    mappings = []
    for row in rows:
        refs = row.get("evidence_refs", [])
        if not isinstance(refs, list):
            raise ValueError("evidence references must be a list")
        for ref in refs:
            if not isinstance(ref, str):
                raise ValueError("evidence reference must be text")
            if ref in available:
                canonical_id = ref
            elif policy == "unique_source_alias_v1" and len(aliases.get(ref, ())) == 1:
                canonical_id = next(iter(aliases[ref]))
            else:
                raise ValueError("evidence reference was not supplied uniquely to the current sample and candidate")
            mappings.append({"raw": ref, "source_id": canonical_id})
    return mappings


def dynamic_slot_format(base, ids):
    import copy
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("dynamic Slot requires unique candidate IDs")
    result = copy.deepcopy(base)
    array = result["json_schema"]["schema"]["properties"]["candidates"]
    array["minItems"] = array["maxItems"] = len(ids)
    array["items"]["properties"]["candidate_id"] = {"type": "string", "enum": list(ids)}
    # JSON Schema uniqueItems alone does not enforce uniqueness of a field.
    # Exact coverage is checked at the seal; no missing row is synthesized.
    return result


def dynamic_decision_format(ids):
    return {"type": "json_schema", "json_schema": {"name": "slurp_candidate_decision_v2",
        "strict": True, "schema": {"type": "object", "additionalProperties": False,
        "required": ["candidate_id"], "properties": {"candidate_id": {"type": "string", "enum": list(ids)}}}}}


def derive_answer(cid, slots):
    if cid not in slots or "_" not in cid:
        raise ValueError("selected candidate has no Slot source or SLURP identity")
    scenario, action = cid.split("_", 1)
    if not scenario or not action:
        raise ValueError("invalid SLURP candidate identity")
    return canonical({"scenario": scenario, "action": action, "entities": slots[cid]["entities"]})
