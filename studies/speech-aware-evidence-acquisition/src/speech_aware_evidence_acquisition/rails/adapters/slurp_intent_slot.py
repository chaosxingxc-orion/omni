"""SLURP-specific ontology and final task projection for the generic graph."""
import json

from ..intent_slot_pipeline import candidates, decision, document, slot_results, validate_evidence_refs


def classify_slot_coverage(*, text, response_format, context, config):
    """Classify complete Slot rows without inventing values for missing IDs.

    A duplicate or foreign ID makes the response ambiguous and is refused as a
    whole. Individual invalid rows are never returned as usable results. The
    inventory remains bound to the *original* candidate set, even though each
    row is checked separately.
    """
    import jsonschema

    value = document(text)
    if set(value) != {"candidates"} or not isinstance(value["candidates"], list):
        raise ValueError("Slot requires candidate-keyed results")
    ids = [row["candidate_id"] for row in context["candidates"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("invalid original candidate set")
    schema = response_format["json_schema"]["schema"]
    row_schema = schema["properties"]["candidates"]["items"]
    legal = row_schema["properties"]["entities"]["items"]["properties"]["type"]["enum"]
    valid, invalid, seen = {}, {}, set()
    if not isinstance(context["evidence_inventory"], dict) or context["evidence_inventory"].get("candidate_ids") != ids:
        raise ValueError("evidence inventory belongs to another candidate branch")
    for row in value["candidates"]:
        if not isinstance(row, dict) or not isinstance(row.get("candidate_id"), str):
            raise ValueError("Slot row has no identifiable candidate")
        cid = row["candidate_id"]
        if cid not in ids or cid in seen:
            raise ValueError("Slot candidate is duplicate or belongs to another branch")
        seen.add(cid)
        try:
            jsonschema.validate(row, row_schema)
            parsed = slot_results(json.dumps({"candidates": [row]}), [cid], legal)
            validate_evidence_refs(parsed.values(), context["evidence_inventory"], ids,
                                   policy=config.get("evidence_policy", "strict"))
        except (jsonschema.ValidationError, ValueError, KeyError, TypeError) as error:
            invalid[cid] = str(error)
        else:
            valid[cid] = parsed[cid]
    return {"valid": valid, "missing": [cid for cid in ids if cid not in valid],
            "invalid": invalid}


def registered_slot_format(lock, root, **params):
    from ...data.slurp import load_slurp_ontology
    if params:
        raise ValueError("candidate Slot format accepts no sample-specific parameters")
    ontology = load_slurp_ontology(lock, root)
    entity = {"type": "object", "additionalProperties": False,
              "required": ["type", "filler"], "properties": {
                  "type": {"type": "string", "enum": list(ontology.entity_types)},
                  "filler": {"type": "string"}}}
    row = {"type": "object", "additionalProperties": False,
           "required": ["candidate_id", "entities", "evidence_refs", "unresolved_fields"],
           "properties": {
               "candidate_id": {"type": "string", "enum": list(ontology.intents)},
               "entities": {"type": "array", "items": entity},
               "evidence_refs": {"type": "array", "items": {"type": "string"}},
               "unresolved_fields": {"type": "array", "items": {"type": "string"}}}}
    return {"type": "json_schema", "json_schema": {
        "name": "slurp_candidate_slots", "strict": True,
        "schema": {"type": "object", "additionalProperties": False,
                   "required": ["candidates"], "properties": {
                       "candidates": {"type": "array", "minItems": 1, "maxItems": 3, "items": row}}}}}


def validate_response(*, kind, text, response_format, context, config):
    """Do not infer an intent×slot Cartesian-product legality restriction."""
    import jsonschema
    schema = response_format["json_schema"]["schema"]
    try:
        jsonschema.validate(document(text), schema)
    except jsonschema.ValidationError as error:
        raise ValueError("pipeline response does not satisfy its node schema") from error
    if kind == "perception":
        legal = schema["properties"]["candidates"]["items"]["properties"]["intent_id"]["enum"]
        raw_rows = document(text)["candidates"]
        rows = candidates(text, config["max_candidates"], legal, duplicate_policy=config.get("duplicate_candidate_policy", "strict"))
        validate_evidence_refs(raw_rows if config["schema"].endswith("-v2") else rows, context["evidence_inventory"], [r["candidate_id"] for r in rows], policy=config.get("evidence_policy", "strict"))
        return rows
    if kind == "slot":
        legal = schema["properties"]["candidates"]["items"]["properties"]["entities"]["items"]["properties"]["type"]["enum"]
        ids = [r["candidate_id"] for r in context["candidates"]]
        rows = slot_results(text, ids, legal)
        validate_evidence_refs(rows.values(), context["evidence_inventory"], ids, policy=config.get("evidence_policy", "strict"))
        return rows
    if kind == "solver":
        if config["schema"].endswith("-v2"):
            cid = document(text)["candidate_id"]
            if cid not in context["slot_results"] or cid not in [r["candidate_id"] for r in context["candidates"]]:
                raise ValueError("Decision chose an intent outside the evaluated candidate set")
            return cid
        return decision(text, context["candidates"], context["slot_results"],
                        lambda row: row["scenario"] + "_" + row["action"])
    raise ValueError("unsupported pipeline role")
