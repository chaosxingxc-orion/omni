"""SLURP-specific ontology and final task projection for the generic graph."""
from ..intent_slot_pipeline import candidates, decision, document, slot_results, validate_evidence_refs


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
        rows = candidates(text, config["max_candidates"], legal, duplicate_policy=config.get("duplicate_candidate_policy", "strict"))
        validate_evidence_refs(rows, context["evidence_inventory"], [r["candidate_id"] for r in rows], policy=config.get("evidence_policy", "strict"))
        return rows
    if kind == "slot":
        legal = schema["properties"]["candidates"]["items"]["properties"]["entities"]["items"]["properties"]["type"]["enum"]
        ids = [r["candidate_id"] for r in context["candidates"]]
        rows = slot_results(text, ids, legal)
        validate_evidence_refs(rows.values(), context["evidence_inventory"], ids, policy=config.get("evidence_policy", "strict"))
        return rows
    if kind == "solver":
        return decision(text, context["candidates"], context["slot_results"],
                        lambda row: row["scenario"] + "_" + row["action"])
    raise ValueError("unsupported pipeline role")
