"""Opt-in, model-free knowledge classification and exact deduplication.

Metadata comes only from the pinned snapshot. Unknown metadata is retained.
No inferred intent, string normalization, near-duplicate or contradiction filter.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from typing import Mapping

SCHEMA = "saea-knowledge-context-v1"


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def validate_config(value):
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"schema", "mode", "observability"}:
        raise ValueError("knowledge_context requires schema, mode, observability")
    if value["schema"] != SCHEMA or value["mode"] not in {"legacy", "strict-v1"}:
        raise ValueError("knowledge_context schema/mode is invalid")
    if value["observability"] not in {"normal", "debug"}:
        raise ValueError("knowledge_context observability must be normal or debug")
    return None if value["mode"] == "legacy" else _json(dict(value))


def _measure(parts):
    texts = [part.get("text", "") for part in parts if part.get("type") == "text" and isinstance(part.get("text", ""), str)]
    rendered = "\n".join(
        f'[evidence {part.get("evidence_id", "unresolved")}] {part.get("text", "")}'
        for part in parts if part.get("type") == "text"
    )
    return {
        "body_characters": sum(map(len, texts)),
        "body_utf8_bytes": sum(len(text.encode("utf-8")) for text in texts),
        "rendered_characters": len(rendered),
        "token_basis": "unmeasured; use same pinned tokenizer in paired CPU replay; bytes are not tokens",
        "exact_text_tokens": None,
    }


def _metadata(part, *, snapshot, values, facets):
    evidence_id = part.get("evidence_id")
    role = part.get("knowledge_role")
    item = values.get(evidence_id) if role == "value" else None
    if role == "key" and isinstance(evidence_id, str) and evidence_id.startswith("key:"):
        item = facets.get(evidence_id[4:])
    if item is None:
        return None
    document = item.as_json()
    document.pop("value_id" if role == "value" else "facet_id")
    source_id = evidence_id[4:] if role == "key" else evidence_id
    store = source_id.split("::", 1)[0] if "::" in source_id else None
    return {
        "knowledge_role": role, "source_store": store,
        "source_partition_role": snapshot.source_partition_role,
        "snapshot_id": snapshot.snapshot_id, "snapshot_version": snapshot.snapshot_version,
        "builder_policy_id": snapshot.builder_policy_id,
        "source_split_identity_hash": snapshot.source_split_identity_hash,
        "semantic_payload": document,
        "scope": ({"task_target": item.task_target,
                   "associated_facet_ids": list(item.associated_facet_ids)}
                  if role == "value" else {"facet_id": item.facet_id, "dimension": item.dimension,
                                           "parent_facet_ids": list(item.parent_facet_ids)}),
    }


def build_lookup(snapshot):
    """Caller-owned immutable-source lookup; no mutation of the snapshot."""
    return ({item.value_id: item for item in snapshot.values},
            {item.facet_id: item for item in snapshot.facets})


def prepare_content(content, *, snapshot, snapshot_sha256, config_json, lookup=None):
    """Return stable-order content and provenance; OFF is identity.

    Canonicalize JSON object-key order only. Preserve string bytes, array order,
    priors, sources, scopes, versions. Unknown/index-derived parts remain.
    Structural conflicts are reported, not interpreted as semantic contradictions.
    """
    if config_json is None:
        return content, None
    started = time.perf_counter()
    config = json.loads(config_json)
    if validate_config(config) != config_json:
        raise ValueError("knowledge_context config is not canonical")
    values, facets = lookup if lookup is not None else build_lookup(snapshot)
    kept, records, groups, seen = [], [], {}, {}
    before_categories, after_categories = Counter(), Counter()
    same_body, same_id = {}, {}
    for index, part in enumerate(content):
        if not isinstance(part, Mapping) or part.get("type") != "text":
            kept.append(part)
            continue
        if not all(isinstance(part.get(key), str) and part[key] for key in ("text", "evidence_id", "source_uri")):
            # Nonknowledge or incomplete metadata is retained verbatim.
            kept.append(part)
            records.append({"input_index": index, "output_index": len(kept) - 1,
                "evidence_id": part.get("evidence_id"), "source_uri": part.get("source_uri"),
                "representative_id": part.get("evidence_id"), "semantic_sha256": _hash(part),
                "category": {"metadata_status": "unresolved"}, "dedup_eligible": False})
            category_key = _json({"metadata_status": "unresolved"})
            before_categories[category_key] += 1
            after_categories[category_key] += 1
            continue
        source_id = part["evidence_id"][4:] if part.get("knowledge_role") == "key" else part["evidence_id"]
        bound = part["source_uri"] == f"knowledge-snapshot://{snapshot_sha256}/{source_id}"
        metadata = _metadata(part, snapshot=snapshot, values=values, facets=facets) if bound else None
        category = {"source_store": None, "knowledge_role": part.get("knowledge_role"),
                    "scope": None, "metadata_status": "unresolved"}
        if metadata is not None:
            category.update(source_store=metadata["source_store"], scope=metadata["scope"],
                            metadata_status="snapshot-bound")
        category_key = _json(category)
        before_categories[category_key] += 1
        signature = _hash({"part": {key: value for key, value in part.items()
                                    if key not in {"evidence_id", "source_uri"}},
                           "metadata": metadata, "snapshot_sha256": snapshot_sha256,
                           "unbound_source": None if bound else part["source_uri"]})
        eligible = metadata is not None and bound
        representative = seen.get(signature) if eligible else None
        if representative is None:
            representative = len(kept)
            kept.append(part)
            after_categories[category_key] += 1
            if eligible:
                seen[signature] = representative
        record = {"input_index": index, "output_index": representative,
                  "evidence_id": part["evidence_id"], "source_uri": part["source_uri"],
                  "representative_id": kept[representative]["evidence_id"],
                  "semantic_sha256": signature, "category": category,
                  "dedup_eligible": eligible}
        records.append(record)
        groups.setdefault(representative, []).append(record)
        same_body.setdefault(_hash(part["text"]), set()).add(signature)
        same_id.setdefault(part["evidence_id"], set()).add(signature)
    duplicates = [rows for rows in groups.values() if len(rows) > 1]
    audit = {
        "schema": "saea-knowledge-context-receipt-v1",
        "node": "runtime-knowledge-classification-dedup",
        "config": config, "config_sha256": _hash(config),
        "snapshot_sha256": snapshot_sha256,
        "input_sha256": _hash(content), "output_sha256": _hash(kept),
        "mapping_sha256": _hash(records), "input_count": len(records),
        "output_count": sum(part.get("type") == "text" for part in kept),
        "removed_count": sum(len(rows) - 1 for rows in duplicates),
        "duplicate_group_count": len(duplicates),
        "duplicate_groups": [[{key: row[key] for key in (
            "input_index", "output_index", "evidence_id", "source_uri", "representative_id", "semantic_sha256"
        )} for row in rows] for rows in duplicates],
        "category_count": len(before_categories),
        "category_counts": [{"category": json.loads(key), "input": count,
                             "output": after_categories[key]}
                            for key, count in before_categories.items()],
        "unresolved_count": sum(not row["dedup_eligible"] for row in records),
        "conflicts": {
            "same_id_different_payload_groups": sum(len(items) > 1 for items in same_id.values()),
            "same_text_distinct_semantics_groups": sum(len(items) > 1 for items in same_body.values()),
            "semantic_contradictions": None,
            "policy": "retain all; no semantic inference or conflict resolution",
        },
        "before": _measure(content), "after": _measure(kept),
        "elapsed_seconds": time.perf_counter() - started, "model_calls": 0,
    }
    if config["observability"] == "debug":
        audit["mapping"] = records
    return kept, audit


def prepare_messages(messages, *, snapshot, snapshot_sha256, config_json, lookup=None):
    if config_json is None:
        return messages, None
    if len(messages) != 2 or messages[1].get("role") != "user" or not isinstance(messages[1].get("content"), (list, tuple)):
        raise ValueError("knowledge_context requires native base messages")
    content, audit = prepare_content(messages[1]["content"], snapshot=snapshot,
                                    snapshot_sha256=snapshot_sha256, config_json=config_json, lookup=lookup)
    return (messages[0], {**messages[1], "content": content}), audit
