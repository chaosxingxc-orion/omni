"""Utilities for a TRAIN-only question route; never consumes evaluation labels.

The caller supplies the pinned snapshot, its vector table, TRAIN metadata and
utterance-level exclusion rosters. Returned scores name facets of the original
intent snapshot, so the existing provider can rank and supply original cards.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping

import numpy as np


class QuestionSourceError(ValueError):
    pass


def utterance_id(sample_id: str) -> str:
    match = re.fullmatch(r"slurp:train:([0-9]+):.+", sample_id or "")
    if match is None:
        raise QuestionSourceError("expected an original TRAIN sample ID")
    return str(int(match.group(1)))


def validate_members(snapshot, metadata, excluded_ids, intent_facets):
    """Validate all question rows before excluding reserved utterances.

    metadata is the original TRAIN JSONL indexed by slurp_id. The snapshot
    itself must be checked against its canonical source pin by the caller.
    """
    excluded = {utterance_id(s) for s in excluded_ids}
    facets = {f["facet_id"]: f for f in snapshot["facets"]}
    if len(facets) != len(snapshot["facets"]):
        raise QuestionSourceError("duplicate question facet")
    members = []
    seen = set()
    for value in snapshot["values"]:
        ids = value.get("associated_facet_ids")
        if not isinstance(ids, list) or len(ids) != 1 or ids[0] not in facets or ids[0] in seen:
            raise QuestionSourceError("question facet/value mapping is not one-to-one")
        fid = ids[0]
        seen.add(fid)
        facet = facets[fid]
        match = re.fullmatch(r"value:([0-9]+)", value.get("value_id", ""))
        if match is None:
            raise QuestionSourceError("untraceable utterance")
        uid = str(int(match.group(1)))
        source_ids = facet.get("source_sample_ids")
        if not source_ids or any(utterance_id(s) != uid for s in source_ids):
            raise QuestionSourceError("question source is not the original TRAIN utterance")
        if value.get("source_sample_ids") != source_ids:
            raise QuestionSourceError("facet and value sources differ")
        pointer = f"governed-jsonl:datasets/slurp/metadata/train.jsonl#slurp_id={uid}"
        if pointer not in value.get("provenance_refs", []):
            raise QuestionSourceError("missing TRAIN metadata provenance")
        row = metadata.get(uid)
        if not isinstance(row, Mapping) or str(row.get("slurp_id")) != uid:
            raise QuestionSourceError("missing original TRAIN metadata")
        sentence = row.get("sentence")
        target = row.get("scenario", "") + "_" + row.get("action", "")
        if (not isinstance(sentence, str) or not sentence
                or facet.get("representation_kind") != "text"
                or facet.get("representation") != f'TRAIN utterance {uid} (annotated-exemplar retrieval key): "{sentence}"'
                or value.get("task_target") != target or target not in intent_facets):
            raise QuestionSourceError("question text or intent differs from TRAIN")
        if uid not in excluded:
            members.append((fid, uid, intent_facets[target]))
    if seen != set(facets):
        raise QuestionSourceError("question snapshot has unmapped facets")
    return sorted(members)


def aggregate_max(queries, vectors, members, *, k, dimension="intent-boundary"):
    """Positive cosine max per intent, deterministic facet-ID tie break.

    queries maps sample ID to a sequence of speech segment vectors. Self
    utterances are excluded for every row even if absent from the roster.
    No fallback fills a short or empty route.
    """
    if type(k) is not int or k < 1 or not members:
        raise QuestionSourceError("positive k and nonempty members required")
    if len({fid for fid, _, _ in members}) != len(members):
        raise QuestionSourceError("duplicate vectors")
    try:
        keys = np.asarray([vectors[fid] for fid, _, _ in members], dtype=np.float64)
    except (KeyError, ValueError, TypeError) as exc:
        raise QuestionSourceError("missing or malformed question vector") from exc
    if keys.ndim != 2 or keys.shape[1] == 0 or not np.isfinite(keys).all():
        raise QuestionSourceError("invalid question vectors")
    norms = np.linalg.norm(keys, axis=1)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise QuestionSourceError("zero or invalid question vector")
    keys = keys / norms[:, None]
    output = {}
    for sid, segments in sorted(queries.items()):
        match = re.fullmatch(r"slurp:train:([0-9]+):.+", sid)
        uid = str(int(match.group(1))) if match else None
        q = np.asarray(segments, dtype=np.float64)
        if q.ndim != 2 or q.shape[0] == 0 or q.shape[1] != keys.shape[1] or not np.isfinite(q).all():
            raise QuestionSourceError("invalid speech query segments")
        qnorm = np.linalg.norm(q, axis=1)
        if np.any(qnorm == 0) or not np.isfinite(qnorm).all():
            raise QuestionSourceError("zero or invalid speech query")
        similarities = (q / qnorm[:, None]) @ keys.T
        scores = {}
        for segment_scores in similarities:
            for (_, source_uid, facet), score in zip(members, segment_scores):
                score = float(score)
                if source_uid != uid and math.isfinite(score) and score > scores.get(facet, 0.0):
                    scores[facet] = score
        chosen = sorted(scores, key=lambda fid: (-scores[fid], fid))[:k]
        output[sid] = {fid: {dimension: scores[fid]} for fid in chosen}
    return output
