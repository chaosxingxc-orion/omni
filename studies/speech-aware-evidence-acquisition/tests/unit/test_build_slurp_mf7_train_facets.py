"""MF-7 TRAIN key roster rebuild: row schema, snapshot order, verbatim text and the audit reader round trip."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from speech_aware_evidence_acquisition.analysis import slurp_retrieval_key_audit as audit
from speech_aware_evidence_acquisition.rails.knowledge_build import (
    VIEW_SLOT_FILLERS,
    KbRow,
    KnowledgeBuildSpec,
    build_knowledge_snapshot,
)

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("mf7_train_facets", REPO / "scripts/build_slurp_mf7_train_facets.py")
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)

# (utterance id, sentence, scenario, action, ((type, filler), ...)); ids sort as strings, so "10" precedes "2".
_UTTERANCES = (
    ("2", "wake me up at nine am on friday", "alarm", "set", (("time", "nine am"), ("date", "friday"))),
    ("10", "what is the weather", "weather", "query", ()),
    ("7", "play some jazz", "play", "music", (("music_genre", "jazz"),)),
)


def _kb_row(utterance_id, sentence, scenario, action, entities):
    sample = f"slurp:train:{int(utterance_id):06d}:audio-{utterance_id}"
    fillers = {}
    for entity_type, filler in entities:
        fillers.setdefault(entity_type, set()).add(filler)
    return KbRow(
        utterance_id=utterance_id, content=sentence, scenario=scenario, action=action,
        intent=f"{scenario}_{action}", entity_types=tuple(sorted(fillers)),
        entity_fillers={key: tuple(sorted(values)) for key, values in fillers.items()},
        sample_ids=(sample,), speaker_by_sample_id={sample: f"S-{utterance_id}"},
        media_relpath_by_sample_id={sample: f"audio/{utterance_id}.flac"},
    )


def _material(utterance_id, sentence, scenario, action, entities):
    return SimpleNamespace(
        utterance_id=utterance_id, sentence=sentence, scenario=scenario, action=action,
        entities=tuple(SimpleNamespace(type=t, filler=f) for t, f in entities),
    )


def _snapshot(view="exemplar"):
    spec = KnowledgeBuildSpec(carrier_lock_key="slurp", datastore_size=None, seed=builder.SEED, view=view)
    return build_knowledge_snapshot(
        spec.validate(), rows=[_kb_row(*u) for u in _UTTERANCES], forbidden_sample_ids=frozenset())


def test_rows_follow_snapshot_content_order_with_verbatim_text_and_gold():
    snapshot = _snapshot()
    rows = builder.facet_rows(snapshot, [_material(*u) for u in _UTTERANCES])
    content_ids = [f.facet_id for f in snapshot.facets if f.dimension == "content"]
    assert [row["facet_id"] for row in rows] == content_ids == [
        "facet:content:10", "facet:content:2", "facet:content:7"]
    assert rows[1] == {
        "facet_id": "facet:content:2", "text": "wake me up at nine am on friday", "intent": "alarm_set",
        "entities": [{"type": "time", "filler": "nine am"}, {"type": "date", "filler": "friday"}],
    }
    assert rows[0]["entities"] == []


def test_written_roster_reads_back_through_the_audit_reader(tmp_path):
    rows = builder.facet_rows(_snapshot(), [_material(*u) for u in _UTTERANCES])
    path = tmp_path / "train-facets.jsonl"
    path.write_bytes(builder.render_jsonl(rows))
    keys = audit.load_key_rows(path)
    assert [k.facet_id for k in keys] == [row["facet_id"] for row in rows]
    assert keys[1].entities == (("time", "nine am"), ("date", "friday"))
    assert [json.loads(line)["text"] for line in path.read_text(encoding="utf-8").splitlines()] == [
        "what is the weather", "wake me up at nine am on friday", "play some jazz"]


def test_render_is_deterministic_and_keeps_non_ascii():
    rows = [{"facet_id": "facet:content:1", "text": "café", "intent": "a_b", "entities": []}]
    assert builder.render_jsonl(rows) == builder.render_jsonl(rows)
    assert "café".encode("utf-8") in builder.render_jsonl(rows)


def test_a_non_exemplar_view_is_refused():
    with pytest.raises(builder.TrainFacetsError, match="verbatim"):
        builder.facet_rows(_snapshot(VIEW_SLOT_FILLERS), [_material(*u) for u in _UTTERANCES])


def test_a_facet_without_its_utterance_is_refused():
    with pytest.raises(builder.TrainFacetsError, match="no TRAIN utterance"):
        builder.facet_rows(_snapshot(), [_material(*u) for u in _UTTERANCES[:2]])


def test_repeated_material_is_refused():
    materials = [_material(*u) for u in _UTTERANCES]
    with pytest.raises(builder.TrainFacetsError, match="repeats"):
        builder.facet_rows(_snapshot(), materials + materials[:1])
