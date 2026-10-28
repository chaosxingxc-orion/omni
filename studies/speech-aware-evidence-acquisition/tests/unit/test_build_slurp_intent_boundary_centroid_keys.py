"""The intent-boundary centroid key builder under a pins document.

Three things are proven here. (a) The pins validator refuses anything that is
not exactly a ``saea-centroid-key-pins-v1`` document -- a missing or extra
key, a digest that is not 64 hex, a placeholder digest -- by name. (b) Without
``--pins`` the builder resolves to its own historical constants, which equal
the diagnostic's committed constants, so the default build is unchanged. (c)
An end-to-end synthetic build under ``--pins`` on a throw-away data root:
sixty synthetic intents, a 1:1 boundary store, an encoding index whose facet
text digests are the facet representations, a pool store whose values carry
``task_target``, and two sealed-vector files -- then the receipt names the
pins, the centroid is the unit mean of the unit member rows, and an intent
with no pool member keeps its written key byte for byte.

No real store, vector file or encoder is touched; the diagnostic module is
imported unchanged from its committed path.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from speech_aware_evidence_acquisition.rails.knowledge import (
    KeyFacet,
    KnowledgeSnapshot,
    ValueBundle,
)
from speech_aware_evidence_acquisition.rails.phase_supply import load_sealed_vectors

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDER = REPO_ROOT / "scripts" / "build_slurp_intent_boundary_centroid_keys.py"
DIAGNOSTIC = REPO_ROOT / "scripts" / "analysis" / "slurp_boundary_key_construction_diagnostic.py"
PORT_PINS = (
    "docs/readiness/2026-10-28-slurp-intent-boundary-centroid-keys-npu-port-v1/pins-v1.json"
)

WIDTH = 5
INTENTS = tuple(f"scenario{i // 6}_action{i % 6}" for i in range(60))
#: The one intent that has no construction utterance in the synthetic pool.
ORPHAN = INTENTS[17]
MEMBERS_PER_INTENT = 3
SPLIT_HASH = "a" * 64


def _builder():
    spec = importlib.util.spec_from_file_location("centroid_keys_builder_under_test", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _builder()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _good_pins(builder) -> dict:
    return copy.deepcopy(builder.DEFAULT_PINS)


# ---------------------------------------------------------------------------
# (a) validation refusals


def test_the_default_pins_validate_to_themselves(builder) -> None:
    assert builder.validate_pins(_good_pins(builder)) == builder.DEFAULT_PINS


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda d: d.pop("pool_m"), "missing ['pool_m']"),
        (lambda d: d.__setitem__("extra", 1), "unexpected ['extra']"),
        (lambda d: d["ib_facets"].pop("sha256"), "ib_facets keys are wrong"),
        (lambda d: d["ib_index"].__setitem__("sha256", "0" * 64), "ib_index keys are wrong"),
        (lambda d: d["pool_facets"].__setitem__("sha256", "0" * 63), "64-hex"),
        (lambda d: d["pool_facets"].__setitem__("sha256", "G" * 64), "64-hex"),
        (lambda d: d["ib_snapshot"].__setitem__("sha256", "A" * 64), "64-hex"),
        (
            lambda d: d["pool_facets"].__setitem__("sha256", "PENDING-MF7B-NPU-PORT-SEAL"),
            "placeholder 'PENDING-MF7B-NPU-PORT-SEAL'",
        ),
        (lambda d: d["ib_snapshot"].__setitem__("relpath", ""), "non-empty relative path"),
        (lambda d: d["ib_snapshot"].__setitem__("relpath", "/abs/path.json"), "relative path"),
        (lambda d: d["pool_m"].__setitem__("relpath", "a/../b.json"), "'..'"),
        (lambda d: d["pool_facets"].__setitem__("tool_contact_ledger_row", " "), "non-empty string"),
        (lambda d: d.__setitem__("schema", "saea-centroid-key-pins-v2"), "schema must be"),
        (lambda d: d.__setitem__("artifact_id", "nested/name"), "bare directory name"),
        (lambda d: d.__setitem__("artifact_id", ""), "bare directory name"),
    ],
)
def test_the_validator_refuses_a_malformed_document_by_name(builder, mutate, needle) -> None:
    document = _good_pins(builder)
    mutate(document)
    with pytest.raises(builder.BuildError, match=__import__("re").escape(needle)):
        builder.validate_pins(document)


def test_the_validator_refuses_a_non_object(builder) -> None:
    with pytest.raises(builder.BuildError, match="must be a JSON object"):
        builder.validate_pins([])


def test_resolve_pins_refuses_a_missing_or_unparsable_document(builder, tmp_path) -> None:
    with pytest.raises(builder.BuildError, match="does not exist"):
        builder.resolve_pins(tmp_path, "nowhere.json")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(builder.BuildError, match="not a UTF-8 JSON document"):
        builder.resolve_pins(tmp_path, "broken.json")
    with pytest.raises(builder.BuildError, match="--pins must be a clean"):
        builder.resolve_pins(tmp_path, "/etc/passwd")


def test_the_committed_port_pins_carry_the_placeholder_and_are_refused(builder) -> None:
    """The port document is committed with the pool-vector digest left as a
    placeholder on purpose: the encoder pass has not sealed yet. Running the
    builder against it must refuse, naming the placeholder, until the main
    thread fills the digest in."""

    document = json.loads((REPO_ROOT / PORT_PINS).read_text(encoding="utf-8"))
    assert document["schema"] == builder.PINS_SCHEMA
    assert document["artifact_id"] == "SAEA-SLURP-INTENT-BOUNDARY-CENTROID-KEYS-NPU-PORT-V1"
    assert document["ib_snapshot"] == {
        "relpath": "knowledge/slurp/semantic/intent-boundaries-npu-port-v1/snapshot.json",
        "sha256": "cb6ef966cb3a68a49c325d0bfa643e9e97402d963a49bc93c0c15024fc62eabc",
    }
    assert document["ib_facets"] == {
        "relpath": "prepared/SAEA-SLURP-INTENT-BOUNDARY-KEY-ENCODING-NPU-PORT-V1/facets.jsonl",
        "sha256": "30e29867b1c94bbf5a9df32f92a4a830adbba9dda9ca0064fed44b1b8c3a3781",
    }
    assert document["ib_index"] == {
        "relpath": "prepared/SAEA-SLURP-INTENT-BOUNDARY-KEY-ENCODING-NPU-PORT-V1/index.json",
    }
    assert document["pool_m"] == {
        "relpath": builder.POOL_M,
        "sha256": builder.POOL_M_SHA,
    }
    assert document["pool_facets"]["relpath"] == builder.POOL_FACETS
    assert document["pool_facets"]["tool_contact_ledger_row"] == (
        "SAEA-SLURP-MF7B-NEMOTRON-EMBEDDING-NPU-PORT-V1"
    )
    assert document["pool_facets"]["sha256"] == "PENDING-MF7B-NPU-PORT-SEAL"
    with pytest.raises(builder.BuildError, match="placeholder 'PENDING-MF7B-NPU-PORT-SEAL'"):
        builder.resolve_pins(REPO_ROOT, PORT_PINS)
    # Once the digest is filled in, the same document validates as-is.
    filled = copy.deepcopy(document)
    filled["pool_facets"]["sha256"] = "b" * 64
    assert builder.validate_pins(filled)["pool_facets"]["sha256"] == "b" * 64


# ---------------------------------------------------------------------------
# (b) the default is the historical build


def test_the_default_pins_are_the_historical_constants(builder) -> None:
    resolved = builder.resolve_pins(REPO_ROOT, None)
    assert resolved["relpath"] is None and resolved["sha256"] is None
    pinned = resolved["document"]
    assert pinned == builder.DEFAULT_PINS
    assert pinned["artifact_id"] == "SAEA-SLURP-INTENT-BOUNDARY-CENTROID-KEYS-V1"
    assert pinned["ib_snapshot"]["sha256"] == builder.IB_SNAPSHOT_SHA
    assert pinned["ib_facets"]["sha256"] == builder.IB_FACETS_SHA
    assert pinned["pool_m"]["sha256"] == builder.POOL_M_SHA
    assert pinned["pool_facets"]["sha256"] == builder.POOL_FACETS_SHA
    assert pinned["pool_facets"]["tool_contact_ledger_row"] == (
        "SAEA-SLURP-MF7B-NEMOTRON-EMBEDDING-V1"
    )
    # The diagnostic's committed constants ARE the default pins, attribute by
    # attribute, so rebinding under the default is the identity.
    diag, diag_sha = builder.import_diagnostic(REPO_ROOT)
    assert diag_sha == _sha(DIAGNOSTIC)
    before = {name: getattr(diag, name) for name, _s, _f in builder.DIAGNOSTIC_BINDINGS}
    for name, section, field in builder.DIAGNOSTIC_BINDINGS:
        assert before[name] == pinned[section][field], name
    builder.bind_diagnostic_pins(diag, pinned)
    assert {name: getattr(diag, name) for name in before} == before


def test_import_diagnostic_proves_the_file_unchanged_whatever_the_pins(builder, tmp_path) -> None:
    """A diagnostic whose committed constants moved is refused even though a
    pins document would rebind them afterwards."""

    repo = tmp_path / "repo"
    (repo / "scripts" / "analysis").mkdir(parents=True)
    text = DIAGNOSTIC.read_text(encoding="utf-8")
    assert builder.POOL_FACETS_SHA in text
    (repo / builder.DIAGNOSTIC_RECIPE).write_text(
        text.replace(builder.POOL_FACETS_SHA, "c" * 64), encoding="utf-8"
    )
    with pytest.raises(builder.BuildError, match="POOL_FACETS_SHA is not the pin"):
        builder.import_diagnostic(repo)


# ---------------------------------------------------------------------------
# (c) end-to-end synthetic build under --pins


def _facet_id(intent: str) -> str:
    return f"intent-boundary:{intent}"


def _vector(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(round(x, 6)) for x in rng.normal(size=WIDTH)]


def _boundary_snapshot() -> KnowledgeSnapshot:
    sample = "slurp:train:000001"
    facets = tuple(
        KeyFacet(
            facet_id=_facet_id(intent),
            dimension="intent-boundary",
            representation_kind="text",
            representation=f"key text for {intent}",
            source_sample_ids=(sample,),
        )
        for intent in INTENTS
    )
    values = tuple(
        ValueBundle(
            value_id=_facet_id(intent),
            content=f"delivered body for {intent}",
            associated_facet_ids=(_facet_id(intent),),
            source_sample_ids=(sample,),
            provenance_refs=(f"prov:{intent}",),
        )
        for intent in INTENTS
    )
    return KnowledgeSnapshot(
        snapshot_id="synthetic-intent-boundaries",
        source_split_identity_hash=SPLIT_HASH,
        source_sample_ids=(sample,),
        builder_policy_id="synthetic-boundary-builder",
        facets=facets,
        values=values,
    ).validate()


def _pool_snapshot() -> tuple[KnowledgeSnapshot, dict[str, str]]:
    """A pool with ``MEMBERS_PER_INTENT`` utterances per intent except the
    orphan; returns the snapshot and ``{pool facet id: intent}``."""

    facets, values, samples, labels = [], [], [], {}
    number = 0
    for intent in INTENTS:
        if intent == ORPHAN:
            continue
        for _ in range(MEMBERS_PER_INTENT):
            number += 1
            facet_id = f"utterance:{number:04d}"
            sample = f"slurp:train:{number:06d}"
            samples.append(sample)
            labels[facet_id] = intent
            facets.append(
                KeyFacet(
                    facet_id=facet_id,
                    dimension="content",
                    representation_kind="text",
                    representation=f"utterance {number}",
                    source_sample_ids=(sample,),
                )
            )
            values.append(
                ValueBundle(
                    value_id=f"value:{number}",
                    content=f"exemplar {number}",
                    associated_facet_ids=(facet_id,),
                    source_sample_ids=(sample,),
                    provenance_refs=(f"prov:{number}",),
                    task_target=intent,
                    prior_score=0.5,
                )
            )
    snapshot = KnowledgeSnapshot(
        snapshot_id="synthetic-pool-minus-r01",
        source_split_identity_hash=SPLIT_HASH,
        source_sample_ids=tuple(samples),
        builder_policy_id="synthetic-pool-builder",
        facets=tuple(facets),
        values=tuple(values),
    ).validate()
    return snapshot, labels


def _write_world(root: Path, repo: Path) -> dict:
    """The synthetic data root and a repo carrying the pins document plus a
    byte-identical copy of the diagnostic; returns what the test asserts on."""

    # The diagnostic, unchanged: the builder imports it from the repo root it
    # is given, and the pins document must live under that same root.
    (repo / "scripts" / "analysis").mkdir(parents=True)
    shutil.copyfile(DIAGNOSTIC, repo / "scripts" / "analysis" / DIAGNOSTIC.name)

    boundary = _boundary_snapshot()
    ib_dir = root / "knowledge" / "synthetic" / "intent-boundaries"
    ib_dir.mkdir(parents=True)
    (ib_dir / "snapshot.json").write_text(json.dumps(boundary.frozen_json()), encoding="utf-8")

    enc_dir = root / "prepared" / "SYNTHETIC-IB-KEY-ENCODING"
    enc_dir.mkdir(parents=True)
    written = {_facet_id(intent): _vector(1000 + i) for i, intent in enumerate(INTENTS)}
    ib_facets = enc_dir / "facets.jsonl"
    ib_facets.write_text(
        "".join(
            json.dumps({"id": fid, "vectors": {"intent-boundary": vec}}, separators=(",", ":"))
            + "\n"
            for fid, vec in written.items()
        ),
        encoding="utf-8",
    )
    index = {
        "schema": "saea-prepared-artifact-v1",
        "inputs": {
            "snapshot_sha256": boundary.content_hash(),
            "dimension": "intent-boundary",
            "facets": [
                {
                    "id": facet.facet_id,
                    "text_sha256": hashlib.sha256(
                        facet.representation.encode("utf-8")
                    ).hexdigest(),
                }
                for facet in boundary.facets
            ],
            "geometry": {"audio": False, "batch_size": 1, "prompt": "passage: "},
            "encoder_identity": {"inventory_sha256": "d" * 64},
        },
    }
    (enc_dir / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")

    pool, labels = _pool_snapshot()
    pool_dir = root / "knowledge" / "synthetic" / "pool-minus-r01"
    pool_dir.mkdir(parents=True)
    (pool_dir / "snapshot.json").write_text(json.dumps(pool.frozen_json()), encoding="utf-8")
    vec_dir = root / "vectors" / "synthetic-kus3"
    vec_dir.mkdir(parents=True)
    pool_vectors = {fid: _vector(i) for i, fid in enumerate(sorted(labels))}
    pool_facets = vec_dir / "facet-vectors.jsonl"
    # The kus3 writer's own line shape (``", "``/``": "`` separators): the
    # diagnostic's streaming loader scans each line head for ``"id": "``.
    pool_facets.write_text(
        "".join(
            json.dumps({"id": fid, "vectors": {"content": vec}}, separators=(", ", ": ")) + "\n"
            for fid, vec in pool_vectors.items()
        )
        # One line the released store does not declare: skipped, never parsed.
        + json.dumps({"id": "utterance:9999", "vectors": {"content": _vector(9999)}}) + "\n",
        encoding="utf-8",
    )

    pins = {
        "schema": "saea-centroid-key-pins-v1",
        "artifact_id": "SYNTHETIC-CENTROID-KEYS",
        "ib_snapshot": {
            "relpath": "knowledge/synthetic/intent-boundaries/snapshot.json",
            "sha256": boundary.content_hash(),
        },
        "ib_facets": {
            "relpath": "prepared/SYNTHETIC-IB-KEY-ENCODING/facets.jsonl",
            "sha256": _sha(ib_facets),
        },
        "ib_index": {"relpath": "prepared/SYNTHETIC-IB-KEY-ENCODING/index.json"},
        "pool_m": {
            "relpath": "knowledge/synthetic/pool-minus-r01/snapshot.json",
            "sha256": pool.content_hash(),
        },
        "pool_facets": {
            "relpath": "vectors/synthetic-kus3/facet-vectors.jsonl",
            "sha256": _sha(pool_facets),
            "tool_contact_ledger_row": "SYNTHETIC-LEDGER-ROW",
        },
    }
    pins_rel = "docs/readiness/synthetic/pins-v1.json"
    (repo / pins_rel).parent.mkdir(parents=True)
    (repo / pins_rel).write_text(json.dumps(pins, indent=2) + "\n", encoding="utf-8")
    return {
        "pins": pins,
        "pins_rel": pins_rel,
        "pins_sha": _sha(repo / pins_rel),
        "written": written,
        "labels": labels,
        "pool_vectors": pool_vectors,
        "index_sha": _sha(enc_dir / "index.json"),
    }


def _expected_centroid(member_rows: list[list[float]]) -> np.ndarray:
    """The diagnostic's arithmetic, restated independently: members are read
    as float32, unit-normalised row-wise in float64, averaged, unit-normalised."""

    block = np.asarray(np.asarray(member_rows, dtype=np.float32), dtype=np.float64)
    block = block / np.linalg.norm(block, axis=1, keepdims=True)
    mean = block.mean(axis=0, keepdims=True)
    return (mean / np.linalg.norm(mean, axis=1, keepdims=True))[0]


def test_a_synthetic_build_under_pins_writes_the_port_shaped_receipt(builder, tmp_path) -> None:
    root, repo, out = tmp_path / "data", tmp_path / "repo", tmp_path / "out"
    world = _write_world(root, repo)

    summary = builder.build(repo, root, out, world["pins_rel"])
    assert summary["pins"] == world["pins_rel"]
    assert summary["rows"] == 60
    assert summary["written_fallback_ids"] == [_facet_id(ORPHAN)]

    out_dir = out / "prepared" / "SYNTHETIC-CENTROID-KEYS"
    facets_path, receipt_path = out_dir / "facets.jsonl", out_dir / "receipt.json"
    assert Path(summary["facets"]) == facets_path and Path(summary["receipt"]) == receipt_path
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    # The receipt names the pins document and every resolved value.
    assert receipt["artifact_id"] == "SYNTHETIC-CENTROID-KEYS"
    assert receipt["artifact"]["relpath"] == "prepared/SYNTHETIC-CENTROID-KEYS/facets.jsonl"
    assert receipt["artifact"]["sha256"] == summary["facets_sha256"] == _sha(facets_path)
    assert receipt["pins"] == {
        "relpath": world["pins_rel"],
        "sha256": world["pins_sha"],
        "schema": "saea-centroid-key-pins-v1",
        "resolved": world["pins"],
    }
    pins = world["pins"]
    inputs = receipt["inputs"]
    assert inputs["target_store_snapshot"]["relpath"] == pins["ib_snapshot"]["relpath"]
    assert inputs["target_store_snapshot"]["sha256"] == pins["ib_snapshot"]["sha256"]
    assert inputs["written_key_vectors"] == {
        "relpath": pins["ib_facets"]["relpath"],
        "sha256": pins["ib_facets"]["sha256"],
        "role": inputs["written_key_vectors"]["role"],
    }
    assert inputs["written_key_encoding_index"]["relpath"] == pins["ib_index"]["relpath"]
    assert inputs["written_key_encoding_index"]["sha256"] == world["index_sha"]
    assert inputs["written_key_encoding_index"]["encoded_is_representation"] == 60
    assert inputs["construction_pool_snapshot"]["relpath"] == pins["pool_m"]["relpath"]
    assert inputs["construction_pool_snapshot"]["sha256"] == pins["pool_m"]["sha256"]
    assert inputs["construction_pool_snapshot"]["utterances"] == 59 * MEMBERS_PER_INTENT
    assert inputs["pool_facet_vectors"]["relpath"] == pins["pool_facets"]["relpath"]
    assert inputs["pool_facet_vectors"]["sha256"] == pins["pool_facets"]["sha256"]
    assert inputs["pool_facet_vectors"]["tool_contact_ledger_row"] == "SYNTHETIC-LEDGER-ROW"
    assert inputs["same_space"]["encoder_inventory_sha256"] == "d" * 64
    assert inputs["same_space"]["geometry"] == {"audio": False, "batch_size": 1, "prompt": "passage: "}
    assert receipt["construction"]["arithmetic_source"]["sha256"] == _sha(DIAGNOSTIC)
    assert receipt["measured"] == {
        **receipt["measured"],
        "intents": 60,
        "centroid_facets": 59,
        "written_fallback_facets": 1,
        "written_fallback_ids": [_facet_id(ORPHAN)],
        "members_total": 59 * MEMBERS_PER_INTENT,
        "members_min_over_centroid_facets": MEMBERS_PER_INTENT,
        "members_max": MEMBERS_PER_INTENT,
        "distinct_pool_labels": 59,
    }

    # The emitted file reads back through the production loader, and the
    # arithmetic is the unit mean of the unit member rows, in member order.
    loaded = load_sealed_vectors(
        facets_path, expected_sha256=summary["facets_sha256"], dimensions=("intent-boundary",)
    )
    assert set(loaded) == {_facet_id(intent) for intent in INTENTS}
    by_record = {record["id"]: record for record in receipt["facets"]}
    members_of: dict[str, list[str]] = {}
    for facet_id in sorted(world["labels"]):
        members_of.setdefault(world["labels"][facet_id], []).append(facet_id)
    for intent in INTENTS:
        facet_id = _facet_id(intent)
        emitted = np.asarray(loaded[facet_id]["intent-boundary"], dtype=np.float64)
        record = by_record[facet_id]
        if intent == ORPHAN:
            assert record["construction"] == "written-fallback"
            assert record["members"] == 0
            assert record["member_ids_sha256"] == hashlib.sha256(b"").hexdigest()
            # Byte for byte the written key: the same floats the flown file carries.
            assert list(emitted) == world["written"][facet_id]
            continue
        chosen = members_of[intent]
        expected = _expected_centroid([world["pool_vectors"][fid] for fid in chosen])
        assert record["construction"] == "centroid"
        assert record["members"] == MEMBERS_PER_INTENT
        assert record["member_ids_sha256"] == hashlib.sha256(
            "\n".join(sorted(chosen)).encode("utf-8")
        ).hexdigest()
        assert np.array_equal(emitted, expected), facet_id
        assert abs(float(np.linalg.norm(emitted)) - 1.0) < 1e-12
        assert record["vector_sha256"] == hashlib.sha256(
            json.dumps([float(x) for x in expected], separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    # A second build refuses to overwrite.
    with pytest.raises(builder.BuildError, match="refusing to overwrite"):
        builder.build(repo, root, out, world["pins_rel"])


def test_a_pins_digest_that_does_not_match_the_bytes_is_refused_before_arithmetic(
    builder, tmp_path
) -> None:
    root, repo, out = tmp_path / "data", tmp_path / "repo", tmp_path / "out"
    world = _write_world(root, repo)
    document = copy.deepcopy(world["pins"])
    document["pool_facets"]["sha256"] = "e" * 64
    (repo / world["pins_rel"]).write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Exception, match="does not hash to the flown pin"):
        builder.build(repo, root, out, world["pins_rel"])
    assert not (out / "prepared").exists()


def test_the_cli_runs_a_pins_build_and_reports_a_refusal_as_json(
    builder, tmp_path, monkeypatch, capsys
) -> None:
    root, repo, out = tmp_path / "data", tmp_path / "repo", tmp_path / "out"
    world = _write_world(root, repo)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    argv = ["--repo-root", str(repo), "--data-root", str(root), "--output-root", str(out)]
    assert builder.main(argv + ["--pins", world["pins_rel"]]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["pins"] == world["pins_rel"] and summary["rows"] == 60
    # The same argv again: the overwrite refusal is reported, not raised.
    assert builder.main(argv + ["--pins", world["pins_rel"]]) == 1
    captured = capsys.readouterr()
    assert "refusing to overwrite" in json.loads(captured.err)["error"]
