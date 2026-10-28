"""Serving views: a Qwen2.5-Omni Thinker embedding checkpoint in the layout vLLM >= 0.30 loads natively.

Port note (npu-vllm-port, 2026-10-28). The speech encoders of this study (Omni-Embed-Nemotron-3B, and the LCO-Embedding
Omni 3B / 7B arms) all ship a bare Qwen2.5-Omni Thinker checkpoint: tensors named ``model.*``, ``audio_tower.*`` and
``visual.*``, no ``lm_head``, and a Thinker-level ``config.json``. vLLM 0.30 serves this family only through
``Qwen2_5OmniThinkerForConditionalGeneration`` registered under the FULL Omni architectures, which (a) reads
``hf_config.thinker_config`` from a ``qwen2_5_omni`` config and (b) maps tensors with the ``thinker.`` prefix
(``thinker.model.`` -> ``language_model.model.``). A bare ``model.*`` tensor has no target there and fails the load.

A serving view is a NEW directory that differs from its source in exactly these declared ways, and nothing else:

weights   every tensor name gains the ``thinker.`` prefix. Only the safetensors JSON header is rewritten; the tensor
          data region is copied byte for byte and its SHA-256 is recorded on both sides and required equal.
config    the source config becomes ``thinker_config`` of a ``qwen2_5_omni`` config (architecture
          ``Qwen2_5OmniForConditionalGeneration``, ``enable_audio_output`` false). ``auto_map`` is dropped, so no remote
          code is ever loaded. When the checkpoint carries no ``lm_head``, ``text_config.tie_word_embeddings`` is set
          true: the pooling runner never computes logits, and an untied absent head would fail vLLM's strict
          all-weights-loaded check.
template  ``chat_template.jinja`` is the source's ``additional_chat_templates/sentence_transformers.jinja`` byte for
          byte -- the template the study's encode recipe renders with -- and the serve arguments pin it explicitly.

Attention stays CAUSAL. Omni-Embed-Nemotron's remote modeling file only overrides the Thinker to bidirectional
attention; that override is deliberately not carried (owner decision 2026-10-28: variable-length speech, in-context
steering). This is a declared model difference from the historical sealed vectors, which must be regenerated.

Every other file the server reads (tokenizer, processor, generation config) is copied verbatim. The receipt
(``serving_view.json``) records the source and view hash of every file, the recipe read from the source's
sentence-transformers modules, and the exact list of differences; :func:`verify_serving_view` re-hashes the view
against it and refuses any extra, missing or changed file.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

SERVING_VIEW_SCHEMA = "saea-vllm-serving-view-v1"
RECEIPT_NAME = "serving_view.json"
THINKER_PREFIX = "thinker."
OMNI_ARCHITECTURE = "Qwen2_5OmniForConditionalGeneration"
OMNI_MODEL_TYPE = "qwen2_5_omni"
THINKER_MODEL_TYPE = "qwen2_5_omni_thinker"
TEXT_MODEL_TYPE = "qwen2_5_omni_text"
SOURCE_THINKER_MODEL_TYPES = frozenset({THINKER_MODEL_TYPE, "nvomniembed"})
SOURCE_TEMPLATE = "additional_chat_templates/sentence_transformers.jinja"
VIEW_TEMPLATE = "chat_template.jinja"
#: Tensor-name roots of a bare Thinker checkpoint; anything else is refused rather than guessed at.
THINKER_TENSOR_ROOTS = ("model.", "audio_tower.", "visual.", "lm_head.")
#: Files copied verbatim when the source has them (the first five are required).
VERBATIM_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "generation_config.json",
)
_REQUIRED_VERBATIM = frozenset(VERBATIM_FILES[:5])
#: Source config keys that only make sense for remote code or one training run.
_DROPPED_CONFIG_KEYS = ("architectures", "auto_map", "_attn_implementation_autoset")
#: sentence-transformers pooling mode -> vLLM SequencePoolingType.
_POOLING = {"mean": "MEAN", "lasttoken": "LAST"}
#: The sentence-transformers module chain this port reproduces: forward -> pool -> L2 normalize.
_MODULE_CHAIN = (
    "sentence_transformers.base.modules.transformer.Transformer",
    "sentence_transformers.sentence_transformer.modules.pooling.Pooling",
    "sentence_transformers.sentence_transformer.modules.normalize.Normalize",
)
_COPY_CHUNK = 64 * 1024 * 1024
_MAX_HEADER = 100 * 1024 * 1024


class ServingViewError(ValueError):
    """Refuse a source, a view or a receipt that is not exactly what this module declares."""


@dataclass(frozen=True)
class EmbeddingRecipe:
    """The encode recipe a view serves, read from the source's sentence-transformers modules."""

    pooling: str
    normalize: bool
    dimensions: int
    prompts: Mapping[str, str]
    template_sha256: str

    def as_json(self) -> dict[str, object]:
        return {
            "pooling": self.pooling,
            "normalize": self.normalize,
            "dimensions": self.dimensions,
            "prompts": dict(sorted(self.prompts.items())),
            "template_sha256": self.template_sha256,
        }


def _sha256_file(path: Path) -> tuple[int, str]:
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as handle:
        while chunk := handle.read(_COPY_CHUNK):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _read_json(path: Path, label: str) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ServingViewError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(document, dict):
        raise ServingViewError(f"{label} {path} is not a JSON object")
    return document


def _canonical(document: Mapping[str, object]) -> bytes:
    return (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    """The parsed JSON header of one safetensors file and the byte offset where its data region starts."""

    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ServingViewError(f"{path} is not a safetensors file")
        (length,) = struct.unpack("<Q", prefix)
        if not 0 < length <= _MAX_HEADER:
            raise ServingViewError(f"{path} declares an implausible header length {length}")
        raw = handle.read(length)
    if len(raw) != length:
        raise ServingViewError(f"{path} header is truncated")
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as error:
        raise ServingViewError(f"{path} header is not JSON: {error}") from error
    if not isinstance(header, dict):
        raise ServingViewError(f"{path} header is not a JSON object")
    return header, 8 + length


def _encode_header(header: Mapping[str, object]) -> bytes:
    raw = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    # The safetensors format allows trailing spaces in the header; pad so the data region stays 8-byte aligned.
    raw += b" " * (-(8 + len(raw)) % 8)
    return struct.pack("<Q", len(raw)) + raw


def renamed_header(header: Mapping[str, object]) -> dict[str, object]:
    """``header`` with every tensor name prefixed by ``thinker.``; offsets are relative to the data region, unchanged."""

    renamed: dict[str, object] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            renamed[name] = entry
            continue
        if not name.startswith(THINKER_TENSOR_ROOTS):
            raise ServingViewError(f"tensor {name!r} is not under a Thinker root {THINKER_TENSOR_ROOTS}")
        renamed[THINKER_PREFIX + name] = entry
    return renamed


def _data_region_end(header: Mapping[str, object]) -> int:
    ends = [entry["data_offsets"][1] for name, entry in header.items() if name != "__metadata__"]
    return max(ends, default=0)


def rewrite_shard(source: Path, target: Path) -> dict[str, object]:
    """Write ``target`` = renamed header + the source data region verbatim; return both sides' identities."""

    header, data_start = read_safetensors_header(source)
    size = source.stat().st_size
    if _data_region_end(header) != size - data_start:
        raise ServingViewError(f"{source} data region does not end where its tensors end")
    source_digest, data_digest, target_digest = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    encoded = _encode_header(renamed_header(header))
    with source.open("rb") as reader, target.open("xb") as writer:
        head = reader.read(data_start)
        source_digest.update(head)
        writer.write(encoded)
        target_digest.update(encoded)
        while chunk := reader.read(_COPY_CHUNK):
            source_digest.update(chunk)
            data_digest.update(chunk)
            target_digest.update(chunk)
            writer.write(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    if source.stat().st_size != size:
        raise ServingViewError(f"{source} changed while it was copied")
    _, written_data_start = read_safetensors_header(target)
    written = hashlib.sha256()
    with target.open("rb") as handle:
        handle.seek(written_data_start)
        while chunk := handle.read(_COPY_CHUNK):
            written.update(chunk)
    if written.hexdigest() != data_digest.hexdigest():
        raise ServingViewError(f"{target} data region differs from {source}")
    return {
        "source_sha256": source_digest.hexdigest(),
        "source_bytes": size,
        "data_sha256": data_digest.hexdigest(),
        "data_bytes": size - data_start,
        "view_sha256": target_digest.hexdigest(),
        "view_bytes": len(encoded) + size - data_start,
        "tensors": len([name for name in header if name != "__metadata__"]),
    }


def read_recipe(source: Path) -> EmbeddingRecipe:
    """The source's sentence-transformers recipe; anything this port does not reproduce exactly is refused."""

    pooling = _read_json(source / "1_Pooling/config.json", "pooling config")
    mode = pooling.get("pooling_mode")
    if mode not in _POOLING:
        raise ServingViewError(f"pooling_mode {mode!r} is not one of {sorted(_POOLING)}")
    if pooling.get("include_prompt") is not True:
        raise ServingViewError("include_prompt must be true: vLLM pools over every prompt token")
    dimensions = pooling.get("embedding_dimension")
    if type(dimensions) is not int or dimensions <= 0:
        raise ServingViewError("embedding_dimension must be a positive integer")
    try:
        modules = json.loads((source / "modules.json").read_text(encoding="utf-8"))
        types = tuple(module["type"] for module in modules)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ServingViewError(f"cannot read modules.json: {error}") from error
    if types != _MODULE_CHAIN:
        raise ServingViewError(f"modules.json must be {_MODULE_CHAIN}, not {types}")
    settings = _read_json(source / "config_sentence_transformers.json", "sentence-transformers config")
    prompts = settings.get("prompts")
    if not isinstance(prompts, dict) or not all(isinstance(v, str) for v in prompts.values()):
        raise ServingViewError("sentence-transformers prompts must be a string map")
    _, template_sha256 = _sha256_file(source / SOURCE_TEMPLATE)
    return EmbeddingRecipe(
        pooling=_POOLING[mode], normalize=True, dimensions=dimensions, prompts=prompts,
        template_sha256=template_sha256,
    )


def view_config(source_config: Mapping[str, object], *, has_lm_head: bool) -> tuple[dict[str, object], list[str]]:
    """The view's ``config.json`` and the human-readable list of every difference from the source config."""

    if source_config.get("model_type") not in SOURCE_THINKER_MODEL_TYPES:
        raise ServingViewError(
            f"source model_type {source_config.get('model_type')!r} is not a Thinker {sorted(SOURCE_THINKER_MODEL_TYPES)}"
        )
    text = source_config.get("text_config")
    if not isinstance(text, dict) or text.get("model_type") != TEXT_MODEL_TYPE:
        raise ServingViewError(f"source text_config must be {TEXT_MODEL_TYPE!r}")
    changes: list[str] = []
    thinker = {key: value for key, value in source_config.items() if key not in _DROPPED_CONFIG_KEYS}
    for key in _DROPPED_CONFIG_KEYS:
        if key in source_config:
            changes.append(f"thinker_config.{key}: dropped")
    if thinker["model_type"] != THINKER_MODEL_TYPE:
        changes.append(f"thinker_config.model_type: {thinker['model_type']!r} -> {THINKER_MODEL_TYPE!r}")
        thinker["model_type"] = THINKER_MODEL_TYPE
    if not has_lm_head and text.get("tie_word_embeddings") is not True:
        thinker["text_config"] = {**text, "tie_word_embeddings": True}
        changes.append(
            f"thinker_config.text_config.tie_word_embeddings: {text.get('tie_word_embeddings')!r} -> True "
            "(no lm_head is shipped; pooling never computes logits)"
        )
    dtype = source_config.get("torch_dtype", source_config.get("dtype"))
    config = {
        "architectures": [OMNI_ARCHITECTURE],
        "model_type": OMNI_MODEL_TYPE,
        "enable_audio_output": False,
        "thinker_config": thinker,
    }
    if dtype is not None:
        config["torch_dtype"] = dtype
    changes.append(
        f"wrapped as thinker_config of a {OMNI_MODEL_TYPE!r} config, architecture {OMNI_ARCHITECTURE!r}, "
        "enable_audio_output false"
    )
    return config, changes


def build_serving_view(source: Path, view: Path, *, view_id: str) -> dict[str, object]:
    """Build ``view`` from ``source`` (see the module docstring) and return its receipt.

    The view is assembled in a sibling temporary directory and renamed into place only when complete, so a crashed
    build never leaves a directory that looks like a view. An existing ``view`` is never touched.
    """

    source, view = Path(source).resolve(strict=True), Path(view)
    if not view_id or "/" in view_id:
        raise ServingViewError("view_id must be a non-empty single path segment")
    if view.exists():
        raise ServingViewError(f"{view} already exists; a serving view is never rebuilt in place")
    recipe = read_recipe(source)
    index = _read_json(source / "model.safetensors.index.json", "weight index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ServingViewError("weight index has no weight_map")
    shards = sorted(set(weight_map.values()))
    has_lm_head = any(name.startswith("lm_head.") for name in weight_map)
    config, changes = view_config(_read_json(source / "config.json", "config"), has_lm_head=has_lm_head)
    missing = sorted(_REQUIRED_VERBATIM - {name for name in VERBATIM_FILES if (source / name).is_file()})
    if missing:
        raise ServingViewError(f"source lacks required files {missing}")

    staging = view.parent / f".{view.name}.building-{os.getpid()}"
    staging.mkdir(parents=True)
    try:
        files: dict[str, dict[str, object]] = {}
        shard_records: dict[str, dict[str, object]] = {}
        for shard in shards:
            record = rewrite_shard(source / shard, staging / shard)
            shard_records[shard] = record
            files[shard] = {
                "source_sha256": record["source_sha256"], "source_bytes": record["source_bytes"],
                "view_sha256": record["view_sha256"], "view_bytes": record["view_bytes"],
            }
        renamed_index = {
            **index,
            "weight_map": {THINKER_PREFIX + name: shard for name, shard in sorted(weight_map.items())},
        }
        for name, document in (("model.safetensors.index.json", renamed_index), ("config.json", config)):
            (staging / name).write_bytes(_canonical(document))
        for name in ("model.safetensors.index.json", "config.json"):
            source_bytes, source_sha256 = _sha256_file(source / name)
            view_bytes, view_sha256 = _sha256_file(staging / name)
            files[name] = {"source_sha256": source_sha256, "source_bytes": source_bytes,
                           "view_sha256": view_sha256, "view_bytes": view_bytes}
        copies = [(name, name) for name in VERBATIM_FILES if (source / name).is_file()]
        copies.append((SOURCE_TEMPLATE, VIEW_TEMPLATE))
        for source_name, view_name in copies:
            shutil.copyfile(source / source_name, staging / view_name)
            source_bytes, source_sha256 = _sha256_file(source / source_name)
            view_bytes, view_sha256 = _sha256_file(staging / view_name)
            if (source_bytes, source_sha256) != (view_bytes, view_sha256):
                raise ServingViewError(f"{source_name} changed while it was copied")
            files[view_name] = {"source": source_name, "source_sha256": source_sha256, "source_bytes": source_bytes,
                                "view_sha256": view_sha256, "view_bytes": view_bytes}
        if (source / "modeling_nv_omni_embed.py").is_file():
            changes.append("modeling_nv_omni_embed.py (bidirectional attention override): not carried; attention is causal")
        changes.append(f"every tensor name prefixed with {THINKER_PREFIX!r}; tensor data regions byte-identical")
        changes.append(f"{VIEW_TEMPLATE} = source {SOURCE_TEMPLATE}")
        receipt = {
            "schema": SERVING_VIEW_SCHEMA,
            "view_id": view_id,
            "source_dir": str(source),
            "architecture": OMNI_ARCHITECTURE,
            "attention": "causal",
            "recipe": recipe.as_json(),
            "differences": changes,
            "files": dict(sorted(files.items())),
            "shards": dict(sorted(shard_records.items())),
        }
        (staging / RECEIPT_NAME).write_bytes(_canonical(receipt))
        os.rename(staging, view)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return receipt


def receipt_sha256(view: Path) -> str:
    return _sha256_file(Path(view) / RECEIPT_NAME)[1]


def verify_serving_view(view: Path, *, expected_receipt_sha256: str | None = None) -> dict[str, object]:
    """Re-hash every file of ``view`` against its receipt; refuse an extra, missing or changed file."""

    view = Path(view)
    receipt = _read_json(view / RECEIPT_NAME, "serving view receipt")
    if receipt.get("schema") != SERVING_VIEW_SCHEMA:
        raise ServingViewError(f"receipt schema must be {SERVING_VIEW_SCHEMA!r}")
    if expected_receipt_sha256 is not None and receipt_sha256(view) != expected_receipt_sha256:
        raise ServingViewError("serving view receipt differs from the pinned receipt hash")
    files = receipt.get("files")
    if not isinstance(files, dict) or not files:
        raise ServingViewError("serving view receipt lists no files")
    present = {
        str(path.relative_to(view)) for path in view.rglob("*") if path.is_file() or path.is_symlink()
    } - {RECEIPT_NAME}
    if present != set(files):
        raise ServingViewError(
            f"serving view files differ from its receipt: extra {sorted(present - set(files))}, "
            f"missing {sorted(set(files) - present)}"
        )
    for name, entry in files.items():
        path = view / name
        if path.is_symlink():
            raise ServingViewError(f"{name} is a symlink; a serving view holds its own bytes")
        if _sha256_file(path) != (entry["view_bytes"], entry["view_sha256"]):
            raise ServingViewError(f"{name} differs from its receipt")
    return receipt


def vllm_embedding_serve_args(
    view: Path,
    receipt: Mapping[str, object],
    *,
    served_model_name: str,
    max_model_len: int,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> list[str]:
    """The ``vllm serve`` argument vector that serves ``view`` with exactly its recipe."""

    recipe = receipt["recipe"]
    pooler = {"seq_pooling_type": recipe["pooling"], "use_activation": bool(recipe["normalize"])}
    return [
        "vllm", "serve", str(view),
        "--served-model-name", served_model_name,
        "--host", host, "--port", str(port),
        "--runner", "pooling",
        "--convert", "embed",
        "--pooler-config", json.dumps(pooler, sort_keys=True, separators=(",", ":")),
        "--chat-template", str(Path(view) / VIEW_TEMPLATE),
        "--chat-template-content-format", "openai",
        "--dtype", "bfloat16",
        "--max-model-len", str(max_model_len),
        "--limit-mm-per-prompt", json.dumps({"audio": 1, "image": 0, "video": 0}, separators=(",", ":")),
        "--seed", "0",
    ]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build or verify a vLLM serving view of a Thinker embedding checkpoint.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("source", type=Path)
    build.add_argument("view", type=Path)
    build.add_argument("--view-id", required=True)
    check = sub.add_parser("verify")
    check.add_argument("view", type=Path)
    check.add_argument("--receipt-sha256")
    args = parser.parse_args(argv)
    if args.command == "build":
        build_serving_view(args.source, args.view, view_id=args.view_id)
    else:
        verify_serving_view(args.view, expected_receipt_sha256=args.receipt_sha256)
    print(f"{args.view} receipt_sha256={receipt_sha256(args.view)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
