"""``core.vllm_serving_view``: a bare Thinker embedding checkpoint rewritten into the layout vLLM 0.30 loads.

Every test builds a tiny synthetic checkpoint with the real file set of Omni-Embed-Nemotron-3B (config, index, shards,
tokenizer/processor files, sentence-transformers modules and template). No model is loaded.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct

import pytest

from speech_aware_evidence_acquisition.core import vllm_serving_view as sv

_MODULES = [
    {"idx": 0, "name": "0", "path": "", "type": sv._MODULE_CHAIN[0]},
    {"idx": 1, "name": "1", "path": "1_Pooling", "type": sv._MODULE_CHAIN[1]},
    {"idx": 2, "name": "2", "path": "2_Normalize", "type": sv._MODULE_CHAIN[2]},
]
_TEMPLATE = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{{ x }}<|im_end|>\n"


def _write_safetensors(path, tensors):
    header, blobs, offset = {"__metadata__": {"format": "pt"}}, [], 0
    for name, values in tensors.items():
        blob = struct.pack(f"<{len(values)}f", *values)
        header[name] = {"dtype": "F32", "shape": [len(values)], "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def _checkpoint(root, *, model_type="nvomniembed", pooling="mean", lm_head=False, extra_tensor=None):
    root.mkdir(parents=True)
    shard_a = {"model.embed_tokens.weight": [1.0, 2.0, 3.0], "model.layers.0.mlp.up_proj.weight": [4.0, 5.0]}
    shard_b = {"audio_tower.conv1.weight": [6.0], "visual.merger.mlp.0.bias": [7.0, 8.0]}
    if lm_head:
        shard_b["lm_head.weight"] = [9.0]
    if extra_tensor:
        shard_b[extra_tensor] = [10.0]
    _write_safetensors(root / "model-00001-of-00002.safetensors", shard_a)
    _write_safetensors(root / "model-00002-of-00002.safetensors", shard_b)
    weight_map = {name: "model-00001-of-00002.safetensors" for name in shard_a}
    weight_map.update({name: "model-00002-of-00002.safetensors" for name in shard_b})
    (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map}))
    config = {
        "architectures": ["NVOmniEmbedModel"],
        "auto_map": {"AutoModel": "modeling_nv_omni_embed.NVOmniEmbedModel"},
        "model_type": model_type,
        "torch_dtype": "bfloat16",
        "audio_token_index": 151646,
        "text_config": {"model_type": "qwen2_5_omni_text", "hidden_size": 2048, "num_hidden_layers": 36},
        "audio_config": {"model_type": "qwen2_5_omni_audio_encoder"},
        "vision_config": {"model_type": "qwen2_5_omni_vision_encoder"},
    }
    (root / "config.json").write_text(json.dumps(config))
    for name in sv.VERBATIM_FILES:
        (root / name).write_text(json.dumps({"file": name}))
    (root / "modeling_nv_omni_embed.py").write_text("# bidirectional override\n")
    (root / "1_Pooling").mkdir()
    (root / "1_Pooling/config.json").write_text(
        json.dumps({"embedding_dimension": 2048, "pooling_mode": pooling, "include_prompt": True})
    )
    (root / "modules.json").write_text(json.dumps(_MODULES))
    (root / "config_sentence_transformers.json").write_text(json.dumps({"prompts": {"query": "query: ", "document": "passage: "}}))
    (root / "additional_chat_templates").mkdir()
    (root / sv.SOURCE_TEMPLATE).write_text(_TEMPLATE)
    (root / "chat_template.jinja").write_text("the generic Qwen template, not the recipe's")
    return root


def _data_region(path):
    header, start = sv.read_safetensors_header(path)
    return header, path.read_bytes()[start:]


def test_the_view_renames_every_tensor_and_keeps_every_data_byte(tmp_path):
    source = _checkpoint(tmp_path / "src")
    receipt = sv.build_serving_view(source, tmp_path / "view", view_id="nemotron-3b-causal-v1")
    view = tmp_path / "view"
    for shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        source_header, source_data = _data_region(source / shard)
        view_header, view_data = _data_region(view / shard)
        assert view_data == source_data
        assert sv.read_safetensors_header(view / shard)[1] % 8 == 0
        assert view_header["__metadata__"] == source_header["__metadata__"]
        assert {k: v for k, v in view_header.items() if k != "__metadata__"} == {
            "thinker." + k: v for k, v in source_header.items() if k != "__metadata__"
        }
        assert receipt["shards"][shard]["data_sha256"] == hashlib.sha256(source_data).hexdigest()
    index = json.loads((view / "model.safetensors.index.json").read_text())
    assert all(name.startswith("thinker.") for name in index["weight_map"])
    assert len(index["weight_map"]) == 4


def test_the_view_shards_load_with_safetensors_under_the_new_names(tmp_path):
    safetensors = pytest.importorskip("safetensors")
    source = _checkpoint(tmp_path / "src")
    sv.build_serving_view(source, tmp_path / "view", view_id="v")
    with safetensors.safe_open(str(tmp_path / "view/model-00001-of-00002.safetensors"), framework="numpy") as handle:
        assert sorted(handle.keys()) == ["thinker.model.embed_tokens.weight", "thinker.model.layers.0.mlp.up_proj.weight"]
        assert handle.get_tensor("thinker.model.embed_tokens.weight").tolist() == [1.0, 2.0, 3.0]


def test_every_view_tensor_lands_on_a_vllm_thinker_module_under_its_mapper():
    # vLLM 0.30 Qwen2_5OmniThinkerForConditionalGeneration.hf_to_vllm_mapper, applied in order.
    mapper = (("thinker.lm_head.", "language_model.lm_head."), ("thinker.model.", "language_model.model."), ("thinker.", ""))

    def mapped(name):
        for old, new in mapper:
            if name.startswith(old):
                return new + name[len(old):]
        return name

    roots = ("language_model.model.", "language_model.lm_head.", "audio_tower.", "visual.")
    for root in sv.THINKER_TENSOR_ROOTS:
        assert mapped(sv.THINKER_PREFIX + root + "x").startswith(roots)
    # why the rename is needed: a bare language-model tensor has no Thinker module to land on
    assert mapped("model.layers.0.x") == "model.layers.0.x"


def test_the_config_is_a_qwen2_5_omni_config_without_remote_code_and_with_a_tied_absent_head(tmp_path):
    source = _checkpoint(tmp_path / "src")
    receipt = sv.build_serving_view(source, tmp_path / "view", view_id="v")
    config = json.loads((tmp_path / "view/config.json").read_text())
    assert config["architectures"] == ["Qwen2_5OmniForConditionalGeneration"]
    assert config["model_type"] == "qwen2_5_omni" and config["enable_audio_output"] is False
    thinker = config["thinker_config"]
    assert "auto_map" not in thinker and "architectures" not in thinker
    assert thinker["model_type"] == "qwen2_5_omni_thinker"
    assert thinker["text_config"]["tie_word_embeddings"] is True
    assert thinker["audio_token_index"] == 151646
    assert any("tie_word_embeddings" in line for line in receipt["differences"])
    assert any("not carried; attention is causal" in line for line in receipt["differences"])
    assert receipt["attention"] == "causal"


def test_transformers_reads_the_view_config_as_the_omni_config_vllm_expects(tmp_path):
    transformers = pytest.importorskip("transformers")
    source = _checkpoint(tmp_path / "src")
    sv.build_serving_view(source, tmp_path / "view", view_id="v")
    config = transformers.AutoConfig.from_pretrained(str(tmp_path / "view"))
    assert type(config).__name__ == "Qwen2_5OmniConfig"
    assert type(config.thinker_config).__name__ == "Qwen2_5OmniThinkerConfig"
    assert config.thinker_config.text_config.tie_word_embeddings is True
    assert config.thinker_config.text_config.hidden_size == 2048


def test_a_shipped_head_is_left_untied(tmp_path):
    source = _checkpoint(tmp_path / "src", model_type="qwen2_5_omni_thinker", lm_head=True)
    receipt = sv.build_serving_view(source, tmp_path / "view", view_id="v")
    thinker = json.loads((tmp_path / "view/config.json").read_text())["thinker_config"]
    assert "tie_word_embeddings" not in thinker["text_config"]
    assert not any("tie_word_embeddings" in line for line in receipt["differences"])


def test_the_recipe_template_replaces_the_generic_one_and_files_are_copied_verbatim(tmp_path):
    source = _checkpoint(tmp_path / "src")
    receipt = sv.build_serving_view(source, tmp_path / "view", view_id="v")
    view = tmp_path / "view"
    assert (view / "chat_template.jinja").read_text() == _TEMPLATE
    assert not (view / "modeling_nv_omni_embed.py").exists()
    for name in sv.VERBATIM_FILES:
        assert (view / name).read_bytes() == (source / name).read_bytes()
    assert receipt["files"]["chat_template.jinja"]["source"] == sv.SOURCE_TEMPLATE
    assert receipt["recipe"] == {
        "pooling": "MEAN", "normalize": True, "dimensions": 2048,
        "prompts": {"document": "passage: ", "query": "query: "},
        "template_sha256": hashlib.sha256(_TEMPLATE.encode()).hexdigest(),
    }


def test_lasttoken_pooling_is_served_as_last(tmp_path):
    source = _checkpoint(tmp_path / "src", model_type="qwen2_5_omni_thinker", pooling="lasttoken")
    receipt = sv.build_serving_view(source, tmp_path / "view", view_id="v")
    assert receipt["recipe"]["pooling"] == "LAST"
    args = sv.vllm_embedding_serve_args(tmp_path / "view", receipt, served_model_name="lco-3b", max_model_len=8192)
    assert args[args.index("--pooler-config") + 1] == '{"seq_pooling_type":"LAST","use_activation":true}'
    assert args[args.index("--chat-template") + 1] == str(tmp_path / "view/chat_template.jinja")
    assert args[args.index("--runner") + 1] == "pooling" and args[args.index("--convert") + 1] == "embed"


def test_verify_accepts_the_built_view_and_pins_the_receipt(tmp_path):
    source = _checkpoint(tmp_path / "src")
    sv.build_serving_view(source, tmp_path / "view", view_id="v")
    pinned = sv.receipt_sha256(tmp_path / "view")
    assert sv.verify_serving_view(tmp_path / "view", expected_receipt_sha256=pinned)["view_id"] == "v"
    with pytest.raises(sv.ServingViewError, match="pinned receipt"):
        sv.verify_serving_view(tmp_path / "view", expected_receipt_sha256="0" * 64)


@pytest.mark.parametrize("tamper", ["extra", "missing", "changed", "symlink"])
def test_verify_refuses_any_drift(tmp_path, tamper):
    source = _checkpoint(tmp_path / "src")
    view = tmp_path / "view"
    sv.build_serving_view(source, view, view_id="v")
    if tamper == "extra":
        (view / "modeling_nv_omni_embed.py").write_text("x")
    elif tamper == "missing":
        (view / "vocab.json").unlink()
    elif tamper == "changed":
        (view / "chat_template.jinja").write_text(_TEMPLATE + " ")
    else:
        (view / "vocab.json").unlink()
        os.symlink(source / "vocab.json", view / "vocab.json")
    with pytest.raises(sv.ServingViewError):
        sv.verify_serving_view(view)


def test_build_never_touches_an_existing_view_and_leaves_nothing_on_failure(tmp_path):
    source = _checkpoint(tmp_path / "src")
    (tmp_path / "view").mkdir()
    with pytest.raises(sv.ServingViewError, match="already exists"):
        sv.build_serving_view(source, tmp_path / "view", view_id="v")
    bad = _checkpoint(tmp_path / "bad", extra_tensor="talker.codec_head.weight")
    with pytest.raises(sv.ServingViewError, match="Thinker root"):
        sv.build_serving_view(bad, tmp_path / "view2", view_id="v")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bad", "src", "view"]


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda root: (root / "1_Pooling/config.json").write_text(json.dumps({"embedding_dimension": 8, "pooling_mode": "cls", "include_prompt": True})), "pooling_mode"),
        (lambda root: (root / "1_Pooling/config.json").write_text(json.dumps({"embedding_dimension": 8, "pooling_mode": "mean", "include_prompt": False})), "include_prompt"),
        (lambda root: (root / "modules.json").write_text(json.dumps(_MODULES[:2])), "modules.json"),
        (lambda root: (root / "tokenizer.json").unlink(), "required files"),
        (lambda root: (root / "config.json").write_text(json.dumps({"model_type": "qwen2_5_omni", "text_config": {}})), "not a Thinker"),
    ],
)
def test_build_refuses_a_source_it_cannot_reproduce_exactly(tmp_path, mutate, match):
    source = _checkpoint(tmp_path / "src")
    mutate(source)
    with pytest.raises(sv.ServingViewError, match=match):
        sv.build_serving_view(source, tmp_path / "view", view_id="v")
