"""One serving geometry, two server dialects: llama-server's ``-np``/``-c`` and vLLM's ``--max-model-len``.

The vLLM dialect is protocol-only (owner decision 2026-10-28): ``--max-model-len`` is the hashed per-sequence window;
``--max-num-seqs`` is a deployment fact owned by the deployment profile, so a vllm fragment must not declare it and the
merge never writes it.
"""

from __future__ import annotations

import pytest

from speech_aware_evidence_acquisition.core.flight.runtime import FlightRuntimeError
from speech_aware_evidence_acquisition.core.flight.runtime import merge_server_args as flight_merge
from speech_aware_evidence_acquisition.core.session.serving import (
    ServingSessionError,
    declared_server_geometry,
    merge_server_args,
    resolve_serving_profile,
)
from test_serving_session import _fragment_document, _repo_with_fragment, _runtime_world

_VLLM_ARGS = ["--max-model-len", "24576", "--reasoning-parser", "qwen3"]


def test_the_llama_dialect_is_the_default_and_is_unchanged():
    args = ["-ngl", "999", "-np", "1", "-c", "10"]
    assert merge_server_args(args, slots=4, context_tokens=98304) == ["-ngl", "999", "-np", "4", "-c", "98304"]
    assert merge_server_args(args, slots=4, context_tokens=98304, transport="llama-server") == (
        merge_server_args(args, slots=4, context_tokens=98304)
    )
    assert declared_server_geometry(["-c", "98304", "-np", "4"]) == (4, 24576)


def test_vllm_writes_only_the_per_sequence_window_never_the_deployment_slot_flag():
    merged = merge_server_args(_VLLM_ARGS, slots=4, context_tokens=4 * 24576, transport="vllm")
    assert merged == ["--max-model-len", "24576", "--reasoning-parser", "qwen3"]
    appended = merge_server_args(["--dtype", "bfloat16"], slots=2, context_tokens=2 * 8192, transport="vllm")
    assert appended == ["--dtype", "bfloat16", "--max-model-len", "8192"]
    # The client-side slot count only divides the total into a window; it never reaches the argv.
    assert merge_server_args(_VLLM_ARGS, slots=9, context_tokens=9 * 24576, transport="vllm") == merged
    # Sequences per instance is not declared by a vllm fragment, so there is no slot count to read back.
    assert declared_server_geometry(_VLLM_ARGS, transport="vllm") == (None, 24576)


@pytest.mark.parametrize("spelling", [["--max-num-seqs", "4"], ["--max-num-seqs=4"]])
def test_a_vllm_fragment_that_declares_sequences_per_instance_is_refused(spelling):
    args = [*_VLLM_ARGS, *spelling]
    # The separate-value spelling is refused as a deployment fact; the inline spelling by the one-spelling rule.
    match = "deployment fact" if len(spelling) == 2 else "only as"
    with pytest.raises(ServingSessionError, match=match):
        merge_server_args(args, slots=4, context_tokens=4 * 24576, transport="vllm")
    with pytest.raises(ServingSessionError, match=match):
        declared_server_geometry(args, transport="vllm")
    # The llama dialect has no such flag; the same tokens are just opaque server flags there.
    assert merge_server_args(["-np", "1", "-c", "8", *spelling], slots=4, context_tokens=98304) == [
        "-np", "4", "-c", "98304", *spelling,
    ]


@pytest.mark.parametrize(
    "args, match",
    [
        (["-np", "4", "--max-model-len", "24576"], "only as"),
        (["--max-model-len=24576"], "only as"),
        (["--ctx-size", "98304"], "only as"),
    ],
)
def test_a_vllm_argv_carries_one_dialect_in_one_spelling(args, match):
    with pytest.raises(ServingSessionError, match=match):
        merge_server_args(args, slots=4, context_tokens=4 * 24576, transport="vllm")
    with pytest.raises(ServingSessionError, match=match):
        declared_server_geometry(args, transport="vllm")


def test_vllm_refuses_a_total_context_that_does_not_split_into_whole_slots():
    with pytest.raises(ServingSessionError, match="whole number of tokens per slot"):
        merge_server_args([], slots=3, context_tokens=1000, transport="vllm")


def test_vllm_declared_window_must_be_present_and_positive():
    with pytest.raises(ServingSessionError, match="must declare --max-model-len"):
        declared_server_geometry(["--dtype", "bfloat16"], transport="vllm")
    with pytest.raises(ServingSessionError, match="must be positive"):
        declared_server_geometry(["--max-model-len", "0"], transport="vllm")
    with pytest.raises(ServingSessionError, match="without an integer value"):
        declared_server_geometry(["--max-model-len", "big"], transport="vllm")


def test_every_other_transport_keeps_the_llama_dialect():
    for transport in ("fake", "mock", "llama-server", None):
        assert merge_server_args([], slots=2, context_tokens=16384, transport=transport) == ["-np", "2", "-c", "16384"]


def test_the_flight_wrapper_passes_the_dialect_and_keeps_its_error_type():
    assert flight_merge([], slots=4, context_tokens=4 * 24576, transport="vllm") == ["--max-model-len", "24576"]
    with pytest.raises(FlightRuntimeError):
        flight_merge(["-c", "1"], slots=4, context_tokens=4 * 24576, transport="vllm")


def test_the_serving_profile_reads_a_vllm_fragment_in_its_own_dialect(tmp_path):
    _runtime_root, _data_root, runtime_receipt = _runtime_world(tmp_path)
    fragment = _fragment_document()
    fragment["transport"] = "vllm"
    fragment["server_args"] = list(_VLLM_ARGS)
    fragment["serving_geometry"] = {"slots": 4, "slot_tokens": 24576, "max_tokens": 4096}
    repo_root = _repo_with_fragment(tmp_path, runtime_receipt, fragment)
    profile = resolve_serving_profile(repo_root, "frag")
    assert profile["serving_geometry"] == {"slots": 4, "slot_tokens": 24576, "max_tokens": 4096}
    assert profile["extra_argv"] == _VLLM_ARGS

    # A fragment that declares no serving_geometry has no slot count at all: sequences per instance is deployment.
    undeclared = dict(fragment)
    del undeclared["serving_geometry"]
    profile = resolve_serving_profile(_repo_with_fragment(tmp_path / "bare", runtime_receipt, undeclared), "frag")
    assert profile["serving_geometry"]["slots"] is None and profile["serving_geometry"]["slot_tokens"] == 24576
    assert profile["extra_argv"] == _VLLM_ARGS

    fragment["serving_geometry"] = {"slots": 4, "slot_tokens": 8192, "max_tokens": 4096}
    repo_root = _repo_with_fragment(tmp_path / "drift", runtime_receipt, fragment)
    with pytest.raises(ServingSessionError, match="disagree with its declared serving_geometry"):
        resolve_serving_profile(repo_root, "frag")


def test_the_runtime_view_checks_a_vllm_fragment_in_its_own_dialect():
    from speech_aware_evidence_acquisition.rails.adapters.runtime_view import RuntimeViewError, configured_geometry

    values = {"transport": "vllm", "server_args": list(_VLLM_ARGS), "serving_slots": 4, "serving_slot_tokens": 24576}
    assert configured_geometry(values, defaults=(8, 8192, 64)) == {"slots": 4, "slot_tokens": 24576, "max_tokens": 64}
    values["serving_slot_tokens"] = 6144
    with pytest.raises(RuntimeViewError, match="--max-model-len differs"):
        configured_geometry(values, defaults=(8, 8192, 64))
    values = {"transport": "vllm", "server_args": ["-np", "4", "-c", "98304"], "serving_slots": 4,
              "serving_slot_tokens": 24576}
    with pytest.raises(RuntimeViewError, match="--max-model-len exactly once"):
        configured_geometry(values, defaults=(8, 8192, 64))
    values = {"transport": "vllm", "server_args": [*_VLLM_ARGS, "--max-num-seqs", "4"], "serving_slots": 4,
              "serving_slot_tokens": 24576}
    with pytest.raises(RuntimeViewError, match="deployment fact"):
        configured_geometry(values, defaults=(8, 8192, 64))
    # The llama dialect still checks both flags, unchanged.
    llama = {"server_args": ["-np", "4", "-c", "98304"], "serving_slots": 4, "serving_slot_tokens": 24576}
    assert configured_geometry(llama, defaults=(8, 8192, 64))["slots"] == 4
    llama["server_args"] = ["-c", "98304"]
    with pytest.raises(RuntimeViewError, match="-np exactly once"):
        configured_geometry(llama, defaults=(8, 8192, 64))


@pytest.mark.parametrize("tokens", [["--mm-processor-cache-type", "shm"], ["--mm-processor-cache-type=lru"],
                                    ["--mm-processor-cache-gb", "8"], ["--disable-mm-preprocessor-cache"]])
def test_a_vllm_fragment_may_not_carry_a_multimodal_cache_flag(tokens):
    flag = tokens[0].split("=", 1)[0]
    with pytest.raises(ServingSessionError, match=flag):
        merge_server_args([*_VLLM_ARGS, *tokens], slots=4, context_tokens=4 * 24576, transport="vllm")
    with pytest.raises(ServingSessionError, match=flag):
        declared_server_geometry([*_VLLM_ARGS, *tokens], transport="vllm")
    from speech_aware_evidence_acquisition.rails.adapters.runtime_view import RuntimeViewError, configured_geometry

    values = {"transport": "vllm", "server_args": [*_VLLM_ARGS, *tokens], "serving_slots": 4, "serving_slot_tokens": 24576}
    with pytest.raises(RuntimeViewError, match=flag):
        configured_geometry(values, defaults=(8, 8192, 64))
