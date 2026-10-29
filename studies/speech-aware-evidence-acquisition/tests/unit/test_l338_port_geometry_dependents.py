"""The committed L338 NPU-port cells after the ``--max-num-seqs`` / ``profile_slot_count`` removal (stream S13).

Commit db798a83 moved sequences-per-instance out of the hashed vLLM fragment (owner decision 2026-10-28). These tests pin
the dependents on the REAL committed port configs: the two baselines (``serving_slots`` 4, ``serving_slot_tokens`` 24576,
owner ruling) still pass the runtime geometry check against the fragment's ``--max-model-len`` window alone; the OLD
fragment form (``--max-num-seqs`` inside ``server_args``, ``profile_slot_count``) is REFUSED, never silently dropped; the
llama-profile paths of ``core.driver.speech_native`` (``-np`` / ``-c``) fail closed on a vLLM fragment; the writer's
dependents list reads "moved".
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _extra in (REPO / "scripts" / "flights" / "nx", REPO / "scripts" / "flights" / "direct", REPO / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import l338_port_family as pf  # noqa: E402

from speech_aware_evidence_acquisition.core.config import compose  # noqa: E402
from speech_aware_evidence_acquisition.core.driver import speech_native  # noqa: E402
from speech_aware_evidence_acquisition.core.session.serving import (  # noqa: E402
    ServingSessionError,
    declared_server_geometry,
    merge_server_args,
)
from speech_aware_evidence_acquisition.rails.adapters.runtime_view import (  # noqa: E402
    RuntimeViewError,
    configured_geometry,
)

DEFAULTS = (8, 8192, 64)


def _composed_values(key: str) -> dict:
    spec = pf.CELLS[key]
    cell = json.loads((REPO / "configs" / "experiment" / f"{spec['cell']}.json").read_text(encoding="utf-8"))["cells"][0]
    return dict(compose(REPO, cell["model_fragment"], cell["dataset_fragment"], cell["baseline_fragment"], spec["cell"]).values)


@pytest.mark.parametrize("key", list(pf.ALL_KEYS))
def test_the_committed_port_baselines_pass_the_runtime_geometry_check(key):
    values = _composed_values(key)
    assert values["transport"] == "vllm"
    assert values["serving_slots"] == 4 and values["serving_slot_tokens"] == 24576
    assert "--max-num-seqs" not in values["server_args"] and "profile_slot_count" not in values
    geometry = configured_geometry(values, defaults=DEFAULTS)
    assert geometry["slots"] == 4 and geometry["slot_tokens"] == 24576 and geometry["max_tokens"] == 4096


@pytest.mark.parametrize("key", list(pf.ALL_KEYS))
@pytest.mark.parametrize("spelling", [["--max-num-seqs", "4"], ["--max-num-seqs=4"]])
def test_the_old_fragment_form_is_refused_not_silently_dropped(key, spelling):
    values = _composed_values(key)
    old = copy.deepcopy(values)
    old["server_args"] = [*values["server_args"], *spelling]
    old["profile_slot_count"] = 4
    args = [str(token) for token in old["server_args"]]
    with pytest.raises((RuntimeViewError, ServingSessionError)):
        configured_geometry(old, defaults=DEFAULTS)
    with pytest.raises(ServingSessionError):
        declared_server_geometry(args, transport="vllm", model_fragment=pf.MODEL_FRAGMENT)
    with pytest.raises(ServingSessionError):
        merge_server_args(args, slots=4, context_tokens=4 * 24576, transport="vllm")


@pytest.mark.parametrize("key", list(pf.ALL_KEYS))
def test_the_llama_profile_paths_of_speech_native_fail_closed_on_the_vllm_fragment(key):
    """``_profile_geometry`` reads ``-np`` / ``-c`` and ``profile_slot_count``; a vLLM fragment carries none of them."""

    values = _composed_values(key)
    with pytest.raises(speech_native.SpeechNativeProfileError, match="profile_slot_count"):
        speech_native._require_positive_int(values, "profile_slot_count")
    with pytest.raises(speech_native.SpeechNativeProfileError, match="-np and -c"):
        speech_native._profile_geometry(values, 4)


def test_the_dependents_list_reads_moved():
    assert pf.MAX_NUM_SEQS_DEPENDENTS
    for entry in pf.MAX_NUM_SEQS_DEPENDENTS:
        assert entry.startswith(("moved:", "unchanged by owner ruling:", "open:")), entry
    assert not any(entry.startswith("moved:") and "pre-sweep" in entry for entry in pf.MAX_NUM_SEQS_DEPENDENTS)
    assert "profile_slot_count" in pf.PORT_DEPLOYMENT_PROFILE["not_a_fact_here"]
    assert "current server_args" not in pf.PORT_DEPLOYMENT_PROFILE["not_a_fact_here"]
