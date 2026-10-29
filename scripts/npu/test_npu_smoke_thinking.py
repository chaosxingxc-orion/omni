"""Offline tests for the template check of npu_smoke_thinking.py (no server, no model).

Run inside saea-base: bash port_py.sh -m pytest -q -p no:cacheprovider <this file>
The raw tails are the exact strings from the 2026-10-29 smoke on vLLM 0.30.0 (smoke-181/out/template.json).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import npu_smoke_thinking as smoke  # noqa: E402

# U+0120 is the byte-level surface form of a space, U+010A of a newline.
RAW_TRUE = "ameĠtheĠintent.<|im_end|>Ċ<|im_start|>assistantĊ"
RAW_FALSE = "_end|>Ċ<|im_start|>assistantĊ<think>ĊĊ</think>ĊĊ"


def test_raw_tail_cannot_match_the_literal_comparison():
    # The defect: comparing the raw surface form against the literal text can never succeed.
    assert not RAW_FALSE.endswith("<think>\n\n</think>\n\n")


def test_decode_recovers_text_and_keeps_special_tokens():
    assert smoke.decode_byte_level(RAW_TRUE) == "ame the intent.<|im_end|>\n<|im_start|>assistant\n"
    assert smoke.decode_byte_level(RAW_FALSE) == "_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def test_decode_passes_unmapped_characters_through_and_handles_multibyte():
    assert smoke.decode_byte_level("<|audio_start|>") == "<|audio_start|>"
    assert smoke.decode_byte_level("\u2603") == "\u2603"  # not in the map: stands for itself
    assert smoke.decode_byte_level("Ã©") == "é"  # the two byte-level chars of UTF-8 e-acute


def test_decoder_is_the_inverse_of_a_complete_256_entry_map():
    assert sorted(smoke._BYTE_LEVEL_DECODER.values()) == list(range(256))


def test_template_tails_ok_on_the_measured_server_tails():
    decoded = {True: smoke.decode_byte_level(RAW_TRUE), False: smoke.decode_byte_level(RAW_FALSE)}
    assert smoke.template_tails_ok(decoded)


def test_template_tails_ok_rejects_wrong_behaviour():
    good_false = smoke.decode_byte_level(RAW_FALSE)
    assert not smoke.template_tails_ok({True: "x<|im_start|>assistant\n<think>\n", False: good_false})
    assert not smoke.template_tails_ok({True: smoke.decode_byte_level(RAW_TRUE), False: "<|im_start|>assistant\n"})
    # The raw (undecoded) tails never pass: this is the pre-fix behaviour.
    assert not smoke.template_tails_ok({True: RAW_TRUE, False: RAW_FALSE})
