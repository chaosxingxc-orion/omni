"""NPU smoke for the port's Thinking server (design doc section 10, item 2). Read-only against one served model.

usage (inside saea-base, PYTHONPATH=<study>/src, server from npu_serve.sh thinking):
    /opt/saea-venv/bin/python npu_smoke_thinking.py --base-url http://127.0.0.1:8200 --audio <a 16 kHz wav> --out <dir>

Each check writes its raw request and response to <out>/<check>.json and prints one PASS/FAIL/RECORD line. Nothing is
retried. The checks:

1. forced-close  thinking_token_budget 16 on a question that needs long reasoning: the reasoning must end with the
                 fragment's message; the exact bytes around it are RECORDED to settle the section-4 assumption.
2. overflow      a prompt longer than max_model_len: the 400 must be relabelled by the study's own
                 vllm_context_error_body (section 3); a reworded message FAILS here, before any flight.
3. tool-call     one hermes tool call through --tool-call-parser hermes. PASS if one call arrives; otherwise RECORD (not
                 FAIL) for this build: the 2026-10-29 smoke on vLLM 0.30.0 got HTTP 400 "default chat template is no
                 longer allowed" on this request (no system message, so not the system-parts failure); evidence
                 tool-call.json and system-tools.json; Fold B declares no tools (design doc section 5.5).
4. audio-count   /tokenize (chat form, as measure_vllm_multimodal_request sends it) must equal the engine's
                 usage.prompt_tokens for the same audio request (section 5.3).
5. template      the rendered prompt of the Fold B wire shape (system string, user [audio, text]) in both thinking
                 modes: exactly one audio span; enable_thinking true must NOT pre-insert <think> (so the budget
                 counts from the model's own <think>, section 4); false must end in the empty think block. The
                 prompt tails are RECORDED, raw and decoded: /tokenize token strings are GPT-2 byte-level BPE surface
                 forms (U+0120 = space, U+010A = newline), so they are decoded before any text comparison.
6. answer-only   the decision-278 re-ask (enable_thinking false, max_tokens = the 1024 reserve, no budget): the
                 answer must arrive in content, with no reasoning and no think marker in content (the
                 refuse-think-marker-in-content guard).
7. system-tools  RECORD only (Fold B declares no tools): a system string beside tools. Offline prediction: vLLM wraps
                 the string into text parts and the template's messages[0].content + '\n\n' fails on parts. Measured
                 2026-10-29: HTTP 400 "default chat template is no longer allowed" before that point is reached, the
                 same 400 as check 3. A tool-bearing arm needs a decision before it runs on vLLM (design doc
                 section 5.5).
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from speech_aware_evidence_acquisition.core.model.vllm_transport import vllm_context_error_body
from speech_aware_evidence_acquisition.core.preparation import _VLLM_RENDER_KEYS

MODEL = "qwen3-omni-30b-a3b-thinking"
MESSAGE = "Considering the limited time by the user, I have to give the solution based on the thinking directly now."


def _byte_level_decoder() -> dict[str, int]:
    """Inverse of the GPT-2 bytes_to_unicode map (the byte-level BPE surface alphabet), implemented locally."""
    kept = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    chars, extra = list(kept), 0
    for byte in range(256):
        if byte not in kept:
            kept.append(byte)
            chars.append(256 + extra)
            extra += 1
    return {chr(char): byte for byte, char in zip(kept, chars)}


_BYTE_LEVEL_DECODER = _byte_level_decoder()


def decode_byte_level(text: str) -> str:
    """Map byte-level BPE surface forms back to text. Characters outside the map stand for themselves (UTF-8)."""
    raw = bytearray()
    for char in text:
        if char in _BYTE_LEVEL_DECODER:
            raw.append(_BYTE_LEVEL_DECODER[char])
        else:
            raw.extend(char.encode("utf-8"))
    return raw.decode("utf-8", "replace")


def template_tails_ok(tails: dict[bool, str]) -> bool:
    """Decoded prompt tails: thinking true must not pre-insert <think>; false must end in the empty think block."""
    return "<think>" not in tails[True] and tails[False].endswith("<think>\n\n</think>\n\n")


def post(base_url: str, path: str, body: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        base_url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:  # noqa: S310 (loopback)
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            return error.code, json.loads(raw)
        except ValueError:
            return error.code, {"raw": raw.decode("utf-8", "replace")}


def record(out: Path, name: str, request: dict, status: int, response: dict, verdict: str, detail: str) -> bool:
    (out / f"{name}.json").write_text(json.dumps(
        {"request": request, "status": status, "response": response, "verdict": verdict, "detail": detail},
        ensure_ascii=False, indent=1,
    ))
    print(f"{verdict:6} {name}: {detail}")
    return verdict != "FAIL"


def forced_close(base_url: str, out: Path) -> bool:
    body = {
        "model": MODEL, "temperature": 0, "seed": 20260913, "max_tokens": 256, "thinking_token_budget": 16,
        "chat_template_kwargs": {"enable_thinking": True},
        "messages": [{"role": "user", "content": "Prove that there are infinitely many primes, step by step."}],
    }
    status, response = post(base_url, "/v1/chat/completions", body)
    message = (response.get("choices") or [{}])[0].get("message") or {}
    reasoning = message.get("reasoning") or ""
    ok = status == 200 and reasoning.rstrip().endswith(MESSAGE)
    tail = reasoning[-(len(MESSAGE) + 16):]
    return record(out, "forced-close", body, status, response, "PASS" if ok else "FAIL",
                  f"reasoning tail {tail!r}; content head {str(message.get('content'))[:40]!r}")


def overflow(base_url: str, out: Path) -> bool:
    body = {
        "model": MODEL, "max_tokens": 16, "chat_template_kwargs": {"enable_thinking": True},
        "messages": [{"role": "user", "content": "word " * 30000}],
    }
    status, response = post(base_url, "/v1/chat/completions", body)
    relabelled = vllm_context_error_body(status, json.dumps(response).encode())
    return record(out, "overflow", body, status, response, "PASS" if relabelled else "FAIL",
                  f"status {status}; message {str((response.get('error') or {}).get('message'))[:160]!r}")


def tool_call(base_url: str, out: Path) -> bool:
    body = {
        "model": MODEL, "temperature": 0, "seed": 20260913, "max_tokens": 2048, "thinking_token_budget": 1024,
        "chat_template_kwargs": {"enable_thinking": True}, "tool_choice": "auto",
        "tools": [{"type": "function", "function": {
            "name": "lookup_intent", "description": "Look up the meaning of a spoken-language intent label.",
            "parameters": {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]},
        }}],
        "messages": [{"role": "user", "content": "Use the tool to look up the intent label 'alarm_set'."}],
    }
    status, response = post(base_url, "/v1/chat/completions", body)
    message = (response.get("choices") or [{}])[0].get("message") or {}
    calls = message.get("tool_calls") or []
    ok = status == 200 and len(calls) == 1 and calls[0]["function"]["name"] == "lookup_intent"
    detail = f"{len(calls)} call(s): {[c.get('function') for c in calls]}"
    if not ok:
        error = response.get("error")
        reason = str(error.get("message") if isinstance(error, dict) else response)[:160]
        detail = (f"RECORD for this build, not a gate: status {status}; {reason!r}; {detail}; evidence tool-call.json "
                  "and system-tools.json (2026-10-29 smoke, vLLM 0.30.0); Fold B declares no tools, design doc section 5.5")
    return record(out, "tool-call", body, status, response, "PASS" if ok else "RECORD", detail)


def audio_count(base_url: str, out: Path, audio: Path) -> bool:
    audio_part = {"type": "input_audio",
                  "input_audio": {"data": base64.b64encode(audio.read_bytes()).decode(), "format": "wav"}}
    body = {
        "model": MODEL, "temperature": 0, "seed": 20260913, "max_tokens": 1, "thinking_token_budget": 1,
        "chat_template_kwargs": {"enable_thinking": True},
        "messages": [{"role": "system", "content": "Answer briefly."},
                     {"role": "user", "content": [audio_part, {"type": "text", "text": "What was said?"}]}],
    }
    tokenize = {key: body[key] for key in _VLLM_RENDER_KEYS if key in body}
    tokenize.update({"model": MODEL, "add_generation_prompt": True, "add_special_tokens": False})
    t_status, counted = post(base_url, "/tokenize", tokenize)
    c_status, completion = post(base_url, "/v1/chat/completions", body)
    measured = counted.get("count")
    engine = (completion.get("usage") or {}).get("prompt_tokens")
    ok = t_status == 200 and c_status == 200 and isinstance(measured, int) and measured == engine
    for item in (body, tokenize):
        item["messages"][1]["content"][0]["input_audio"]["data"] = f"<{audio.name}, base64 elided>"
    counted.pop("tokens", None)
    return record(out, "audio-count", {"chat": body, "tokenize": tokenize}, c_status,
                  {"tokenize": counted, "chat": completion}, "PASS" if ok else "FAIL",
                  f"/tokenize count {measured} vs engine prompt_tokens {engine}")


def _audio_messages(audio: Path) -> list[dict]:
    audio_part = {"type": "input_audio",
                  "input_audio": {"data": base64.b64encode(audio.read_bytes()).decode(), "format": "wav"}}
    return [{"role": "system", "content": "You label spoken requests."},
            {"role": "user", "content": [audio_part, {"type": "text", "text": "Name the intent."}]}]


def _elide(messages: list[dict], audio: Path) -> list[dict]:
    out = json.loads(json.dumps(messages))
    out[1]["content"][0]["input_audio"]["data"] = f"<{audio.name}, base64 elided>"
    return out


def template(base_url: str, out: Path, audio: Path) -> bool:
    messages, tails, raw_tails, spans, statuses = _audio_messages(audio), {}, {}, {}, {}
    for thinking in (True, False):
        body = {"model": MODEL, "messages": messages, "chat_template_kwargs": {"enable_thinking": thinking},
                "add_generation_prompt": True, "add_special_tokens": False, "return_token_strs": True}
        status, response = post(base_url, "/tokenize", body)
        joined = "".join(response.get("token_strs") or [])
        text = decode_byte_level(joined)
        statuses[thinking], tails[thinking], raw_tails[thinking] = status, text[-48:], joined[-48:]
        spans[thinking] = (text.count("<|audio_start|>"), text.count("<|audio_end|>"))
    ok = (all(code == 200 for code in statuses.values()) and all(span == (1, 1) for span in spans.values())
          and template_tails_ok(tails))
    request = {"messages": _elide(messages, audio), "modes": [True, False]}
    return record(out, "template", request, max(statuses.values()),
                  {"tails": tails, "raw_tails": raw_tails, "audio_spans": spans},
                  "PASS" if ok else "FAIL", f"tails {tails!r}; raw tails {raw_tails!r}; audio spans {spans}")


def answer_only(base_url: str, out: Path, audio: Path) -> bool:
    body = {
        "model": MODEL, "temperature": 0, "seed": 20260913, "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": False}, "messages": _audio_messages(audio),
    }
    status, response = post(base_url, "/v1/chat/completions", body)
    message = (response.get("choices") or [{}])[0].get("message") or {}
    content, reasoning = message.get("content") or "", message.get("reasoning") or ""
    ok = status == 200 and content.strip() != "" and not reasoning and "<think>" not in content \
        and "</think>" not in content
    body["messages"] = _elide(body["messages"], audio)
    return record(out, "answer-only", body, status, response, "PASS" if ok else "FAIL",
                  f"content head {content[:60]!r}; reasoning {len(reasoning)} chars")


def system_tools(base_url: str, out: Path) -> bool:
    body = {
        "model": MODEL, "chat_template_kwargs": {"enable_thinking": True},
        "add_generation_prompt": True, "add_special_tokens": False,
        "tools": [{"type": "function", "function": {"name": "lookup_intent", "parameters": {"type": "object"}}}],
        "messages": [{"role": "system", "content": "You label spoken requests."},
                     {"role": "user", "content": "Name the intent of 'wake me at seven'."}],
    }
    status, response = post(base_url, "/tokenize", body)
    response.pop("tokens", None)
    message = str((response.get("error") or {}).get("message") if isinstance(response.get("error"), dict) else "")
    return record(out, "system-tools", body, status, response, "RECORD",
                  f"status {status}; {'renders' if status == 200 else message[:160]!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    base_url = args.base_url.rstrip("/")
    results = [forced_close(base_url, args.out), overflow(base_url, args.out), tool_call(base_url, args.out),
               audio_count(base_url, args.out, args.audio), template(base_url, args.out, args.audio),
               answer_only(base_url, args.out, args.audio), system_tools(base_url, args.out)]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
