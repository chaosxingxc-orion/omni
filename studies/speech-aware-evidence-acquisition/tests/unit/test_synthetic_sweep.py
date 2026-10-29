"""The synthetic concurrency sweep (``scripts/serving/synthetic_sweep.py``) against tiny fake vLLM-shaped servers.

The fake server speaks just enough of vLLM 0.30's surface: ``/v1/models`` (with ``max_model_len``), chat-form
``/tokenize`` (a deterministic, deliberately non-linear count: words longer than five letters cost two tokens, audio
costs 12.5 tokens per second of WAV), ``/v1/chat/completions`` (streaming SSE with ``reasoning`` deltas and a usage
chunk, or one JSON body) and ``/metrics`` (Prometheus text with a per-engine-labelled ``vllm:num_preemptions_total``
that grows whenever more than ``preempt_above`` requests are in flight). No model is involved.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import random
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "serving" / "synthetic_sweep.py"
_spec = importlib.util.spec_from_file_location("synthetic_sweep", SCRIPT)
sweep = importlib.util.module_from_spec(_spec)
sys.modules["synthetic_sweep"] = sweep
_spec.loader.exec_module(sweep)

MODEL = "fake-omni"


def _fake_tokens(messages) -> int:
    count = 10
    for message in messages:
        content = message["content"]
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
        for part in parts:
            if part["type"] == "text":
                count += sum(2 if len(word) > 5 else 1 for word in part["text"].split())
            elif part["type"] == "input_audio":
                with wave.open(io.BytesIO(base64.b64decode(part["input_audio"]["data"]))) as handle:
                    count += int(handle.getnframes() / handle.getframerate() * 12.5)
    return count


class FakeServer:
    def __init__(self, *, delay=0.02, completion_tokens=8, preempt_above=None, window=24576, model=MODEL):
        self.delay, self.completion_tokens, self.preempt_above = delay, completion_tokens, preempt_above
        self.window, self.model = window, model
        self.lock = threading.Lock()
        self.in_flight = self.preemptions = self.max_in_flight = 0
        self.chat_requests: list[tuple[str, dict]] = []
        self.tokenize_calls = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _json(self, status, document):
                data = json.dumps(document).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/v1/models":
                    self._json(200, {"data": [{"id": server.model, "max_model_len": server.window}]})
                elif self.path == "/health":
                    self._json(200, {})
                elif self.path == "/metrics":
                    with server.lock:
                        text = (
                            "# TYPE vllm:num_preemptions_total counter\n"
                            f'vllm:num_preemptions_total{{engine="0",model_name="{server.model}"}} {float(server.preemptions)}\n'
                            f'vllm:num_preemptions_created{{engine="0"}} 1.7e9\n'
                            f'vllm:prefix_cache_queries_total{{engine="0"}} 100.0\n'
                            f'vllm:prefix_cache_hits_total{{engine="0"}} 40.0\n'
                            f'vllm:kv_cache_usage_perc{{engine="0"}} {0.1 * server.in_flight}\n'
                            f'vllm:num_requests_running{{engine="0"}} {float(server.in_flight)}\n'
                            f'vllm:num_requests_waiting{{engine="0"}} 0.0\n'
                        )
                    data = text.encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self._json(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/tokenize":
                    with server.lock:
                        server.tokenize_calls += 1
                    assert body["add_generation_prompt"] is True and body["add_special_tokens"] is False
                    count = _fake_tokens(body["messages"])
                    self._json(200, {"count": count, "tokens": list(range(count)), "max_model_len": server.window})
                    return
                assert self.path == "/v1/chat/completions"
                if body.get("model") != server.model:
                    self._json(404, {"error": {"message": "model not found"}})
                    return
                with server.lock:
                    server.chat_requests.append((self.headers.get("X-Request-Id"), body))
                    server.in_flight += 1
                    server.max_in_flight = max(server.max_in_flight, server.in_flight)
                    if server.preempt_above is not None and server.in_flight > server.preempt_above:
                        server.preemptions += 1
                try:
                    prompt = _fake_tokens(body["messages"])
                    completion = min(body["max_tokens"], server.completion_tokens)
                    usage = {"prompt_tokens": prompt, "completion_tokens": completion,
                             "total_tokens": prompt + completion, "prompt_tokens_details": None}
                    if not body.get("stream"):
                        time.sleep(server.delay)
                        self._json(200, {"choices": [{"message": {"content": "{}", "reasoning": "think"},
                                                      "finish_reason": "stop"}], "usage": usage})
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    events = [{"choices": [{"delta": {"role": "assistant"}}]},
                              {"choices": [{"delta": {"reasoning": "think"}}]},
                              "sleep",
                              {"choices": [{"delta": {"content": "{}"}, "finish_reason": "stop"}]},
                              {"choices": [], "usage": usage}]
                    for event in events:
                        if event == "sleep":
                            time.sleep(server.delay)
                            continue
                        self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                        self.wfile.flush()
                    self.wfile.write(b"data: [DONE]\n\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    with server.lock:
                        server.in_flight -= 1

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def servers(request):
    made = []

    def make(count=2, **kwargs):
        made.extend(FakeServer(**kwargs) for _ in range(count))
        return made[-count:]

    yield make
    for server in made:
        server.close()


def _config(urls, **overrides):
    values = dict(
        base_urls=[server.url for server in urls] if urls and not isinstance(urls[0], str) else urls,
        model=MODEL, levels=[1, 2], seed="t", contacts_per_trajectory=2, prompt_mean=300, prompt_min=150,
        prompt_max=600, prompt_sigma=0.4, audio_min_seconds=0.2, audio_max_seconds=0.5, max_tokens=64,
        thinking_budget=32, metrics_interval_s=0.05, warmup=1, fit_workers=4, idle_wait_s=2.0, timeout_s=10.0,
    )
    values.update(overrides)
    return sweep.SweepConfig(**values)


# --- synthetic content ------------------------------------------------------------------------------------------------


def test_the_synthetic_wav_is_16khz_mono_pcm_of_the_requested_length_and_seeded():
    data = sweep.synthetic_wav("a", 2.5)
    with wave.open(io.BytesIO(data)) as handle:
        assert (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) == (16000, 1, 2)
        assert handle.getnframes() == 40000
        frames = handle.readframes(handle.getnframes())
    assert any(frames) and data == sweep.synthetic_wav("a", 2.5) and data != sweep.synthetic_wav("b", 2.5)


def test_prompt_targets_have_the_requested_truncated_mean_and_range():
    mu = sweep.lognormal_mu_for_mean(8500, 0.5, 3000, 20000)
    targets = sweep.sample_prompt_targets(random.Random(1), 20000, mu=mu, sigma=0.5, low=3000, high=20000)
    assert min(targets) >= 3000 and max(targets) <= 20000
    assert abs(sum(targets) / len(targets) - 8500) < 8500 * 0.02
    with pytest.raises(sweep.SweepError):
        sweep.lognormal_mu_for_mean(25000, 0.5, 3000, 20000)


def test_contact_messages_follow_the_raw_query_prefix_layout_and_share_the_evidence_prefix():
    words = list(sweep.WORDS[:50])
    first = sweep.contact_messages("QUJD", words, 20, 0, 2, 0)
    second = sweep.contact_messages("QUJD", words, 40, 1, 2, 3)
    assert first[0] == {"role": "system", "content": sweep.SYSTEM_TEXT}
    assert [part["type"] for part in first[1]["content"]] == ["input_audio", "text", "text"]
    assert first[1]["content"][0]["input_audio"]["format"] == "wav"
    assert second[1]["content"][1]["text"].startswith(first[1]["content"][1]["text"])
    assert second[1]["content"][2]["text"].endswith(" x x x")


@pytest.mark.parametrize("target", [40, 41, 97, 250, 1001, 4321])
def test_the_fit_hits_every_target_exactly_with_a_nonlinear_tokenizer(target):
    words = [random.Random(target).choice(sweep.WORDS) for _ in range(5000)]

    def count(n_words, pad):
        return 30 + sum(2 if len(word) > 5 else 1 for word in words[:n_words]) + pad

    fitted = sweep.fit_contact(count, target)
    assert fitted.exact and fitted.tokenized == target and fitted.tokenize_calls <= 32
    with pytest.raises(sweep.SweepError, match="below the fixed overhead"):
        sweep.fit_contact(count, 29)


# --- end to end against fake servers -----------------------------------------------------------------------------------


def test_a_sweep_pins_workers_to_instances_and_finds_the_knee_below_preemption(servers):
    fakes = servers(2, preempt_above=2)
    report = sweep.run_sweep(_config(fakes, levels=[1, 2, 3]))
    assert report["schema"] == sweep.SCHEMA and report["synthetic"] is True and report["dataset_rows"] == 0
    assert [level["concurrency_per_instance"] for level in report["levels"]] == [1, 2, 3]
    for index, fake in enumerate(fakes):
        measured = [(rid, body) for rid, body in fake.chat_requests if not rid.startswith("sweep-warmup")]
        workers = {int(rid.split("-w")[1].split("-")[0]) for rid, _ in measured}
        assert workers and all(worker % 2 == index for worker in workers)
        for _, body in measured:
            assert body["thinking_token_budget"] == 32 and body["chat_template_kwargs"] == {"enable_thinking": True}
            assert body["temperature"] == 0 and body["max_tokens"] == 64 and body["stream"] is True
            assert "id_slot" not in body and "tools" not in body
            assert [part["type"] for part in body["messages"][1]["content"]] == ["input_audio", "text", "text"]
            words = body["messages"][1]["content"][1]["text"].split()[1:]
            assert set(words) <= set(sweep.WORDS)
    for level in report["levels"]:
        assert level["requests"] == level["ok"] == level["concurrency_per_instance"] * 2 * 2
        assert level["prompt_tokens_exact"] and level["prompt_tokens_server_matches_tokenize"]
        assert level["e2e_tok_s"]["p10"] > 10 and level["ttft_s"]["n"] == level["ok"]
        assert level["gauges"][0]["readable"] >= 1
    assert [level["preemptions_delta"] for level in report["levels"]][:2] == [0.0, 0.0]
    assert report["levels"][2]["preemptions_delta"] > 0
    knee = report["knee"]
    assert knee["recommended_concurrency_per_instance"] == 2
    assert any("preemption" in reason for reason in knee["failing_reasons"]["3"])
    json.dumps(report, sort_keys=True)


def test_requests_past_the_timeout_are_timeouts_and_no_knee_is_recommended(servers):
    fakes = servers(1, delay=1.0)
    report = sweep.run_sweep(_config(fakes, levels=[1], timeout_s=0.4, warmup=0))
    level = report["levels"][0]
    assert level["timeouts"] == level["requests"] == 2 and level["ok"] == 0
    assert report["knee"]["recommended_concurrency_per_instance"] is None
    assert any("timeout" in reason for reason in report["knee"]["failing_reasons"]["1"])


def test_the_cli_writes_a_sorted_json_report_without_streaming(servers, tmp_path):
    fakes = servers(2)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"profile_id": "p"}), encoding="utf-8")
    out = tmp_path / "report.json"
    assert sweep.main([
        "--base-url", ",".join(fake.url for fake in fakes), "--model", MODEL, "--levels", "1", "--seed", "s",
        "--contacts-per-trajectory", "1", "--prompt-mean", "300", "--prompt-min", "150", "--prompt-max", "600",
        "--audio-min-seconds", "0.2", "--audio-max-seconds", "0.3", "--max-tokens", "64", "--thinking-budget", "16",
        "--no-stream", "--metrics-interval", "0.05", "--requests-per-level", "4", "--deployment-profile", str(profile),
        "--label", "unit", "--out", str(out),
    ]) == 0
    text = out.read_text(encoding="utf-8")
    report = json.loads(text)
    assert text == json.dumps(report, indent=2, sort_keys=True) + "\n"
    assert report["deployment_profile"]["document"] == {"profile_id": "p"} and report["label"] == "unit"
    level = report["levels"][0]
    assert level["requests"] == level["ok"] == 4 and level["ttft_s"]["n"] == 0
    assert {request["instance"] for request in report["requests"]} == {0, 1}
    assert all(request["worker"] % 2 == request["instance"] for request in report["requests"])


def test_preflight_refuses_a_wrong_model_a_small_window_and_too_few_requests(servers):
    (wrong,) = servers(1, model="other")
    with pytest.raises(sweep.SweepError, match="does not serve"):
        sweep.run_sweep(_config([wrong]))
    (small,) = servers(1, window=500)
    with pytest.raises(sweep.SweepError, match="window"):
        sweep.run_sweep(_config([small]))
    with pytest.raises(sweep.SweepError, match="cannot keep"):
        sweep.build_level(_config(["http://127.0.0.1:1", "http://127.0.0.1:2"], requests_per_level=3), 1)
    with pytest.raises(sweep.SweepError, match="strictly increasing"):
        _config(["http://127.0.0.1:1"], levels=[2, 1]).validate()


# --- metrics and the knee rule ----------------------------------------------------------------------------------------


def test_prometheus_text_is_summed_over_label_sets():
    text = (
        "# HELP vllm:num_preemptions Cumulative number of preemption from the engine.\n"
        "# TYPE vllm:num_preemptions_total counter\n"
        'vllm:num_preemptions_total{engine="0",model_name="m"} 2.0\n'
        'vllm:num_preemptions_total{engine="1",model_name="m"} 3.0\n'
        "vllm:kv_cache_usage_perc 0.5 1700000000\n"
        "garbage line\n"
    )
    totals = sweep.parse_prometheus(text)
    assert totals["vllm:num_preemptions_total"] == 5.0 and totals["vllm:kv_cache_usage_perc"] == 0.5


def _level(c, *, p10=20.0, errors=0, timeouts=0, preemptions=0.0, tok_s=100.0):
    return {"concurrency_per_instance": c, "errors": errors, "timeouts": timeouts, "preemptions_delta": preemptions,
            "e2e_tok_s": {"p10": p10}, "output_tok_s": tok_s}


def test_the_knee_is_the_highest_level_passing_every_gate():
    levels = [_level(8, tok_s=300), _level(16, tok_s=500), _level(24, p10=9.5, tok_s=600),
              _level(32, preemptions=None), _level(48, timeouts=1, errors=2)]
    knee = sweep.recommend_knee(levels, min_seq_tok_s=10.0)
    assert knee["recommended_concurrency_per_instance"] == 16
    assert knee["highest_throughput_passing_concurrency"] == 16
    assert knee["failing_reasons"]["8"] == [] and "p10 9.5" in knee["failing_reasons"]["24"][0]
    assert knee["failing_reasons"]["32"] == ["preemption counter unreadable"]
    assert knee["failing_reasons"]["48"] == ["2 error(s)", "1 timeout(s)"]
    assert sweep.recommend_knee([_level(8, p10=None)], min_seq_tok_s=10.0)["recommended_concurrency_per_instance"] is None


# --- min_tokens payload -----------------------------------------------------------------------------------------------


def test_min_tokens_is_sent_on_measured_requests_only_when_set(servers):
    (plain,), (floored,) = servers(1), servers(1)
    sweep.run_sweep(_config([plain], levels=[1], liveness_interval_s=0))
    sweep.run_sweep(_config([floored], levels=[1], min_tokens=48, liveness_interval_s=0))
    measured = lambda fake: [body for rid, body in fake.chat_requests if not rid.startswith("sweep-warmup")]
    warmups = lambda fake: [body for rid, body in fake.chat_requests if rid.startswith("sweep-warmup")]
    assert measured(plain) and all("min_tokens" not in body for body in measured(plain))
    assert measured(floored) and all(body["min_tokens"] == 48 and body["max_tokens"] == 64 for body in measured(floored))
    # the unmeasured warmup (max_tokens 16) never carries the floor, which could exceed its max_tokens
    assert warmups(floored) and all("min_tokens" not in body for body in warmups(floored))


def test_the_cli_passes_min_tokens_through(servers, tmp_path):
    (fake,) = servers(1)
    out = tmp_path / "report.json"
    assert sweep.main([
        "--base-url", fake.url, "--model", MODEL, "--levels", "1", "--seed", "s", "--contacts-per-trajectory", "1",
        "--prompt-mean", "300", "--prompt-min", "150", "--prompt-max", "600", "--audio-min-seconds", "0.2",
        "--audio-max-seconds", "0.3", "--max-tokens", "64", "--thinking-budget", "16", "--min-tokens", "40",
        "--metrics-interval", "0.05", "--liveness-interval", "0", "--out", str(out),
    ]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["settings"]["min_tokens"] == 40
    assert {body["min_tokens"] for rid, body in fake.chat_requests if not rid.startswith("sweep-warmup")} == {40}


# --- liveness poll ----------------------------------------------------------------------------------------------------


def test_probe_endpoint_reads_v1_models(servers):
    (fake,) = servers(1)
    assert sweep.probe_endpoint(fake.url, MODEL) is None
    assert "does not list" in sweep.probe_endpoint(fake.url, "another-model")
    fake.close()
    assert sweep.probe_endpoint(fake.url, MODEL, timeout=1.0)


def test_the_poller_needs_consecutive_failures_and_records_the_dead_endpoint():
    calls = []

    def flaky(url, model, timeout):  # one slow answer on the first attempt only: not a death
        calls.append(url)
        return "timed out" if len(calls) == 1 else None

    poller = sweep.LivenessPoller(["http://a"], MODEL, 30.0, retry_gap=0.0, probe=flaky)
    assert poller.poll_once() is None and not poller.dead_event.is_set() and poller.polls == 1

    dead = sweep.LivenessPoller(["http://a", "http://b"], MODEL, 30.0, retry_gap=0.0,
                                probe=lambda url, model, timeout: "refused" if url == "http://b" else None)
    record = dead.poll_once("poll")
    assert dead.dead_event.is_set() and dead.died is record
    assert record["detected_by"] == "poll"
    assert record["endpoints"] == [{"instance": 1, "base_url": "http://b", "error": "refused"}]


def test_the_poller_thread_polls_immediately_then_on_the_interval_and_stops_on_death():
    seen = []
    poller = sweep.LivenessPoller(["http://a"], MODEL, 0.05, retry_gap=0.0,
                                  probe=lambda url, model, timeout: seen.append(url) or ("down" if len(seen) >= 3 else None))
    poller.start()
    assert poller.dead_event.wait(5.0)
    poller.join(5.0)
    assert not poller.is_alive() and len(seen) == 4  # 2 healthy polls, then 2 failed attempts of the third


def test_a_healthy_sweep_reports_no_death_and_polls_the_endpoints(servers):
    fakes = servers(2)
    report = sweep.run_sweep(_config(fakes, levels=[1], liveness_interval_s=0.05))
    assert report["endpoint_died"] is None and report["levels"][0]["endpoint_died"] is None
    assert report["levels"][0]["liveness"]["polls"] >= 2 and report["levels"][0]["aborted"] is False
    assert report["knee"]["recommended_concurrency_per_instance"] == 1


def test_a_dead_endpoint_aborts_the_level_and_the_sweep_with_an_explicit_field(servers, monkeypatch):
    monkeypatch.setattr(sweep, "LIVENESS_RETRY_GAP_S", 0.05)
    fakes = servers(2, delay=3.0)
    killer = threading.Timer(0.8, fakes[1].close)  # listener gone, in-flight handlers keep running
    killer.start()
    started = time.monotonic()
    report = sweep.run_sweep(_config(fakes, levels=[1, 2], liveness_interval_s=0.1, liveness_timeout_s=1.0, warmup=0))
    killer.join()
    assert time.monotonic() - started < 10  # aborted, not awaited to the end of the level
    died = report["endpoint_died"]
    assert died["level"] == 1 and died["endpoints"][0]["instance"] == 1
    assert died["endpoints"][0]["base_url"] == fakes[1].url
    assert [level["concurrency_per_instance"] for level in report["levels"]] == [1]  # level 2 never started
    assert report["levels_requested"] == [1, 2]
    assert report["levels"][0]["aborted"] is True and report["levels"][0]["endpoint_died"]["endpoints"]
    knee = report["knee"]
    assert knee["recommended_concurrency_per_instance"] is None
    assert "endpoint died" in knee["failing_reasons"]["1"]
    json.dumps(report, sort_keys=True)


def test_the_cli_exits_non_zero_after_writing_the_report_when_an_endpoint_died(servers, tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "LIVENESS_RETRY_GAP_S", 0.05)
    fakes = servers(2, delay=3.0)
    threading.Timer(0.8, fakes[0].close).start()
    out = tmp_path / "report.json"
    status = sweep.main([
        "--base-url", ",".join(fake.url for fake in fakes), "--model", MODEL, "--levels", "1", "--seed", "s",
        "--contacts-per-trajectory", "1", "--prompt-mean", "300", "--prompt-min", "150", "--prompt-max", "600",
        "--audio-min-seconds", "0.2", "--audio-max-seconds", "0.3", "--max-tokens", "64", "--thinking-budget", "16",
        "--metrics-interval", "0.05", "--liveness-interval", "0.1", "--liveness-timeout", "1", "--warmup", "0",
        "--out", str(out),
    ])
    assert status == sweep.EXIT_ENDPOINT_DIED == 3
    assert json.loads(out.read_text(encoding="utf-8"))["endpoint_died"]["endpoints"][0]["instance"] == 0


def test_a_level_with_endpoint_died_is_never_a_knee_even_if_every_other_gate_passes():
    dead = _level(16) | {"endpoint_died": {"endpoints": [{"instance": 0}]}}
    knee = sweep.recommend_knee([_level(8), dead], min_seq_tok_s=10.0)
    assert knee["recommended_concurrency_per_instance"] == 8
    assert knee["failing_reasons"]["16"] == ["endpoint died"]


def test_liveness_arguments_are_validated():
    with pytest.raises(sweep.SweepError, match="liveness"):
        _config(["http://127.0.0.1:1"], liveness_interval_s=-1).validate()
