"""Offline tests for record_runs.py (CPU only: fake operations, no docker, no card, no network).

Run: python3 -m pytest -q test_record_runs.py   (from the scripts directory)
"""
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prelim_chain as pc  # noqa: E402
import record_runs as rr  # noqa: E402

HERE = Path(__file__).resolve().parent
EXAMPLE = json.loads((HERE / "record_run_list.example-armA-knee32.json").read_text())
PROFILE = rr.load_profile("qwen3-omni-thinking-bf16-181-4xtp2-v2")


def run_list(**over):
    doc = json.loads(json.dumps(EXAMPLE))
    doc.update(over)
    return doc


def chain(tmp_path, doc=None, **ops_kw):
    rl = rr.validate_run_list(doc or run_list(), PROFILE)
    ops = rr.FakeRecordOps(PROFILE, **ops_kw)
    c = rr.RecordChain(ops, tmp_path / "chain", "T", str(tmp_path), rl, PROFILE, dry=True, sleep=lambda _s: None)
    return c, ops


def status(c):
    return json.loads((c.dir / "chain-status.json").read_text())


def steps(c):
    return sorted(p.name for p in c.steps_dir.iterdir())


# ---------------------------------------------------------------- run list validation
def test_example_list_is_marked_example_and_validates():
    rl = rr.validate_run_list(run_list(), PROFILE)
    assert rl["example"] is True and "EXAMPLE" in EXAMPLE["label"]
    assert [r["id"] for r in rl["runs"]] == ["R-A1", "R-A2", "R-A3"]
    assert rl["runs"][1]["levels"] == [32, 24] and rl["runs"][1]["passes"] == 3
    assert rl["runs"][2]["min_tokens"] == 4096 and rl["runs"][2]["admissibility"] is True


def test_rule_b_refuses_levels_above_the_admitted_level():
    doc = run_list()
    doc["runs"][0]["levels"] = [8, 40]
    with pytest.raises(pc.ChainStop, match="exceed the admitted level 32"):
        rr.validate_run_list(doc, PROFILE)


def test_admitted_level_cannot_exceed_the_profile_ceiling():
    with pytest.raises(pc.ChainStop, match="admitted_level"):
        rr.validate_run_list(run_list(admitted_level=48), PROFILE)


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d.update(schema="x"), "schema"),
    (lambda d: d["runs"].append(dict(d["runs"][0])), "repeated"),
    (lambda d: d["runs"][0].update(passes=0), "passes"),
    (lambda d: d["runs"][0].update(on_fail="restart"), "on_fail"),
    (lambda d: d["runs"][0].update(min_tokens=0), "min_tokens"),
    (lambda d: d["runs"][0].update(levels=[8, 8]), "levels"),
    (lambda d: d["r0"].pop("base_lock"), "base_lock"),
    (lambda d: d.update(profile="other"), "profile"),
])
def test_bad_run_lists_are_refused(mutate, msg):
    doc = run_list()
    mutate(doc)
    with pytest.raises(pc.ChainStop, match=msg):
        rr.validate_run_list(doc, PROFILE)


def test_requests_per_level_is_passes_times_workers_times_three_contacts():
    run = rr.validate_run_list(run_list(), PROFILE)["runs"][1]
    assert rr.requests_for(run, 32, 4) == 3 * 32 * 4 * 3 == 1152


# ---------------------------------------------------------------- commands
def test_sweep_command_carries_min_tokens_requests_seed_and_the_liveness_interval():
    cmd = rr.sweep_cmd("n", ["http://a", "http://b"], 32, 288, 4096, "20261028", "lbl", "/o.json", "224-255")
    assert cmd[cmd.index("--min-tokens") + 1] == "4096" and cmd[cmd.index("--requests-per-level") + 1] == "288"
    assert cmd[cmd.index("--seed") + 1] == "20261028" and cmd[cmd.index("--liveness-interval") + 1] == "30"
    assert "--min-tokens" not in rr.sweep_cmd("n", ["http://a"], 8, 72, None, "s", "l", "/o.json", "1")
    assert "--restart" not in cmd


def test_receipt_command_names_every_instance_and_attests_the_image():
    cmd = rr.receipt_cmd({"i0": 11, "i1": 12}, "sha256:" + "a" * 64, "/x/r.json", "0:0")
    assert cmd.count("--instance") == 2 and "i0=11" in cmd and "i1=12" in cmd
    assert cmd[cmd.index("--image-id") + 1] == "sha256:" + "a" * 64 and "--pid" in cmd


def test_verification_needs_all_eleven_checks_selected_and_passing():
    names = ["a"] * 11
    assert rr.verification_verdict({"selected_checks": names, "failed_checks": [], "verdict": "pass"})[0]
    assert not rr.verification_verdict({"selected_checks": names[:10], "failed_checks": [], "verdict": "pass"})[0]
    assert not rr.verification_verdict({"selected_checks": names, "failed_checks": ["kv"], "verdict": "fail"})[0]
    assert not rr.verification_verdict(None)[0]


def test_server_seed_is_read_from_the_container_cmd():
    assert rr.seed_from_cmd(["serve", "m", "--seed", "0", "--port", "1"]) == "0"
    assert rr.seed_from_cmd(["serve", "--seed=7"]) == "7" and rr.seed_from_cmd(["serve"]) is None


# ---------------------------------------------------------------- the chain
def test_the_example_list_runs_r0_in_order_then_every_level(tmp_path):
    c, ops = chain(tmp_path)
    assert c.run() == 0
    st = status(c)
    assert st["state"] == "done" and st["example"] is True
    order = list(c.status["steps"])  # insertion order (the file is key-sorted)
    r0 = ["gate-cards", "disk", "deploy-up", "health", "deploy-status", "startup-lines", "receipt", "verify-base",
          "smoke", "manifest"]
    assert order[:10] == r0
    assert order[10:] == ["R-A1-L8", "R-A1-L16", "R-A2-L32", "R-A2-L24", "R-A3-L32"]
    flat = [" ".join(cmd) for cmd in ops.commands]
    assert "npu_deploy.sh up qwen3-omni-thinking-bf16-181-4xtp2-v2" in flat[0]
    # nothing in the whole run stops, restarts or removes a server
    assert not any(w in " ".join(flat) for w in (" stop", " down", "--restart", " rm "))
    assert st["decisions"]["R-A2"]["passed_levels"] == [32, 24]


def test_a_probe_before_and_after_each_level_with_agreement_recorded(tmp_path):
    c, ops = chain(tmp_path)
    c.run()
    sends = [cmd for cmd in ops.commands if "greedy_probe.py" in " ".join(cmd) and "send" in cmd]
    assert len(sends) == 5 * 2 * 4 and all("--sequential" in cmd for cmd in sends)
    row = json.loads(next(Path(tmp_path).glob("sweep-rec-R-A1-T-L8.row.json")).read_text())
    assert row["greedy_probe"]["gated"] is False and row["client_seed"] == "20261028"
    assert row["server_seeds"] == {n: "0" for n in c.names} and row["max_num_seqs_ceiling"] == 32
    assert len(row["greedy_probe"]["sequential_before_vs_after"]) == 4


def test_manifest_records_seeds_profile_and_example_flag(tmp_path):
    c, _ = chain(tmp_path)
    c.run()
    m = json.loads(Path(c.paths["manifest"]).read_text())
    assert m["example"] is True and m["client_seed"] == "20261028" and m["probe"]["decoding_seed"] == 777
    assert m["server_seeds"] and m["max_num_seqs_ceiling"] == 32 and m["admitted_level"] == 32
    assert m["profile_sha256"] == PROFILE["_sha256"] and "npu_smi_after" in m


def test_busy_cards_stop_before_anything_is_started(tmp_path):
    c, ops = chain(tmp_path, cards_busy=True)
    assert c.run() == rr.EXIT_STOP
    assert status(c)["state"] == "failed" and "busy_threshold_mib 4096" in status(c)["reason"]
    assert not any("npu_deploy.sh" in " ".join(cmd) for cmd in ops.commands)


def test_verification_below_11_of_11_stops_before_smoke_and_levels(tmp_path):
    c, ops = chain(tmp_path, verify_fail=True)
    assert c.run() == rr.EXIT_STOP
    assert "verify-base" in status(c)["reason"] and not any("npu_smoke" in " ".join(cmd) for cmd in ops.commands)
    assert not any("synthetic_sweep" in " ".join(cmd) for cmd in ops.commands)


def test_smoke_failure_stops_before_the_levels(tmp_path):
    c, ops = chain(tmp_path, smoke_fail=True)
    assert c.run() == rr.EXIT_STOP and "smoke" in status(c)["reason"]
    assert not any("synthetic_sweep" in " ".join(cmd) for cmd in ops.commands)


def test_on_fail_stop_run_skips_the_rest_of_the_run_and_continues_with_the_next(tmp_path):
    c, _ = chain(tmp_path, fail={"R-A1-L8"})
    assert c.run() == 0
    st = status(c)
    assert st["decisions"]["R-A1"]["failed_levels"] == [8] and st["decisions"]["R-A1"]["stopped_at_level"] == 8
    assert "skipped" in json.loads((c.steps_dir / "R-A1-L16.done.json").read_text())["decision"]
    assert st["decisions"]["R-A2"]["passed_levels"] == [32, 24]


def test_on_fail_continue_measures_every_level(tmp_path):
    c, _ = chain(tmp_path, fail={"R-A2-L32"})
    assert c.run() == 0
    assert status(c)["decisions"]["R-A2"]["passed_levels"] == [24]
    assert status(c)["decisions"]["R-A2"]["failed_levels"] == [32]


def test_stop_chain_run_failing_stops_with_gate_failed(tmp_path):
    c, _ = chain(tmp_path, preempt={"R-A3-L32"})
    assert c.run() == rr.EXIT_GATE_FAILED
    assert status(c)["state"] == "failed" and "preemption" in status(c)["reason"]


def test_worst_case_admissibility_rule_applies_only_where_the_run_asks(tmp_path):
    c, _ = chain(tmp_path, preempt={"R-A3-L32"})
    c.run()
    row = json.loads((tmp_path / "sweep-rec-R-A3-mintok4096-T-L32.row.json").read_text())
    assert row["worst_case"] is True and row["worst_case_admissible"] is False and row["passes"] is False
    doc = run_list()
    doc["runs"][2]["admissibility"] = False
    c2, _ = chain(tmp_path / "b", doc, kv_peak={"R-A3-L32": 0.97})
    assert c2.run() == 0
    assert status(c2)["decisions"]["R-A3"]["passed_levels"] == [32]


def test_kv_peak_above_90_percent_fails_an_admissibility_run(tmp_path):
    c, _ = chain(tmp_path, kv_peak={"R-A3-L32": 0.95})
    assert c.run() == rr.EXIT_GATE_FAILED
    assert any("KV peak" in r for r in status(c)["reason"].split("[")[1].split(","))


def test_a_ceiling_bound_level_fails(tmp_path):
    c, _ = chain(tmp_path, ceiling_bound_levels={32})
    assert c.run() == rr.EXIT_GATE_FAILED or status(c)["decisions"]["R-A2"]["failed_levels"]
    row = json.loads((tmp_path / "sweep-rec-R-A2-T-L32.row.json").read_text())
    assert row["ceiling_bound"] and not row["passes"]
    assert any("ceiling-bound" in r for r in row["failing_reasons"])


def test_a_dead_endpoint_is_a_fail_and_a_stop_never_a_restart(tmp_path):
    c, ops = chain(tmp_path, die={"R-A2-L32"})
    assert c.run() == rr.EXIT_DIED
    st = status(c)
    assert st["state"] == "failed" and st["reason"].startswith("endpoint_died:")
    failed = json.loads((c.steps_dir / "R-A2-L32.failed.json").read_text())
    assert failed["passes"] is False and failed["failing_reasons"][0].startswith("endpoint_died:")
    assert not (c.steps_dir / "R-A2-L32.done.json").exists()
    assert not (c.steps_dir / "R-A2-L24.done.json").exists() and not (c.steps_dir / "R-A3-L32.done.json").exists()
    assert not any(w in " ".join(" ".join(cmd) for cmd in ops.commands) for w in (" stop", "--restart", " down"))


def test_a_death_found_by_the_check_after_the_level_stops_the_chain(tmp_path):
    c, _ = chain(tmp_path, dead_after={"R-A1-L16": "saea-omni-i2"})
    assert c.run() == rr.EXIT_DIED
    assert (c.steps_dir / "R-A1-L16.failed.json").exists()


def test_resume_skips_finished_steps_and_restarts_nothing(tmp_path):
    c, ops = chain(tmp_path, die={"R-A2-L32"})
    assert c.run() == rr.EXIT_DIED
    first = len(ops.commands)
    c2, ops2 = chain(tmp_path)  # same directory: the died level is re-measured, everything finished is skipped
    assert c2.run() == 0
    flat = [" ".join(cmd) for cmd in ops2.commands]
    assert not any("npu_deploy.sh" in f for f in flat) and not any("write_vllm_runtime_receipt" in f for f in flat)
    assert not any("R-A1-DRYRUN" in f for f in flat) and any("R-A2" in f for f in flat)
    assert status(c2)["state"] == "done" and first > 0


def test_watchdog_entries_are_for_npu2_only(tmp_path):
    c, _ = chain(tmp_path)
    jobs = c.jobs("chain") + c.jobs("servers") + c.jobs("sweep", c.rl["runs"][0])
    assert {j["host"] for j in jobs} == {"npu2"} and {j["kind"] for j in jobs} == {"generic_log", "vllm_server", "sweep"}
    assert jobs[0]["params"]["status_file"].endswith("chain-status.json")


def test_the_service_is_left_up_no_stop_or_down_exists_in_the_ops():
    import inspect
    src = inspect.getsource(rr.RecordOps)
    assert "deploy_down" not in src and '"down"' not in src and "docker\", \"stop\"" not in src
    assert "--restart" not in inspect.getsource(rr.sweep_cmd)


def test_example_lists_are_refused_for_real_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["record_runs.py", "--run-list", str(HERE / "record_run_list.example-armA-knee32.json")])
    with pytest.raises(SystemExit) as stop:
        rr.main()
    assert "EXAMPLE" in str(stop.value)


def test_dry_run_cli_prints_commands_and_exits_zero(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["record_runs.py", "--dry-run", "--out-root", str(tmp_path), "--ts", "D",
                                      "--run-list", str(HERE / "record_run_list.example-armA-knee32.json")])
    assert rr.main() == 0
    out = capsys.readouterr().out
    assert "npu_deploy.sh up qwen3-omni-thinking-bf16-181-4xtp2-v2" in out and "CHAIN DONE" in out


# ---------------------------------------------------------------- per-run sweep overrides and latency rule (raised-cap worst case)
def test_sweep_overrides_default_to_none_and_leave_the_command_unchanged():
    rl = rr.validate_run_list(run_list(), PROFILE)
    assert all(r["sweep"] == {} and r["max_latency_s"] == 420.0 for r in rl["runs"])
    base = rr.sweep_cmd("n", ["http://a"], 8, 72, 4096, "s", "l", "/o.json", "1")
    assert base == rr.sweep_cmd("n", ["http://a"], 8, 72, 4096, "s", "l", "/o.json", "1", {})
    assert "--max-tokens" not in base and "--timeout" not in base


def test_sweep_overrides_become_sweep_client_flags():
    sweep = {"max_tokens": 7168, "thinking_budget": 6144, "prompt_max": 17408, "timeout": 800}
    cmd = rr.sweep_cmd("n", ["http://a"], 32, 96, 7168, "s", "l", "/o.json", "1", sweep)
    for flag, value in (("--max-tokens", "7168"), ("--thinking-budget", "6144"), ("--prompt-max", "17408"),
                        ("--timeout", "800"), ("--min-tokens", "7168")):
        assert cmd[cmd.index(flag) + 1] == value


def _raised(doc):
    doc["runs"][2].update(min_tokens=7168, sweep={"max_tokens": 7168, "timeout": 800}, max_latency_s=735)


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d["runs"][2].update(min_tokens=8192), "exceeds the sweep max_tokens"),
    (lambda d: d["runs"][2].update(sweep={"max_tokens": 8192, "bogus": 1}, min_tokens=8192), "sweep must be"),
    (lambda d: d["runs"][2].update(sweep={"max_tokens": 0}), "sweep must be"),
    (lambda d: d["runs"][2].update(sweep={"max_tokens": True}), "sweep must be"),
    (lambda d: d["runs"][2].update(max_latency_s=300), "max_latency_s"),
    (lambda d: d["runs"][2].update(max_latency_s="735"), "max_latency_s"),
    (lambda d: d["runs"][2].update(max_latency_s=735), "exceeds the sweep client timeout"),
])
def test_bad_sweep_overrides_and_latency_rules_are_refused(mutate, msg):
    doc = run_list()
    mutate(doc)
    with pytest.raises(pc.ChainStop, match=msg):
        rr.validate_run_list(doc, PROFILE)


def test_max_latency_scales_the_evaluate_rule_and_defaults_to_420():
    ops = rr.FakeRecordOps(PROFILE)
    measure = ops.measure_level("R-A3-L32", "/tmp/p", 32, ["http://a"], 7168, "l", names=["i0"], requests=96)
    measure["summary"]["latency_s"]["max"] = 600.0
    assert any("max latency" in r for r in pc.evaluate(measure, 32, 32)[0])
    assert any("max latency" in r for r in pc.evaluate(measure, 32, 32, max_latency_s=420.0)[0])
    assert not any("max latency" in r for r in pc.evaluate(measure, 32, 32, max_latency_s=735.0)[0])


def test_dry_chain_passes_the_overrides_to_the_sweep_command(tmp_path):
    doc = run_list()
    _raised(doc)
    doc["runs"][2]["sweep"]["thinking_budget"] = 6144
    c, ops = chain(tmp_path, doc)
    c.run()
    sweeps = [cmd for cmd in ops.commands if "scripts/serving/synthetic_sweep.py" in cmd]
    assert any("--max-tokens" in cmd and cmd[cmd.index("--max-tokens") + 1] == "7168" for cmd in sweeps)
    assert sum("--max-tokens" in cmd for cmd in sweeps) == 1  # only the run that declared it


def test_base_lock_v1_combined_list_has_one_r0_and_three_runs():
    doc = json.loads((HERE / "record_run_list.base-lock-v1.json").read_text())
    rl = rr.validate_run_list(doc, PROFILE)
    assert rl["example"] is False
    assert [r["id"] for r in rl["runs"]] == ["R-A2-L32", "R-A3-L32", "R-A3-b6144"]
    assert [r["levels"] for r in rl["runs"]] == [[32], [32], [32, 24, 16]]
    assert [r["min_tokens"] for r in rl["runs"]] == [None, 4096, 7168]
    assert [r["on_fail"] for r in rl["runs"]] == ["stop_chain", "stop_chain", "continue"]
    b = json.loads((HERE / "record_run_list.budget6144-worstcase.json").read_text())["runs"][0]
    assert rl["runs"][2]["sweep"] == b["sweep"] and rl["runs"][2]["max_latency_s"] == b["max_latency_s"] == 735.0
    assert rl["runs"][1]["admissibility"] and rl["runs"][2]["admissibility"]


def test_base_lock_v1_combined_list_runs_r0_once_then_every_run(tmp_path):
    import subprocess
    out = subprocess.run([sys.executable, str(HERE / "record_runs.py"), "--dry-run", "--run-list",
                          str(HERE / "record_run_list.base-lock-v1.json"), "--out-root", str(tmp_path)],
                         capture_output=True, text=True)
    text = out.stdout + out.stderr
    assert out.returncode == 0 and "CHAIN DONE" in text
    steps = [l.split("STEP ")[1] for l in text.splitlines() if "STEP " in l]
    assert steps.count("deploy-up") == 1
    assert steps[-5:] == ["R-A2-L32-L32", "R-A3-L32-L32", "R-A3-b6144-L32", "R-A3-b6144-L24", "R-A3-b6144-L16"]
