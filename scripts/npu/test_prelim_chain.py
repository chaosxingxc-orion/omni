"""Offline tests for the decision logic of prelim_chain.py (CPU only: no docker, no card, no network).

Run: python3 -m unittest -v test_prelim_chain   (from the scripts directory)   or   python3 -m pytest -q test_prelim_chain.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prelim_chain as pc  # noqa: E402


def measure(p10=12.0, lat=300.0, pre=0.0, free=4100, errors=0, timeouts=0, waiting=0.0, requests=100, ok=None):
    summary = {"requests": requests, "ok": requests if ok is None else ok, "errors": errors, "timeouts": timeouts,
               "wall_s": 600.0, "output_tok_s": 1500.0, "completion_tokens": {}, "e2e_tok_s": {"p10": p10},
               "decode_tok_s": {}, "latency_s": {"max": lat}, "ttft_s": {}, "preemptions_delta": pre,
               "metrics_delta": {}, "gauges": [{"num_requests_waiting": {"max": waiting}}]}
    return {"summary": summary, "report": "r.json", "report_sha256": "0" * 64, "min_hbm_free_mib": free,
            "instances": 4}


class ThresholdTests(unittest.TestCase):
    def reasons(self, **kw):
        return pc.evaluate(measure(**kw), 56, 64)[0]

    def test_all_good(self):
        self.assertEqual(self.reasons(), [])

    def test_p10_boundary(self):
        self.assertEqual(self.reasons(p10=10.0), [])
        self.assertTrue(self.reasons(p10=9.99))
        self.assertTrue(self.reasons(p10=None))

    def test_latency_boundary_is_strictly_below_420(self):
        self.assertEqual(self.reasons(lat=419.9), [])
        self.assertTrue(self.reasons(lat=420.0))
        self.assertTrue(self.reasons(lat=None))

    def test_preemptions_must_be_exactly_zero_and_readable(self):
        self.assertEqual(self.reasons(pre=0.0), [])
        self.assertTrue(self.reasons(pre=1.0))
        self.assertTrue(self.reasons(pre=None))

    def test_hbm_boundary_is_2_gib(self):
        self.assertEqual(self.reasons(free=2048), [])
        self.assertTrue(self.reasons(free=2047))
        self.assertTrue(self.reasons(free=None))

    def test_errors_timeouts_and_missing_results(self):
        self.assertTrue(self.reasons(errors=1))
        self.assertTrue(self.reasons(timeouts=1))
        self.assertTrue(self.reasons(ok=99))

    def test_ceiling_bound_only_at_the_scheduler_ceiling_and_not_a_failure(self):
        reasons, row = pc.evaluate(measure(waiting=2.0), 64, 64)
        self.assertEqual(reasons, [])
        self.assertTrue(row["ceiling_bound"])
        self.assertEqual(row["num_requests_waiting_peak"], 2.0)
        self.assertFalse(pc.evaluate(measure(waiting=2.0), 56, 64)[1]["ceiling_bound"])
        self.assertFalse(pc.evaluate(measure(waiting=0.0), 64, 64)[1]["ceiling_bound"])

    def test_poller_waiting_counts_even_when_the_sweep_gauge_missed_it(self):
        m = measure(waiting=0.0)
        m["poll_waiting_max"] = 1.0
        self.assertTrue(pc.evaluate(m, 64, 64)[1]["ceiling_bound"])

    def test_row_uses_the_host_name_for_the_aggregate(self):
        row = pc.evaluate(measure(), 56, 64)[1]
        self.assertIn("host_output_tok_s", row)
        self.assertNotIn("aggregate_output_tok_s", row)
        self.assertNotIn("output_tok_s", row)


class LadderTests(unittest.TestCase):
    def test_p2_start_is_capped_at_48(self):
        self.assertEqual(pc.p2_start([56, 64]), 48)  # plan 3.8: the allowed worst-case levels are 48, 40, 32
        self.assertEqual(pc.p2_start([56]), 48)
        self.assertEqual(pc.p2_start([]), 48)  # 48 passed in the earlier scan

    def test_p2_ladder_is_48_40_32(self):
        self.assertEqual(pc.P2_LADDER, (48, 40, 32))
        self.assertEqual(pc.p2_ladder(64), [48, 40, 32])
        self.assertEqual(pc.p2_ladder(48), [48, 40, 32])
        self.assertEqual(pc.p2_ladder(40), [40, 32])
        self.assertEqual(pc.p2_ladder(32), [32])
        self.assertEqual(pc.P2B_LADDER, (40, 32))


class ChainTests(unittest.TestCase):
    def run_chain(self, root, ops=None, **kw):
        ops = ops or pc.FakeOps(**kw)
        chain_dir = Path(root) / "prelim-chain-T"
        chain_dir.mkdir(exist_ok=True)
        code = pc.Chain(ops, chain_dir, "T", str(root)).run()
        status = json.loads((chain_dir / "chain-status.json").read_text())
        return code, status, ops

    def measured(self, ops):
        return [c.split()[1] for c in ops.calls if c.startswith("measure")]

    def test_everything_passes(self):
        with tempfile.TemporaryDirectory() as d:
            code, status, ops = self.run_chain(d)
        self.assertEqual(code, 0)
        self.assertEqual(self.measured(ops), ["p1-L56", "p1-L64", "p2-L48", "p3-L16", "p3-L32", "p3-L48", "p3-L64", "p3-L96"])
        self.assertEqual(status["decisions"]["P2"]["worst_case_passing_level"], 48)
        self.assertTrue(status["decisions"]["P4"]["argv_identical_to_capture"])
        self.assertEqual(status["state"], "done")

    def test_56_failing_skips_64_and_p2_starts_at_48(self):
        with tempfile.TemporaryDirectory() as d:
            code, status, ops = self.run_chain(d, fail={"p1-L56"})
            self.assertEqual(code, 0)
            self.assertNotIn("p1-L64", self.measured(ops))
            self.assertEqual(status["decisions"]["P1"]["stopped_at_level"], 56)
            self.assertEqual(status["decisions"]["P2"]["start_level"], 48)
            self.assertEqual(status["steps"]["p1-L64"]["decision"]["skipped"], "level 56 failed")

    def test_p2_descends_until_a_level_passes(self):
        with tempfile.TemporaryDirectory() as d:
            _, status, ops = self.run_chain(d, fail={"p2-L48"})
        self.assertEqual([m for m in self.measured(ops) if m.startswith("p2")], ["p2-L48", "p2-L40"])
        self.assertEqual(status["decisions"]["P2"]["worst_case_passing_level"], 40)

    def test_p2_stops_at_32_failing(self):
        with tempfile.TemporaryDirectory() as d:
            _, status, ops = self.run_chain(d, fail={"p2-L48", "p2-L40", "p2-L32"})
        self.assertEqual([m for m in self.measured(ops) if m.startswith("p2")], ["p2-L48", "p2-L40", "p2-L32"])
        self.assertIsNone(status["decisions"]["P2"]["worst_case_passing_level"])

    def test_p2_sends_min_tokens_4096_and_p1_does_not(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, ops = self.run_chain(d)
        calls = [c for c in ops.calls if c.startswith("measure")]
        self.assertTrue(all(c.endswith("min_tokens None") for c in calls if " p1-" in c or " p3-" in c))
        self.assertTrue(all(c.endswith("min_tokens 4096") for c in calls if " p2-" in c))

    def test_a_failing_b_level_stops_the_higher_ones(self):
        with tempfile.TemporaryDirectory() as d:
            code, status, ops = self.run_chain(d, fail={"p3-L48"})
        self.assertEqual(code, 0)
        self.assertEqual([m for m in self.measured(ops) if m.startswith("p3")], ["p3-L16", "p3-L32", "p3-L48"])
        self.assertEqual(status["decisions"]["P3"]["stopped_at_level"], 48)
        self.assertEqual(status["decisions"]["P3"]["passed_levels"], [16, 32])

    def test_b_launch_is_two_tp4_instances_on_the_right_cards_and_numa_nodes(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, ops = self.run_chain(d)
        launches = [c for c in ops.calls if c.startswith("launch") and "saea-arm-b" in c]
        self.assertEqual(launches, [
            "launch NAME=saea-arm-b-0 CPUSET=128-159,192-223 HCCL_BASE_PORT=60400 MAX_NUM_SEQS=128 "
            "npu_serve.sh thinking 0,1,2,3 8210 4",
            "launch NAME=saea-arm-b-1 CPUSET=0-31,64-95 HCCL_BASE_PORT=60500 MAX_NUM_SEQS=128 "
            "npu_serve.sh thinking 4,5,6,7 8211 4"])

    def test_arm_a_is_relaunched_from_the_capture_after_b_is_stopped(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, ops = self.run_chain(d)
        self.assertLess(ops.calls.index("docker stop ['saea-arm-b-0', 'saea-arm-b-1']"),
                        next(i for i, c in enumerate(ops.calls) if c.startswith("launch NAME=saea-arm-a-0")))

    def test_unhealthy_b_stops_the_chain_with_a_status_and_no_p4(self):
        with tempfile.TemporaryDirectory() as d:
            code, status, ops = self.run_chain(d, unhealthy=True)
        self.assertEqual(code, 2)
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["current_step"], "p3-wait-b")
        self.assertIn("not healthy", status["reason"])
        self.assertFalse([c for c in ops.calls if c.startswith("launch NAME=saea-arm-a")])

    def test_resume_skips_finished_steps(self):
        with tempfile.TemporaryDirectory() as d:
            code, _, first = self.run_chain(d, unhealthy=True)
            self.assertEqual(code, 2)
            code, status, second = self.run_chain(d)
        self.assertEqual(code, 0)
        self.assertEqual(self.measured(second), ["p3-L16", "p3-L32", "p3-L48", "p3-L64", "p3-L96"])
        self.assertEqual(status["state"], "done")
        self.assertEqual(status["decisions"]["P2"]["worst_case_passing_level"], 48)

    def test_checkpoints_hold_the_decision(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_chain(d)
            marker = json.loads((Path(d) / "prelim-chain-T" / "steps" / "p1-L56.done.json").read_text())
        self.assertTrue(marker["decision"]["passes"])
        self.assertIn("host_output_tok_s", marker["decision"])


def per_instance_measure(pre=(0, 0, 0, 0), kv=(0.8, 0.8, 0.8, 0.8), unreadable=(), **kw):
    """measure() plus per-instance counters (metrics_before/after) and per-instance KV gauges."""
    m = measure(**kw)
    names = [f"saea-arm-a-{i}" for i in range(4)]
    m["names"] = names
    m["summary"]["metrics_before"] = [{"preemptions": 5.0} for _ in names]
    m["summary"]["metrics_after"] = [None if i in unreadable else {"preemptions": 5.0 + pre[i]} for i in range(4)]
    m["summary"]["gauges"] = [{"num_requests_waiting": {"max": 0.0}, "kv_cache_usage_perc": {"max": kv[i]}}
                              for i in range(4)]
    m["summary"]["preemptions_delta"] = None if unreadable else float(sum(pre))
    return m


class AdmissibilityTests(unittest.TestCase):
    def check(self, **kw):
        return pc.evaluate(per_instance_measure(**kw), 40, 64, worst_case=True)

    def test_admissible_when_zero_preemptions_everywhere_and_kv_at_or_below_90(self):
        reasons, row = self.check(kv=(0.9, 0.5, 0.83, 0.66))
        self.assertEqual(reasons, [])
        self.assertTrue(row["worst_case_admissible"])
        self.assertTrue(row["passes"])

    def test_one_preemption_on_any_single_instance_fails(self):
        for i in range(4):
            pre = [0, 0, 0, 0]
            pre[i] = 1
            reasons, row = self.check(pre=tuple(pre))
            self.assertFalse(row["passes"])
            self.assertFalse(row["worst_case_admissible"])
            self.assertTrue(any(f"on saea-arm-a-{i}" in r and "preemption" in r for r in reasons), reasons)

    def test_kv_peak_just_above_90_fails_on_the_named_instance(self):
        reasons, row = self.check(kv=(0.5, 0.9001, 0.5, 0.5))
        self.assertFalse(row["worst_case_admissible"])
        self.assertTrue(any("KV peak 90.0" in r and "saea-arm-a-1" in r for r in reasons), reasons)

    def test_polled_kv_peak_counts_when_the_sweep_gauge_missed_it(self):
        m = per_instance_measure()
        m["poll_kv_peak"] = [None, None, 0.95, None]
        reasons, row = pc.evaluate(m, 40, 64, worst_case=True)
        self.assertFalse(row["worst_case_admissible"])
        self.assertEqual(row["per_instance"][2]["kv_peak"], 0.95)

    def test_unreadable_counter_or_gauge_is_not_admissible(self):
        reasons, row = self.check(unreadable=(3,))
        self.assertTrue(any("preemption counter unreadable on saea-arm-a-3" in r for r in reasons), reasons)
        m = per_instance_measure()
        m["summary"]["gauges"][0] = {}
        self.assertFalse(pc.evaluate(m, 40, 64, worst_case=True)[1]["worst_case_admissible"])

    def test_rule_applies_only_to_worst_case_levels_but_stats_are_always_recorded(self):
        reasons, row = pc.evaluate(per_instance_measure(kv=(0.99, 0.5, 0.5, 0.5)), 40, 64)  # not a worst-case level
        self.assertEqual(reasons, [])
        self.assertIsNone(row["worst_case_admissible"])
        self.assertEqual([e["kv_peak"] for e in row["per_instance"]], [0.99, 0.5, 0.5, 0.5])
        self.assertEqual([e["preemptions_delta"] for e in row["per_instance"]], [0, 0, 0, 0])

    def test_existing_3_4_thresholds_still_apply_to_worst_case(self):
        reasons, row = self.check(p10=9.0)
        self.assertFalse(row["passes"])
        self.assertTrue(any("p10" in r for r in reasons))

    def test_decision_records_per_instance_deltas_and_kv(self):
        with tempfile.TemporaryDirectory() as d:
            chain_dir = Path(d) / "c"
            ops = pc.FakeOps(preempt={"p2-L48"}, kv_peak={"p2-L48": [0.95, 0.7, 0.6, 0.5]})
            pc.Chain(ops, chain_dir, "T", d).run()
            decision = json.loads((chain_dir / "steps" / "p2-L48.done.json").read_text())["decision"]
        self.assertFalse(decision["passes"])
        self.assertFalse(decision["worst_case_admissible"])
        self.assertEqual([e["preemptions_delta"] for e in decision["per_instance"]], [2.0, 0.0, 0.0, 0.0])
        self.assertEqual([e["kv_peak"] for e in decision["per_instance"]], [0.95, 0.7, 0.6, 0.5])
        self.assertEqual(decision["per_instance"][0]["name"], "saea-arm-a-0")


class LivenessTests(unittest.TestCase):
    def make(self, up, models, tries=3):
        calls = []

        def get(url, timeout=0):
            calls.append(url)
            ok = models.get(url, [])
            answer = ok.pop(0) if ok else False
            return (200, "{}") if answer else None

        pairs = [(f"saea-arm-a-{i}", f"http://127.0.0.1:820{i}") for i in range(4)]
        live = pc.Liveness(pairs, get, lambda: up, tries=tries, pause=0, sleep=lambda s: None)
        return live, calls

    def healthy(self):
        return {f"http://127.0.0.1:820{i}/v1/models": [True] * 5 for i in range(4)}

    def test_all_alive(self):
        live, _ = self.make({f"saea-arm-a-{i}" for i in range(4)}, self.healthy())
        self.assertEqual(live.dead_instances(), [])

    def test_container_gone_is_dead_without_probing_it(self):
        live, calls = self.make({"saea-arm-a-1", "saea-arm-a-2", "saea-arm-a-3"}, self.healthy())
        self.assertEqual(live.dead_instances(), ["saea-arm-a-0"])
        self.assertNotIn("http://127.0.0.1:8200/v1/models", calls)

    def test_endpoint_not_answering_is_dead_after_all_tries(self):
        models = self.healthy()
        models["http://127.0.0.1:8203/v1/models"] = []
        live, calls = self.make({f"saea-arm-a-{i}" for i in range(4)}, models)
        self.assertEqual(live.dead_instances(), ["saea-arm-a-3"])
        self.assertEqual(calls.count("http://127.0.0.1:8203/v1/models"), 3)

    def test_one_slow_answer_is_not_death(self):
        models = self.healthy()
        models["http://127.0.0.1:8202/v1/models"] = [False, False, True]
        live, _ = self.make({f"saea-arm-a-{i}" for i in range(4)}, models)
        self.assertEqual(live.dead_instances(), [])

    def test_unreadable_docker_ps_judges_the_endpoint_only(self):
        live, _ = self.make(None, self.healthy())
        self.assertEqual(live.dead_instances(), [])
        models = self.healthy()
        models["http://127.0.0.1:8201/v1/models"] = []
        live, _ = self.make(None, models)
        self.assertEqual(live.dead_instances(), ["saea-arm-a-1"])

    def test_several_dead_are_all_reported_in_order(self):
        live, _ = self.make({"saea-arm-a-1", "saea-arm-a-2"}, self.healthy())
        self.assertEqual(live.dead_instances(), ["saea-arm-a-0", "saea-arm-a-3"])


class DeadEndpointStopTests(unittest.TestCase):
    def run_chain(self, d, **kw):
        chain_dir = Path(d) / "c"
        ops = pc.FakeOps(**kw)
        code = pc.Chain(ops, chain_dir, "T", d).run()
        return code, json.loads((chain_dir / "chain-status.json").read_text()), ops, chain_dir

    def measured(self, ops):
        return [c.split()[1] for c in ops.calls if c.startswith("measure")]

    def test_dead_endpoint_fails_the_level_and_stops_the_chain_without_stepping_down(self):
        with tempfile.TemporaryDirectory() as d:
            code, status, ops, chain_dir = self.run_chain(d, die={"p2-L48"})
            record = json.loads((chain_dir / "steps" / "p2-L48.failed.json").read_text())
            self.assertFalse((chain_dir / "steps" / "p2-L48.done.json").exists())
        self.assertEqual(code, 2)
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["reason"], "endpoint_died:saea-arm-a-0")
        self.assertEqual(status["current_step"], "p2-L48")
        self.assertEqual(status["steps"]["p2-L48"]["state"], "failed")
        self.assertEqual(status["steps"]["p2-L48"]["decision"]["failing_reasons"], ["endpoint_died:saea-arm-a-0"])
        self.assertEqual(record["failing_reasons"], ["endpoint_died:saea-arm-a-0"])
        self.assertNotIn("p2-L40", self.measured(ops))  # never steps down against dead endpoints
        self.assertNotIn("p2-L32", self.measured(ops))
        self.assertFalse([c for c in ops.calls if c.startswith("launch NAME=saea-arm-b")])  # no P3 either

    def test_a_dead_endpoint_in_any_phase_stops_the_chain(self):
        with tempfile.TemporaryDirectory() as d:
            code, status, ops, _ = self.run_chain(d, die={"p3-L32"})
        self.assertEqual(code, 2)
        self.assertEqual(status["reason"], "endpoint_died:saea-arm-b-0")
        self.assertEqual(self.measured(ops)[-1], "p3-L32")

    def test_resume_after_a_death_measures_the_level_again_instead_of_skipping_it(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_chain(d, die={"p2-L48"})
            code, status, ops, _ = self.run_chain(d)
        self.assertEqual(code, 0)
        self.assertEqual(self.measured(ops)[0], "p2-L48")
        self.assertEqual(status["decisions"]["P2"]["worst_case_passing_level"], 48)

    def test_measure_level_source_polls_every_30_s(self):
        self.assertEqual(pc.LIVENESS_SECONDS, 30)
        self.assertEqual(pc.KV_PEAK_MAX, 0.90)


class P2bTests(unittest.TestCase):
    def run_chain(self, d, **kw):
        chain_dir = Path(d) / "prelim-chain-p2b-T"
        chain_dir.mkdir(exist_ok=True)
        ops = pc.FakeOps(**kw)
        ops.running = {"saea-arm-a-1", "saea-arm-a-2"}  # the two idle survivors
        pc.write_atomic(chain_dir / "arm-a-launch.json", json.dumps({"instances": ops.inspect_arm_a()}))
        code = pc.Chain(ops, chain_dir, "T", d, mode="p2b").run()
        return code, json.loads((chain_dir / "chain-status.json").read_text()), ops, chain_dir

    def measured(self, ops):
        return [c.split()[1] for c in ops.calls if c.startswith("measure")]

    def launched(self, ops):
        return [c.split()[1] for c in ops.calls if c.startswith("launch")]

    def test_pass_at_40_one_pass_arm_a_left_up(self):
        with tempfile.TemporaryDirectory() as d:
            code, status, ops, _ = self.run_chain(d)
        self.assertEqual(code, 0)
        self.assertEqual(self.measured(ops), ["p2b-L40"])
        self.assertEqual(status["decisions"]["P2b"]["worst_case_admissible_level"], 40)
        self.assertEqual(self.launched(ops), [f"NAME=saea-arm-a-{i}" for i in range(4)])
        self.assertEqual(ops.running, {f"saea-arm-a-{i}" for i in range(4)})
        self.assertTrue(status["decisions"]["P2b"]["arm_a_left_up"])
        self.assertFalse([c for c in ops.calls if "arm-b" in c])
        # the min-tokens flag is sent
        self.assertTrue(all(c.endswith("min_tokens 4096") for c in ops.calls if c.startswith("measure")))

    def test_survivors_are_stopped_before_cards_are_checked_and_before_the_launch(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, ops, _ = self.run_chain(d)
        stop = ops.calls.index("docker stop ['saea-arm-a-1', 'saea-arm-a-2']")
        cards = ops.calls.index("cards 0-7 below 4096 MiB")
        launch = next(i for i, c in enumerate(ops.calls) if c.startswith("launch NAME=saea-arm-a-0"))
        self.assertLess(stop, cards)
        self.assertLess(cards, launch)

    def test_a_preempting_40_is_not_admissible_and_32_runs_on_a_fresh_arm_a(self):
        with tempfile.TemporaryDirectory() as d:
            code, status, ops, _ = self.run_chain(d, preempt={"p2b-L40"})
        self.assertEqual(code, 0)
        self.assertEqual(self.measured(ops), ["p2b-L40", "p2b-L32"])
        self.assertEqual(status["decisions"]["P2b"]["worst_case_admissible_level"], 32)
        self.assertEqual(len(self.launched(ops)), 8)  # fresh arm A before each level
        self.assertEqual(status["steps"]["p2b-L40"]["decision"]["worst_case_admissible"], False)

    def test_kv_above_90_at_40_steps_down_to_32(self):
        with tempfile.TemporaryDirectory() as d:
            _, status, ops, _ = self.run_chain(d, kv_peak={"p2b-L40": 0.93})
        self.assertEqual(self.measured(ops), ["p2b-L40", "p2b-L32"])
        self.assertEqual(status["decisions"]["P2b"]["worst_case_admissible_level"], 32)

    def test_32_failing_stops_with_chain_bad_and_leaves_arm_a_up(self):
        with tempfile.TemporaryDirectory() as d:
            code, status, ops, _ = self.run_chain(d, preempt={"p2b-L40", "p2b-L32"})
        self.assertEqual(code, 5)
        self.assertEqual(status["state"], "failed")
        self.assertIsNone(status["decisions"]["P2b"]["worst_case_admissible_level"])
        self.assertEqual(self.measured(ops), ["p2b-L40", "p2b-L32"])
        self.assertEqual(ops.running, {f"saea-arm-a-{i}" for i in range(4)})

    def test_dead_endpoint_at_40_stops_and_never_measures_32(self):
        with tempfile.TemporaryDirectory() as d:
            code, status, ops, chain_dir = self.run_chain(d, die={"p2b-L40"})
            fresh_markers = sorted(p.name for p in (chain_dir / "steps").glob("p2b-fresh40-*"))
        self.assertEqual(code, 2)
        self.assertEqual(status["reason"], "endpoint_died:saea-arm-a-0")
        self.assertEqual(self.measured(ops), ["p2b-L40"])
        self.assertEqual(len(self.launched(ops)), 4)
        self.assertTrue(all(".died-" in n for n in fresh_markers), fresh_markers)  # resume relaunches a fresh arm A

    def test_resume_after_a_death_relaunches_and_remeasures_level_40(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_chain(d, die={"p2b-L40"})
            chain_dir = Path(d) / "prelim-chain-p2b-T"
            ops = pc.FakeOps()
            ops.running = {"saea-arm-a-1", "saea-arm-a-2"}
            code = pc.Chain(ops, chain_dir, "T", d, mode="p2b").run()
        self.assertEqual(code, 0)
        self.assertEqual(self.measured(ops), ["p2b-L40"])
        self.assertEqual(len(self.launched(ops)), 4)  # a fresh arm A: the survivors are stopped and all four relaunched

    def test_registry_is_patched_for_npu2_only(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, ops, _ = self.run_chain(d)
        patched = [c for c in ops.calls if c.startswith("jobs")]
        self.assertTrue(any("prelim-chain-p2b" in c and "sweep-armA-prelim-P2b-mintok4096" in c for c in patched), patched)

    def test_job_entries_are_npu2_and_name_the_p2b_files(self):
        with tempfile.TemporaryDirectory() as d:
            chain = pc.Chain(pc.FakeOps(), Path(d) / "c", "T", d, mode="p2b")
            entries = chain.p2b_job_entries() + chain.a_job_entries(pc.FakeOps().inspect_arm_a(), {})
        self.assertTrue(all(e["host"] == "npu2" for e in entries))
        sweep = next(e for e in entries if e["kind"] == "sweep")
        self.assertIn("prelim-P2b-mintok4096", sweep["params"]["run_prefix"])

    def test_a_busy_survivor_is_never_stopped(self):
        class Busy(pc.FakeOps):
            def running_requests(self, url): return 3.0
        with tempfile.TemporaryDirectory() as d:
            chain_dir = Path(d) / "c"
            chain_dir.mkdir()
            ops = Busy()
            ops.running = {"saea-arm-a-1"}
            pc.write_atomic(chain_dir / "arm-a-launch.json", json.dumps({"instances": ops.inspect_arm_a()}))
            clock = iter(range(0, 10_000_000, 100))  # every time.time() call advances 100 s: the drain window ends
            saved = pc.time.sleep, pc.time.time
            pc.time.sleep, pc.time.time = (lambda s: None), (lambda: next(clock))
            try:
                chain = pc.Chain(ops, chain_dir, "T", d, mode="p2b")
                with self.assertRaises(pc.ChainStop):
                    chain.fresh_arm_a("p2b-fresh40", ops.inspect_arm_a())
            finally:
                pc.time.sleep, pc.time.time = saved
        self.assertNotIn("docker stop ['saea-arm-a-1']", ops.calls)


class RealMeasureLivenessTests(unittest.TestCase):
    """RealOps.measure_level with docker, HTTP and the sweep client faked: the 30 s poll (shortened) ends the level."""

    def setUp(self):
        import threading
        self.threading = threading
        self.killed = threading.Event()
        self.up = {"saea-arm-a-1", "saea-arm-a-2"}
        self.saved = (pc.subprocess.run, pc.http_get, pc.LIVENESS_SECONDS, pc.POLL_SECONDS, pc.LIVENESS_PAUSE_SECONDS)
        pc.LIVENESS_PAUSE_SECONDS = 0.0
        test = self

        class Done:
            def __init__(self, code=0, out=""):
                self.returncode, self.stdout = code, out

        def fake_run(command, **kw):
            if command[:2] == ["docker", "ps"]:
                return Done(0, "\n".join(sorted(test.up)))
            if command[:3] == ["docker", "rm", "-f"]:
                if command[3].startswith("saea-sweep-") and test.client_started:
                    test.killed.set()
                return Done()
            if command[:2] == ["docker", "run"]:  # the sweep client: runs until it is removed (or ends by itself)
                test.client_started = True
                if test.client_ends_by_itself:
                    return Done(0)
                test.killed.wait(20)
                return Done(137)
            raise AssertionError(command)

        pc.subprocess.run = fake_run
        pc.http_get = lambda url, timeout=5.0: (200, "{}") if "/v1/models" in url and test.endpoint_ok else None
        pc.LIVENESS_SECONDS, pc.POLL_SECONDS = 0.05, 0.05
        self.endpoint_ok, self.client_ends_by_itself, self.client_started = True, False, False

    def tearDown(self):
        (pc.subprocess.run, pc.http_get, pc.LIVENESS_SECONDS, pc.POLL_SECONDS,
         pc.LIVENESS_PAUSE_SECONDS) = self.saved

    def measure(self, d):
        ops = pc.RealOps(d, "T", False)
        return ops.measure_level("p2b-L40", f"{d}/sweep-armA-prelim-P2b-mintok4096-T", 40,
                                 ["http://127.0.0.1:8201", "http://127.0.0.1:8202"], 4096, "label",
                                 names=["saea-arm-a-1", "saea-arm-a-2"])

    def test_container_dying_mid_level_kills_the_client_and_raises_endpoint_died(self):
        with tempfile.TemporaryDirectory() as d:
            self.threading.Timer(0.3, lambda: self.up.discard("saea-arm-a-2")).start()
            with self.assertRaises(pc.EndpointDied) as caught:
                self.measure(d)
        self.assertEqual(caught.exception.name, "saea-arm-a-2")
        self.assertTrue(self.killed.is_set())
        self.assertEqual(str(caught.exception), "endpoint_died:saea-arm-a-2")

    def test_endpoint_not_answering_mid_level_is_death_too(self):
        with tempfile.TemporaryDirectory() as d:
            self.threading.Timer(0.3, lambda: setattr(self, "endpoint_ok", False)).start()
            with self.assertRaises(pc.EndpointDied):
                self.measure(d)

    def test_death_after_the_last_poll_is_caught_when_the_client_returns(self):
        self.client_ends_by_itself = True
        pc.LIVENESS_SECONDS = 1000
        self.up.discard("saea-arm-a-2")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(pc.EndpointDied) as caught:
                self.measure(d)  # saea-arm-a-2 is not in docker ps from the start
        self.assertEqual(caught.exception.name, "saea-arm-a-2")


SWEEP_B6144 = {"max_tokens": 7168, "thinking_budget": 6144, "prompt_max": 17408, "timeout": 800}
WC_B6144 = {"levels": [32, 24, 16], "min_tokens": 7168, "max_latency_s": 735.0}


class ResidentWorstcaseTests(unittest.TestCase):
    """--resident-worstcase / --sweep-json: the budget-6144 worst case against the resident arm A (card-queue job R-A3-b6144-prelim)."""

    def run_chain(self, d, sweep=None, wc=None, **kw):
        ops = pc.FakeOps(**kw)
        chain_dir = Path(d) / "prelim-chain-wc-T"
        chain_dir.mkdir(exist_ok=True)
        code = pc.Chain(ops, chain_dir, "T", str(d), mode="wc", sweep=SWEEP_B6144 if sweep is None else sweep,
                        wc=WC_B6144 if wc is None else wc).run()
        return code, json.loads((chain_dir / "chain-status.json").read_text()), ops, chain_dir

    def measured(self, ops):
        return [c.split()[1] for c in ops.calls if c.startswith("measure")]

    def test_levels_in_order_with_overrides_and_resident_arm_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            code, status, ops, chain_dir = self.run_chain(d)
            summary = json.loads(next(Path(d).glob("sweep-armA-prelim-WC-mintok7168-T.json")).read_text())
        self.assertEqual(code, 0)
        self.assertEqual(self.measured(ops), ["wc-L32", "wc-L24", "wc-L16"])
        self.assertEqual(ops.sweeps, [SWEEP_B6144] * 3)
        self.assertTrue(all(c.endswith("min_tokens 7168") for c in ops.calls if c.startswith("measure")))
        self.assertFalse([c for c in ops.calls if c.startswith(("docker stop", "launch", "cards "))])  # nothing stopped or launched
        self.assertEqual(ops.running, {f"saea-arm-a-{i}" for i in range(4)})
        wcd = status["decisions"]["WC"]
        self.assertEqual(wcd["admissible_levels"], [32, 24, 16])
        self.assertEqual(wcd["highest_admissible_level"], 32)
        self.assertTrue(wcd["arm_a_left_up"])
        self.assertEqual(status["state"], "done")
        self.assertEqual(summary["sweep_overrides"], SWEEP_B6144)
        self.assertEqual(summary["levels"][0]["max_latency_s_rule"], 735.0)
        self.assertEqual(summary["levels"][0]["sweep_overrides"], SWEEP_B6144)

    def test_a_failing_level_is_data_every_level_is_still_measured(self):
        with tempfile.TemporaryDirectory() as d:
            code, status, ops, _ = self.run_chain(d, fail={"wc-L32"}, preempt={"wc-L24"})
        self.assertEqual(code, 0)
        self.assertEqual(self.measured(ops), ["wc-L32", "wc-L24", "wc-L16"])
        wcd = status["decisions"]["WC"]
        self.assertEqual(wcd["passing_levels"], [16])
        self.assertEqual(wcd["admissible_levels"], [32, 16])  # L32 fails only the p10 floor; L24 preempts
        self.assertEqual(wcd["results"][1]["failing_reasons"][0][:2], "2 ")  # preemption delta recorded

    def test_no_admissible_level_is_still_done_with_the_facts(self):
        with tempfile.TemporaryDirectory() as d:
            code, status, _, _ = self.run_chain(d, kv_peak={"wc-L32": 0.95, "wc-L24": 0.95, "wc-L16": 0.95})
        self.assertEqual(code, 0)
        self.assertEqual(status["decisions"]["WC"]["admissible_levels"], [])
        self.assertIsNone(status["decisions"]["WC"]["highest_admissible_level"])

    def test_dead_endpoint_stops_the_chain_and_nothing_is_restarted(self):
        with tempfile.TemporaryDirectory() as d:
            code, status, ops, _ = self.run_chain(d, die={"wc-L24"})
        self.assertNotEqual(code, 0)
        self.assertEqual(self.measured(ops), ["wc-L32", "wc-L24"])
        self.assertEqual(status["state"], "failed")
        self.assertFalse([c for c in ops.calls if c.startswith("launch")])

    def test_resume_skips_finished_levels(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_chain(d, die={"wc-L24"})
            code, status, ops, _ = self.run_chain(d)
        self.assertEqual(code, 0)
        self.assertEqual(self.measured(ops), ["wc-L24", "wc-L16"])  # wc-L32 resumed from its checkpoint

    def test_latency_rule_scales_with_the_cap(self):
        m = measure(lat=500.0)
        self.assertTrue(pc.evaluate(m, 32, 64)[0])                       # 420 s rule: fails
        self.assertEqual(pc.evaluate(m, 32, 64, max_latency_s=735)[0], [])  # 735 s rule (7168 tokens): passes
        self.assertTrue(pc.evaluate(measure(lat=735.0), 32, 64, max_latency_s=735)[0])

    def test_sweep_json_parsing_and_validation(self):
        self.assertEqual(pc.parse_sweep_json(""), {})
        self.assertEqual(pc.parse_sweep_json(json.dumps(SWEEP_B6144)), SWEEP_B6144)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "s.json").write_text(json.dumps({"max_tokens": 5000}))
            self.assertEqual(pc.parse_sweep_json(str(Path(d) / "s.json")), {"max_tokens": 5000})
        for bad in ('{"temperature": 1}', '{"max_tokens": 0}', '{"max_tokens": true}', '{"max_tokens": 1.5}', '[1]', "/no/such/file"):
            with self.assertRaises(pc.ChainStop, msg=bad):
                pc.parse_sweep_json(bad)

    def test_wc_validation_mirrors_record_runs(self):
        pc.validate_wc(WC_B6144, SWEEP_B6144)
        with self.assertRaises(pc.ChainStop):
            pc.validate_wc(WC_B6144, {})  # min_tokens 7168 > default max_tokens 4096
        with self.assertRaises(pc.ChainStop):
            pc.validate_wc(dict(WC_B6144, max_latency_s=900.0), SWEEP_B6144)  # latency rule above the client timeout 800
        with self.assertRaises(pc.ChainStop):
            pc.validate_wc(dict(WC_B6144, levels=[32, 32]), SWEEP_B6144)
        with self.assertRaises(pc.ChainStop):
            pc.validate_wc(dict(WC_B6144, levels=[]), SWEEP_B6144)

    def test_sweep_flags_equal_record_runs(self):
        import record_runs as rr
        self.assertEqual(pc.SWEEP_FLAGS, rr.SWEEP_FLAGS)

    def test_cli_dry_run(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            out = subprocess.run([sys.executable, str(Path(pc.__file__)), "--dry-run", "--resident-worstcase", "--levels", "32,24,16",
                                  "--min-tokens", "7168", "--max-latency-s", "735", "--sweep-json", json.dumps(SWEEP_B6144),
                                  "--out-root", d], capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            self.assertIn("CHAIN DONE", out.stdout + out.stderr)
            self.assertIn("sweep overrides wc-L32", out.stdout + out.stderr)
            bad = subprocess.run([sys.executable, str(Path(pc.__file__)), "--dry-run", "--resident-worstcase", "--levels", "32",
                                  "--min-tokens", "7168", "--out-root", d], capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)  # min_tokens above the default max_tokens: refused
            self.assertIn("refused", bad.stdout + bad.stderr)

    def test_real_measure_level_appends_the_sweep_flags_only_when_given(self):
        base = RealMeasureLivenessTests("test_container_dying_mid_level_kills_the_client_and_raises_endpoint_died")
        base.setUp()
        try:
            base.client_ends_by_itself = True
            seen = []
            inner = pc.subprocess.run

            def spy(command, **kw):
                if command[:2] == ["docker", "run"]:
                    seen.append(command)
                return inner(command, **kw)

            pc.subprocess.run = spy
            with tempfile.TemporaryDirectory() as d:
                ops = pc.RealOps(d, "T", False)
                for sweep in (SWEEP_B6144, None):
                    with self.assertRaises(pc.ChainStop):  # the fake client writes no report
                        ops.measure_level("wc-L32", f"{d}/p", 32, ["http://127.0.0.1:8201"], 7168, "label",
                                          names=["saea-arm-a-1"], sweep=sweep)
            with_flags, without = seen
            tail = with_flags[with_flags.index("--min-tokens"):]
            self.assertEqual(tail, ["--min-tokens", "7168", "--max-tokens", "7168", "--thinking-budget", "6144",
                                    "--prompt-max", "17408", "--timeout", "800"])
            self.assertEqual(without[without.index("--min-tokens"):], ["--min-tokens", "7168"])
        finally:
            base.tearDown()


class StopIdleArmATests(unittest.TestCase):
    """--stop-idle-arm-a (card-queue job stop-idle-arm-a): refuse unless idle, docker stop, wait for free cards, CHAIN DONE."""

    def run_chain(self, d, load=None, running=None):
        ops = pc.FakeOps()
        ops.load.update(load or {})
        if running is not None:
            ops.running = set(running)
        chain_dir = Path(d) / "prelim-chain-stop-T"
        chain_dir.mkdir(exist_ok=True)
        code = pc.Chain(ops, chain_dir, "T", str(d), mode="stop").run()
        return code, json.loads((chain_dir / "chain-status.json").read_text()), ops

    def test_idle_instances_are_stopped_then_cards_checked_and_nothing_launched(self):
        with tempfile.TemporaryDirectory() as d:
            code, status, ops = self.run_chain(d)
            marked = pc.read_json(Path(d) / "prelim-chain-stop-T" / "steps" / "stop.done.json")
        self.assertEqual(code, 0)
        self.assertEqual(status["state"], "done")
        self.assertEqual(ops.running, set())
        calls = ops.calls
        removed = next(i for i, c in enumerate(calls) if c.startswith("jobs upsert=[] remove=['vllm-arm-a-0'"))
        stopped = calls.index("docker stop ['saea-arm-a-0', 'saea-arm-a-1', 'saea-arm-a-2', 'saea-arm-a-3']")
        cards = calls.index("cards 0-7 below 4096 MiB")
        self.assertLess(removed, stopped)
        self.assertLess(stopped, cards)
        self.assertFalse([c for c in calls if c.startswith(("launch", "measure"))])
        self.assertEqual(status["decisions"], {})
        self.assertTrue(marked["decision"]["all_idle_before_stop"])

    def test_any_running_or_waiting_request_or_unreadable_endpoint_refuses_and_stops_nothing(self):
        for load in ({"http://127.0.0.1:8203": (1.0, 0.0)}, {"http://127.0.0.1:8200": (0.0, 3.0)},
                     {"http://127.0.0.1:8201": None}):
            with tempfile.TemporaryDirectory() as d:
                code, status, ops = self.run_chain(d, load=load)
            self.assertNotEqual(code, 0, load)
            self.assertEqual(status["state"], "failed")
            self.assertIn("refusing to stop arm A", status["reason"])
            self.assertEqual(ops.running, {f"saea-arm-a-{i}" for i in range(4)})
            self.assertFalse([c for c in ops.calls if c.startswith(("docker stop", "jobs ", "cards "))])

    def test_nothing_running_is_a_no_op_that_still_requires_free_cards(self):
        with tempfile.TemporaryDirectory() as d:
            code, _, ops = self.run_chain(d, running=set())
        self.assertEqual(code, 0)
        self.assertFalse([c for c in ops.calls if c.startswith("docker stop")])
        self.assertIn("cards 0-7 below 4096 MiB", ops.calls)

    def test_cli_dry_run_and_busy_refusal(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            ok = subprocess.run([sys.executable, str(Path(pc.__file__)), "--dry-run", "--stop-idle-arm-a", "--out-root", d],
                                capture_output=True, text=True)
            self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
            self.assertIn("CHAIN DONE", ok.stdout + ok.stderr)
            busy = subprocess.run([sys.executable, str(Path(pc.__file__)), "--dry-run", "--stop-idle-arm-a", "--busy",
                                   "--out-root", d + "/b"], capture_output=True, text=True)
            self.assertNotEqual(busy.returncode, 0)
            self.assertIn("CHAIN-BAD", busy.stdout + busy.stderr)
            self.assertNotIn("CHAIN DONE", busy.stdout + busy.stderr)

    def test_real_load_gauges_reads_both_gauges(self):
        saved = pc.http_get
        try:
            text = "vllm:num_requests_running{engine=\"0\"} 2.0\nvllm:num_requests_waiting{engine=\"0\"} 0.0\n"
            pc.http_get = lambda url, timeout=5.0: (200, text)
            self.assertEqual(pc.RealOps("/tmp", "T", False).load_gauges("http://x"), (2.0, 0.0))
            pc.http_get = lambda url, timeout=5.0: None
            self.assertIsNone(pc.RealOps("/tmp", "T", False).load_gauges("http://x"))
            pc.http_get = lambda url, timeout=5.0: (200, "vllm:num_requests_running 0\n")  # waiting gauge missing
            self.assertIsNone(pc.RealOps("/tmp", "T", False).load_gauges("http://x"))
        finally:
            pc.http_get = saved


class GuardTests(unittest.TestCase):
    def test_only_our_arm_containers_can_be_touched(self):
        pc.RealOps._check_ours(["saea-arm-a-0", "saea-arm-b-1"])
        for name in ("rl_ascend_20261026", "saea-serve-thinking", "saea-arm-c-0", "saea-sweep-x"):
            with self.assertRaises(pc.ChainStop):
                pc.RealOps._check_ours([name])

    def test_patch_jobs_is_atomic_idempotent_and_keeps_other_entries(self):
        with tempfile.TemporaryDirectory() as d:
            jobs = Path(d) / "jobs.json"
            jobs.write_text(json.dumps([{"id": "keep", "kind": "npu_cards", "host": "npu2", "params": {}},
                                        {"id": "vllm-arm-a-0", "kind": "vllm_server", "host": "npu2",
                                         "params": {"container": "saea-arm-a-0", "port": 8200, "log": "old"}}]))
            saved, pc.JOBS = pc.JOBS, str(jobs)
            try:
                ops = pc.RealOps(d, "T", False)
                b = [{"id": "vllm-arm-b-0", "kind": "vllm_server", "host": "npu2", "params": {"container": "saea-arm-b-0", "port": 8210, "log": None}}]
                ops.patch_jobs(b)
                ops.patch_jobs(b)
                ops.patch_jobs([], set_log={"vllm-arm-a-0": "new"})
                self.assertEqual([j["id"] for j in json.loads(jobs.read_text())], ["keep", "vllm-arm-a-0", "vllm-arm-b-0"])
                self.assertEqual(json.loads(jobs.read_text())[1]["params"]["log"], "new")
                ops.patch_jobs([], remove=["vllm-arm-b-0"])
                self.assertEqual([j["id"] for j in json.loads(jobs.read_text())], ["keep", "vllm-arm-a-0"])
            finally:
                pc.JOBS = saved


if __name__ == "__main__":
    unittest.main()
