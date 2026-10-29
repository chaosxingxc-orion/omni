"""Offline tests for the P3 + P3b + P4 chain (prelim_chain.py --p3). CPU only: no docker, no card, no network.

Run: python3 -m unittest -v test_prelim_P3_P3b   (from the scripts directory)
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prelim_chain as pc  # noqa: E402


class GateOps(pc.FakeOps):
    """FakeOps whose gate file reads running for `waits` reads, then final; the pid outlives the final state by `pid_extra` reads."""

    def __init__(self, waits=2, pid_extra=0, final_state="done", **kw):
        super().__init__(**kw)
        self.waits, self.pid_extra, self.final_state, self.reads = waits, pid_extra, final_state, 0

    def gate_status(self, path):
        self.reads += 1
        if self.reads <= self.waits:
            return {"state": "running", "pid": 4242, "current_step": "tracks-running"}
        return {"state": self.final_state, "pid": 4242, "reason": None}

    def pid_alive(self, pid):
        return self.reads <= self.waits + self.pid_extra


class P3Base(unittest.TestCase):
    def run_chain(self, d, ops=None, sleeps=None, **kw):
        chain_dir = Path(d) / "prelim-chain-p3-T"
        chain_dir.mkdir(exist_ok=True)
        ops = ops or pc.FakeOps(**kw)
        pc.write_atomic(chain_dir / "arm-a-launch.json", json.dumps({"instances": ops.inspect_arm_a()}))
        code = pc.Chain(ops, chain_dir, "T", d, mode="p3", gate_path="/dry/gate.json", gate_poll=120,
                        sleep=(sleeps.append if sleeps is not None else (lambda _s: None))).run()
        return code, json.loads((chain_dir / "chain-status.json").read_text()), ops, chain_dir

    def measured(self, ops):
        return [c.split()[1] for c in ops.calls if c.startswith("measure")]

    def launched(self, ops):
        return [c.split()[1] for c in ops.calls if c.startswith("launch")]


class P3bLadderTests(unittest.TestCase):
    def test_ladder_is_highest_passing_first_then_descending(self):
        self.assertEqual(pc.p3b_ladder([16, 32, 48, 64, 96]), [96, 64, 48, 32, 16])
        self.assertEqual(pc.p3b_ladder([16, 32]), [32, 16])
        self.assertEqual(pc.p3b_ladder([]), [])

    def test_constants(self):
        self.assertEqual(pc.P3_LEVELS, (16, 32, 48, 64, 96))
        self.assertEqual(pc.P3B_MIN_TOKENS, 4096)
        self.assertEqual(pc.GATE_POLL_S, 120)
        self.assertEqual(pc.KV_PEAK_MAX, 0.90)
        self.assertEqual([s["max_num_seqs"] for s in pc.ARM_B], [128, 128])


class GateTests(unittest.TestCase):
    def test_final_requires_a_final_state_and_a_gone_pid(self):
        self.assertTrue(pc.gate_final({"state": "done", "pid": 1}, False)[0])
        self.assertTrue(pc.gate_final({"state": "failed", "pid": 1}, False)[0])
        self.assertTrue(pc.gate_final({"state": "interrupted", "pid": 1}, False)[0])
        self.assertFalse(pc.gate_final({"state": "running", "pid": 1}, False)[0])
        self.assertFalse(pc.gate_final({"state": "done", "pid": 1}, True)[0])
        self.assertFalse(pc.gate_final(None, False)[0])

    def test_e_entry_pattern_matches_only_the_e_chain_ids(self):
        for ident in ("prelim-E-chain", "vllm-prelim-E2-shm", "sweep-prelim-E1-base-pass1", "sweep-prelim-E2-shm-pass3"):
            self.assertTrue(pc.E_ENTRY.match(ident), ident)
        for ident in ("vllm-arm-a-0", "prelim-chain-p2b", "sweep-armA-prelim-P1", "cards-npu2", "suite-wave1",
                      "sweep-armB-prelim-P3", "prelim-chain-p3"):
            self.assertFalse(pc.E_ENTRY.match(ident), ident)


class P3ChainTests(P3Base):
    def test_all_pass_p3_then_p3b_at_96_then_p4_arm_a_restored(self):
        with tempfile.TemporaryDirectory() as d:
            code, status, ops, _ = self.run_chain(d)
        self.assertEqual(code, 0)
        self.assertEqual(status["state"], "done")
        self.assertEqual(self.measured(ops), ["p3-L16", "p3-L32", "p3-L48", "p3-L64", "p3-L96", "p3b-L96"])
        self.assertEqual(status["decisions"]["P3b"]["worst_case_admissible_level"], 96)
        self.assertEqual(status["decisions"]["P3b"]["admitted_host_concurrency"], 192)
        self.assertEqual(ops.running, {f"saea-arm-a-{i}" for i in range(4)})  # B stopped, A back
        self.assertTrue(status["decisions"]["P4"]["argv_identical_to_capture"])

    def test_b_is_two_tp4_with_max_num_seqs_128_and_the_coded_cpusets_and_hccl_ports(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, ops, _ = self.run_chain(d)
        b = [c for c in ops.calls if c.startswith("launch NAME=saea-arm-b")]
        self.assertEqual(len(b), 2)
        self.assertIn("NAME=saea-arm-b-0 CPUSET=128-159,192-223 HCCL_BASE_PORT=60400 MAX_NUM_SEQS=128 "
                      "npu_serve.sh thinking 0,1,2,3 8210 4", b[0])
        self.assertIn("NAME=saea-arm-b-1 CPUSET=0-31,64-95 HCCL_BASE_PORT=60500 MAX_NUM_SEQS=128 "
                      "npu_serve.sh thinking 4,5,6,7 8211 4", b[1])

    def test_p3_levels_are_normal_mix_and_p3b_sends_min_tokens_4096(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, ops, _ = self.run_chain(d)
        for call in (c for c in ops.calls if c.startswith("measure p3-")):
            self.assertTrue(call.endswith("min_tokens None"), call)
        for call in (c for c in ops.calls if c.startswith("measure p3b-")):
            self.assertTrue(call.endswith("min_tokens 4096"), call)

    def test_order_stop_a_then_cards_then_b_then_levels_then_stop_b_then_a(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, ops, _ = self.run_chain(d)
        i = ops.calls.index
        stop_a = i("docker stop ['saea-arm-a-0', 'saea-arm-a-1', 'saea-arm-a-2', 'saea-arm-a-3']")
        cards = i("cards 0-7 below 4096 MiB")
        launch_b = next(k for k, c in enumerate(ops.calls) if c.startswith("launch NAME=saea-arm-b-0"))
        first = next(k for k, c in enumerate(ops.calls) if c.startswith("measure p3-L16"))
        last = next(k for k, c in enumerate(ops.calls) if c.startswith("measure p3b-L96"))
        stop_b = i("docker stop ['saea-arm-b-0', 'saea-arm-b-1']")
        launch_a = next(k for k, c in enumerate(ops.calls) if c.startswith("launch NAME=saea-arm-a-0"))
        self.assertTrue(stop_a < cards < launch_b < first < last < stop_b < launch_a)

    def test_p3_stops_below_the_first_failing_level_and_p3b_starts_at_the_highest_passing(self):
        with tempfile.TemporaryDirectory() as d:
            code, status, ops, _ = self.run_chain(d, fail={"p3-L64"})
        self.assertEqual(code, 0)
        self.assertEqual(self.measured(ops), ["p3-L16", "p3-L32", "p3-L48", "p3-L64", "p3b-L48"])
        self.assertEqual(status["decisions"]["P3b"]["ladder"], [48, 32, 16])
        self.assertEqual(status["decisions"]["P3b"]["worst_case_admissible_level"], 48)

    def test_p3b_steps_down_on_a_preemption_on_any_single_instance(self):
        with tempfile.TemporaryDirectory() as d:
            code, status, ops, _ = self.run_chain(d, preempt={"p3b-L96", "p3b-L64"})
        self.assertEqual(code, 0)
        self.assertEqual([m for m in self.measured(ops) if m.startswith("p3b")], ["p3b-L96", "p3b-L64", "p3b-L48"])
        decision = status["decisions"]["P3b"]
        self.assertEqual(decision["worst_case_admissible_level"], 48)
        self.assertEqual([a["level"] for a in decision["attempts"]], [96, 64, 48])
        self.assertEqual(decision["attempts"][0]["per_instance"][0]["preemptions_delta"], 2.0)
        self.assertEqual(decision["attempts"][0]["per_instance"][1]["preemptions_delta"], 0.0)
        self.assertTrue(any("preemption(s) on saea-arm-b-0" in r for r in decision["attempts"][0]["failing_reasons"]))

    def test_p3b_steps_down_when_kv_peak_is_above_90_percent(self):
        with tempfile.TemporaryDirectory() as d:
            _, status, ops, _ = self.run_chain(d, kv_peak={"p3b-L96": [0.80, 0.91], "p3b-L64": 0.9})
        self.assertEqual([m for m in self.measured(ops) if m.startswith("p3b")], ["p3b-L96", "p3b-L64"])
        self.assertEqual(status["decisions"]["P3b"]["worst_case_admissible_level"], 64)  # exactly 0.90 is admissible
        self.assertTrue(any("KV peak 91.0 %" in r and "saea-arm-b-1" in r
                            for r in status["decisions"]["P3b"]["attempts"][0]["failing_reasons"]))

    def test_no_admissible_level_still_restores_arm_a_and_records_none(self):
        with tempfile.TemporaryDirectory() as d:
            code, status, ops, _ = self.run_chain(d, preempt={f"p3b-L{n}" for n in (16, 32, 48, 64, 96)})
        self.assertEqual(code, 0)
        self.assertIsNone(status["decisions"]["P3b"]["worst_case_admissible_level"])
        self.assertIn("no admissible", status["reason"])
        self.assertEqual(ops.running, {f"saea-arm-a-{i}" for i in range(4)})

    def test_no_passing_p3_level_skips_p3b_and_restores_arm_a(self):
        with tempfile.TemporaryDirectory() as d:
            code, status, ops, _ = self.run_chain(d, fail={"p3-L16"})
        self.assertEqual(code, 0)
        self.assertEqual(self.measured(ops), ["p3-L16"])
        self.assertEqual(status["decisions"]["P3b"]["ladder"], [])
        self.assertIsNone(status["decisions"]["P3b"]["worst_case_admissible_level"])
        self.assertEqual(ops.running, {f"saea-arm-a-{i}" for i in range(4)})

    def test_dead_endpoint_in_p3b_fails_the_level_and_stops_without_stepping_down_or_restoring(self):
        with tempfile.TemporaryDirectory() as d:
            code, status, ops, chain_dir = self.run_chain(d, die={"p3b-L96"})
            record = json.loads((chain_dir / "steps" / "p3b-L96.failed.json").read_text())
            self.assertFalse((chain_dir / "steps" / "p3b-L96.done.json").exists())
        self.assertEqual(code, 2)
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["reason"], "endpoint_died:saea-arm-b-0")
        self.assertEqual(status["steps"]["p3b-L96"]["state"], "failed")
        self.assertEqual(record["failing_reasons"], ["endpoint_died:saea-arm-b-0"])
        self.assertNotIn("p3b-L64", self.measured(ops))
        self.assertFalse([c for c in ops.calls if c.startswith("launch NAME=saea-arm-a")])  # no P4 after a death
        self.assertFalse([c for c in ops.calls if c.startswith("docker stop ['saea-arm-b")])

    def test_resume_skips_finished_steps_and_remeasures_the_dead_level(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_chain(d, die={"p3b-L96"})
            code, status, ops, _ = self.run_chain(d)
        self.assertEqual(code, 0)
        self.assertEqual(self.measured(ops), ["p3b-L96"])  # P3 levels came from the checkpoints
        self.assertEqual(status["decisions"]["P3b"]["worst_case_admissible_level"], 96)

    def test_reports_carry_prelim_p3_and_p3b_in_their_prefixes(self):
        with tempfile.TemporaryDirectory() as d:
            _, status, _, _ = self.run_chain(d)
        self.assertIn("sweep-armB-prelim-P3-T", status["prefixes"]["P3"])
        self.assertIn("sweep-armB-prelim-P3b-mintok4096-T", status["prefixes"]["P3b"])

    def test_no_card_work_and_no_stop_before_the_gate_opens(self):
        with tempfile.TemporaryDirectory() as d:
            ops = GateOps(waits=3)
            sleeps = []
            code, status, ops, _ = self.run_chain(d, ops=ops, sleeps=sleeps)
        self.assertEqual(code, 0)
        self.assertEqual(sleeps[:3], [120, 120, 120])  # one read per 2 minutes while the gate is closed
        self.assertEqual(ops.reads, 4)
        first_card_call = next(k for k, c in enumerate(ops.calls) if "docker stop" in c or c.startswith("launch"))
        self.assertEqual(ops.calls[:first_card_call].count("sampler start"), 1)  # only after the gate
        self.assertEqual(status["steps"]["gate"]["decision"]["waited_s"], 360)

    def test_gate_stays_closed_while_the_pid_is_alive_even_in_a_final_state(self):
        with tempfile.TemporaryDirectory() as d:
            ops = GateOps(waits=1, pid_extra=2)
            sleeps = []
            code, status, ops, _ = self.run_chain(d, ops=ops, sleeps=sleeps)
        self.assertEqual(code, 0)
        self.assertEqual(len(sleeps), 3)
        self.assertEqual(ops.reads, 4)

    def test_failed_gate_chain_still_opens_the_gate_once_its_pid_is_gone(self):
        with tempfile.TemporaryDirectory() as d:
            ops = GateOps(waits=0, final_state="failed")
            code, status, _, _ = self.run_chain(d, ops=ops)
        self.assertEqual(code, 0)
        self.assertEqual(status["steps"]["gate"]["decision"]["gate_state"], "failed")

    def test_registry_entries_are_npu2_only_e_entries_removed_a_entries_removed_then_readded(self):
        with tempfile.TemporaryDirectory() as d:
            _, status, ops, _ = self.run_chain(d)
        jobs = [c for c in ops.calls if c.startswith("jobs ")]
        self.assertIn("jobs upsert=['prelim-chain-p3', 'sweep-armB-prelim-P3', 'sweep-armB-prelim-P3b-mintok4096'] "
                      "remove=[] set_log=[]", jobs[0])
        removed = [c for c in jobs if "remove=['prelim-E-chain'" in c]
        self.assertEqual(len(removed), 1)
        self.assertNotIn("vllm-arm-a-1", removed[0])  # only the E trials' own ids
        self.assertNotIn("prelim-chain-p2b", removed[0])
        self.assertTrue(any("remove=['vllm-arm-a-0', 'vllm-arm-a-1', 'vllm-arm-a-2', 'vllm-arm-a-3']" in c for c in jobs))
        self.assertTrue(any("remove=['vllm-arm-b-0', 'vllm-arm-b-1']" in c for c in jobs))
        readded = [c for c in jobs if "upsert=['vllm-arm-a-0'" in c]
        self.assertEqual(len(readded), 1)

    def test_job_entries_name_the_p3_and_p3b_prefixes_and_are_npu2(self):
        with tempfile.TemporaryDirectory() as d:
            chain = pc.Chain(pc.FakeOps(), Path(d) / "c", "T", d, mode="p3")
            entries = chain.p3_job_entries()
        self.assertEqual({e["host"] for e in entries}, {"npu2"})
        self.assertEqual([e["id"] for e in entries],
                         ["prelim-chain-p3", "sweep-armB-prelim-P3", "sweep-armB-prelim-P3b-mintok4096"])
        self.assertIn("P3b-mintok4096", entries[2]["params"]["run_prefix"])

    def test_a_busy_arm_a_instance_is_never_stopped(self):
        class Busy(pc.FakeOps):
            def running_requests(self, url): return 3.0
        with tempfile.TemporaryDirectory() as d:
            real_sleep, pc_time = pc.time.sleep, pc.time.time
            ticks = iter(range(0, 10_000, 100))
            pc.time.sleep, pc.time.time = lambda _s: None, lambda: next(ticks)
            try:
                code, status, ops, _ = self.run_chain(d, ops=Busy())
            finally:
                pc.time.sleep, pc.time.time = real_sleep, pc_time
        self.assertEqual(code, 2)
        self.assertIn("refusing to stop it", status["reason"])
        self.assertFalse([c for c in ops.calls if c.startswith("docker stop")])
        self.assertFalse(self.launched(ops))


class DryRunCliTests(unittest.TestCase):
    def test_dry_run_p3_cli_runs_end_to_end_without_side_effects(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            done = subprocess.run([sys.executable, str(Path(pc.__file__)), "--dry-run", "--p3", "--out-root", d,
                                   "--kv-peak", "p3b-L96=0.95"], capture_output=True, text=True, timeout=120)
            self.assertEqual(done.returncode, 0, done.stderr[-500:])
            self.assertIn("CHAIN DONE", done.stdout)
            self.assertIn('"worst_case_admissible_level": 64', done.stdout)


if __name__ == "__main__":
    unittest.main()
