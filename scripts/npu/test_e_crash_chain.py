"""Offline tests for e_crash_chain.py, npu_serve_e.sh and the synthetic-contact audio reuse (CPU only: no docker, no
card, no network).

Run: python3 -m unittest -v test_e_crash_chain   (from the scripts directory)   or   python3 -m pytest -q test_e_crash_chain.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import e_crash_chain as e  # noqa: E402
import prelim_chain as pc  # noqa: E402

HERE = Path(__file__).resolve().parent
CAPTURE = Path(e.DEFAULT_CAPTURE)
REAL_LOG = Path(e.LOGS) / "saea-arm-a-0-20261029T081859.log"


def run_dry(scenario=None, out=None, gate_poll=0.0):
    out = Path(out or tempfile.mkdtemp(prefix="edry-"))
    chain_dir = out / "prelim-E-chain-T"
    chain_dir.mkdir(parents=True, exist_ok=True)
    ops = e.FakeOps(scenario)
    chain = e.EChain(ops, chain_dir, "T", str(out), e.FAKE_CAPTURE, "/dry/gate.json", gate_poll=gate_poll,
                     sleep=lambda s: None)
    code = chain.run()
    return code, ops, chain, out, chain_dir


class PureLogicTests(unittest.TestCase):
    def test_gate_needs_final_state_and_dead_pid(self):
        self.assertFalse(e.gate_final({"state": "running", "pid": 1}, True)[0])
        self.assertFalse(e.gate_final({"state": "done", "pid": 1}, True)[0])
        for state in ("done", "failed", "interrupted"):
            self.assertTrue(e.gate_final({"state": state, "pid": 1}, False)[0])
        self.assertFalse(e.gate_final(None, False)[0])

    def test_seeds_and_levels(self):
        self.assertEqual([e.pass_seed(i) for i in (1, 2, 3)], ["20261028", "20261029", "20261030"])
        self.assertEqual(e.next_level(e.E1_PLAN, 1, {"outcome": "completed", "preemptions": 2.0, "level": 48}, False), 56)
        self.assertEqual(e.next_level(e.E2_PLAN, 1, {"outcome": "completed", "preemptions": 12.0, "level": 56}), 56)
        self.assertEqual(e.next_level(e.E2_PLAN, 1, {"outcome": "completed", "preemptions": 3.0, "level": 56}), 64)
        self.assertEqual(e.next_level(e.E2_PLAN, 2, {"outcome": "completed", "preemptions": 3.0, "level": 64}), 64)
        self.assertEqual(e.next_level(e.E2_PLAN, 0, None), 56)

    def test_e1_verdicts(self):
        ok = lambda n, p: {"pass": n, "level": 56, "outcome": "completed", "preemptions": p}
        self.assertEqual(e.e1_verdict([ok(1, 12), ok(2, 15), ok(3, 10)])["verdict"], "not reproduced in isolation")
        self.assertEqual(e.e1_verdict([ok(1, 12), ok(2, 9), ok(3, 10)])["verdict"], "inconclusive_low_preemptions")
        self.assertEqual(e.e1_verdict([ok(1, 12)])["verdict"], "incomplete")
        ev = {"mm_embeddings_assertion": True, "execute_mm_encoder": True, "died_request_ids": ["x-k2-1"], "k2_confirmed": True}
        v = e.e1_verdict([ok(1, 12), {"pass": 2, "level": 56, "outcome": "exit", "evidence": ev}])
        self.assertEqual((v["verdict"], v["pass"], v["k2_confirmed"]), ("reproduced", 2, True))
        other = e.e1_verdict([{"pass": 1, "level": 48, "outcome": "exit", "evidence": {"mm_embeddings_assertion": False}}])
        self.assertEqual(other["verdict"], "exit_other_cause")

    def test_e2_verdicts(self):
        ok = lambda n, p: {"pass": n, "level": 56, "outcome": "completed", "preemptions": p}
        self.assertEqual(e.e2_verdict([ok(1, 10), ok(2, 11), ok(3, 40)])["verdict"], "accepted")
        self.assertEqual(e.e2_verdict([ok(1, 10), ok(2, 9), ok(3, 40)])["verdict"], "inconclusive_low_preemptions")
        self.assertEqual(e.e2_verdict([ok(1, 10), {"pass": 2, "level": 56, "outcome": "exit"}])["verdict"], "failed_exit")
        self.assertEqual(e.e2_verdict([ok(1, None), ok(2, 11), ok(3, 40)])["verdict"], "inconclusive_low_preemptions")

    def test_eq_compare(self):
        a = [{"status": "ok", "reasoning": f"r{i}", "content": "", "finish_reason": "length", "completion_tokens": 64,
              "logprobs": [-0.5, -1.0]} for i in range(16)]
        b = [dict(x) for x in a]
        self.assertEqual(e.compare_eq(a, b)["byte_identical"], 16)
        b[4]["reasoning"] = "other"
        b[7] = dict(b[7], status="error")
        cmp_ = e.compare_eq(a, b)
        self.assertEqual((cmp_["byte_identical"], cmp_["mismatched_indices"]), (14, [4, 7]))
        b[9] = dict(b[9], logprobs=[-0.5, -1.25])
        self.assertAlmostEqual(e.compare_eq(a, b)["max_abs_logprob_diff"], 0.25)

    def test_crash_evidence_synthetic_lines(self):
        lines = [
            "(Worker_TP1 pid=106) ERROR 10-29 10:21:11 [x] File \"gpu_model_runner.py\", line 3187, in _execute_mm_encoder",
            "(Worker_TP1 pid=106) ERROR [x] AssertionError: Expected number ... but got len(mm_embeddings)=0 instead.",
            "(EngineCore pid=86) ERROR [dump_input.py:79] Dumping scheduler output: SchedulerOutput(resumed_req_ids="
            "{'chatcmpl-sweep-c48-w140-t140-k2-8db6fe48', 'chatcmpl-sweep-c48-w188-t188-k2-96e10751'}, "
            "scheduled_encoder_inputs={chatcmpl-sweep-c48-w188-t188-k2-96e10751: [0]}, x=1)",
            "(EngineCore pid=86) ERROR [core.py:1368] EngineCore encountered a fatal error.",
        ]
        ev = e.crash_evidence_from_lines(lines)
        self.assertTrue(ev["mm_embeddings_assertion"] and ev["execute_mm_encoder"] and ev["fatal_engine_error"])
        self.assertEqual(ev["died_request_ids"], ["chatcmpl-sweep-c48-w188-t188-k2-96e10751"])
        self.assertEqual(len(ev["resumed_req_ids"]), 2)
        self.assertTrue(ev["k2_confirmed"])
        none = e.crash_evidence_from_lines(["INFO all fine"])
        self.assertFalse(none["mm_embeddings_assertion"] or none["k2_confirmed"])

    @unittest.skipUnless(REAL_LOG.exists(), "real arm A log not available")
    def test_crash_evidence_on_the_real_arm_a_log(self):
        ev = e.crash_evidence_from_lines(REAL_LOG.read_text(errors="replace").split("\n"))
        self.assertTrue(ev["mm_embeddings_assertion"] and ev["execute_mm_encoder"])
        self.assertEqual(ev["died_request_ids"], ["chatcmpl-sweep-c48-w188-t188-k2-96e10751"])
        self.assertTrue(ev["k2_confirmed"])


class SyntheticContactReuseTests(unittest.TestCase):
    """The crash needs contact k2 to REUSE contact 0's audio (and its evidence prefix). Confirm the sweep does."""

    @unittest.skipUnless(Path(e.TREE, "scripts/serving/synthetic_sweep.py").exists(), "study tree not available")
    def test_contacts_of_a_trajectory_share_audio_and_extend_the_prefix(self):
        ss = e.load_sweep_module()
        config = ss.SweepConfig(base_urls=["http://x"], model="m", levels=[2], min_tokens=4096)
        workers, trajectories = ss.build_level(config, 2)
        self.assertEqual((workers, len(trajectories)), (2, 2))
        trajectory = trajectories[0]
        self.assertEqual(len(trajectory.targets), 3)  # k0, k1, k2
        audio = trajectory.audio_b64()  # computed ONCE per trajectory, exactly as run_level's worker does
        words = trajectory.words(300)
        bodies = [ss.contact_messages(audio, words, n, k, 3, 0) for k, n in enumerate((100, 200, 300))]
        audios = [b[1]["content"][0]["input_audio"]["data"] for b in bodies]
        self.assertEqual(len(set(audios)), 1)  # same clip in every contact
        notes = [b[1]["content"][1]["text"] for b in bodies]
        self.assertTrue(notes[1].startswith(notes[0]) and notes[2].startswith(notes[1]))  # strict extensions
        other = trajectories[1].audio_b64()
        self.assertNotEqual(other, audio)  # different trajectories have different audio
        self.assertEqual(ss._decoding(config)["min_tokens"], 4096)

    @unittest.skipUnless(Path(e.TREE, "scripts/serving/synthetic_sweep.py").exists(), "study tree not available")
    def test_request_ids_carry_the_contact_index_used_to_confirm_the_mechanism(self):
        text = Path(e.TREE, "scripts/serving/synthetic_sweep.py").read_text()
        self.assertIn('f"sweep-c{level}-w{w}-t{trajectory.index}-k{contact}"', text)
        m = e.REQ_ID.search("chatcmpl-sweep-c56-w188-t188-k2-96e10751")
        self.assertEqual((m.group(1), m.group(4)), ("56", "2"))

    @unittest.skipUnless(Path(e.TREE, "scripts/serving/synthetic_sweep.py").exists(), "study tree not available")
    def test_eq_bodies_are_fixed_and_deterministic(self):
        ss = e.load_sweep_module()
        a, b = e.eq_bodies(ss, "m"), e.eq_bodies(ss, "m")
        self.assertEqual(len(a), 16)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        self.assertTrue(all(x["temperature"] == 0 and x["seed"] == e.EQ_DECODING_SEED and x["max_tokens"] <= 64 for x in a))
        self.assertEqual(a[0]["messages"][1]["content"][0], a[1]["messages"][1]["content"][0])  # contact 1 reuses audio


class LauncherCopyTests(unittest.TestCase):
    def printed(self, extra, instance):
        spec = json.loads(CAPTURE.read_text())["instances"][instance]
        env = dict(os.environ, E_PRINT_ONLY="1", NAME=spec["name"], CPUSET=spec["cpuset"],
                   HCCL_BASE_PORT=str(spec["hccl_base_port"]), MAX_NUM_SEQS=str(spec["max_num_seqs"]), EXTRA_ARGS=extra)
        out = subprocess.run(["bash", str(HERE / "npu_serve_e.sh"), "thinking", spec["cards"], str(spec["port"]),
                              str(spec["tp"])], env=env, capture_output=True, text=True, check=True).stdout
        args = [l[len("E_PRINT_ARG "):] for l in out.splitlines() if l.startswith("E_PRINT_ARG ")]
        return spec, args, out

    @unittest.skipUnless(CAPTURE.exists(), "arm A capture not available")
    def test_copy_without_extra_reproduces_the_captured_argv_of_instances_0_and_3(self):
        for instance in (0, 3):
            spec, args, out = self.printed("", instance)
            self.assertEqual(args, spec["cmd"][1:])  # cmd[0] is "serve"
            self.assertIn(f"cpuset={spec['cpuset']}", out)
            self.assertIn(f"hccl={spec['hccl_base_port']}", out)

    @unittest.skipUnless(CAPTURE.exists(), "arm A capture not available")
    def test_extra_args_are_appended_as_separate_words_at_the_end(self):
        spec, args, _ = self.printed("--mm-processor-cache-type shm", 3)
        self.assertEqual(args, spec["cmd"][1:] + ["--mm-processor-cache-type", "shm"])
        spec, args, _ = self.printed("--no-enable-prefix-caching", 0)
        self.assertEqual(args, spec["cmd"][1:] + ["--no-enable-prefix-caching"])

    def test_original_launcher_is_untouched_and_the_copy_differs_only_by_the_extra_hooks(self):
        # The copy was cut from the pre-S10 launcher; after the S10 swap that file is kept as npu_serve.sh.pre-s10-20261030.
        pre_s10 = HERE / "npu_serve.sh.pre-s10-20261030"
        original = (pre_s10 if pre_s10.exists() else HERE / "npu_serve.sh").read_text().splitlines()
        copy = (HERE / "npu_serve_e.sh").read_text().splitlines()
        removed = [l for l in original if l not in copy]
        self.assertEqual(removed, ["    refuse_busy \"$CARDS\"", "      --seed 0"])  # only these two lines are replaced
        self.assertNotIn("EXTRA_ARGS", (HERE / "npu_serve.sh").read_text())
        # The serving launcher (the sidecar until the swap, npu_serve.sh after it) passes --gpu-memory-utilization.
        sidecar = HERE / "npu_serve.sh.s10-next"
        launcher = (sidecar if sidecar.exists() else HERE / "npu_serve.sh").read_text()
        self.assertIn('--gpu-memory-utilization "${GPU_MEM_UTIL:-0.92}"', launcher)

    def test_container_name_guard(self):
        ops = e.EReal.__new__(e.EReal)
        e.EReal._check_ours(["saea-prelim-e1", "saea-prelim-e2-shm", "saea-prelim-e2-noprefix", "saea-arm-a-0", "saea-arm-a-3"])
        for bad in ("saea-arm-a-1", "saea-arm-a-2", "saea-arm-b-0", "elated_jemison", "suite-wave1-work-0"):
            with self.assertRaises(pc.ChainStop):
                e.EReal._check_ours([bad])


class DryRunTests(unittest.TestCase):
    def decisions(self, chain_dir):
        return json.loads((chain_dir / "chain-status.json").read_text())

    def test_default_run_not_reproduced_and_shm_accepted_and_arm_a_restored(self):
        code, ops, chain, out, cd = run_dry()
        self.assertEqual(code, 0)
        st = self.decisions(cd)
        self.assertEqual(st["decisions"]["E1"]["base"]["verdict"], "not reproduced in isolation")
        self.assertEqual(st["decisions"]["E2"]["shm"]["verdict"], "accepted")
        self.assertIn("saea-arm-a-0", ops.running)
        self.assertIn("saea-arm-a-3", ops.running)
        self.assertTrue({"saea-arm-a-1", "saea-arm-a-2"} <= ops.running)  # never stopped
        self.assertNotIn("saea-prelim-e1", ops.running)
        self.assertNotIn("saea-prelim-e2-shm", ops.running)
        self.assertFalse(any("arm-a-1" in c or "arm-a-2" in c for c in ops.calls if "stop" in c or "launch" in c))
        levels = [c for c in ops.calls if c.startswith("measure ")]
        self.assertEqual([int(l.split()[3]) for l in levels if "E1" in l], [48, 56, 56])
        self.assertEqual([l.split()[5] for l in levels if "E1" in l], ["20261028", "20261029", "20261030"])
        compare = json.loads(next(out.glob("prelim-E2-eq-compare-shm-*.json")).read_text())
        self.assertEqual(compare["e1_vs_e2"]["byte_identical"], 16)
        self.assertEqual(compare["e1_self_determinism"]["byte_identical"], 16)

    def test_no_card_work_before_the_gate_opens(self):
        code, ops, chain, out, cd = run_dry({"gate_waits": 4})
        self.assertEqual(code, 0)
        card_calls = [c for c in ops.calls_gate if c[2].startswith(("cards ", "docker stop", "launch", "measure", "sampler start"))]
        self.assertTrue(card_calls)
        self.assertTrue(all(reads > waits for reads, waits, _ in card_calls), card_calls[:3])
        self.assertGreaterEqual(ops.gate_reads, 5)

    def test_gate_stays_closed_while_the_p2b_pid_is_alive(self):
        code, ops, chain, out, cd = run_dry({"gate_waits": 1, "gate_pid_extra": 3})
        self.assertEqual(code, 0)
        self.assertGreaterEqual(ops.gate_reads, 5)

    def test_e1_exit_is_reproduced_and_ends_e1(self):
        code, ops, chain, out, cd = run_dry({"e1_exit_pass": 2})
        st = self.decisions(cd)
        v = st["decisions"]["E1"]["base"]
        self.assertEqual((v["verdict"], v["pass"], v["k2_confirmed"]), ("reproduced", 2, True))
        self.assertEqual(len([c for c in ops.calls if c.startswith("measure ") and "E1" in c]), 2)
        self.assertIn("saea-arm-a-0", ops.running)  # restored even though the E instance died
        evidence = list(out.glob("prelim-E1-base-pass2-crash-evidence-*.json"))
        self.assertEqual(len(evidence), 1)

    def test_shm_start_failure_runs_the_diagnostic_arm_and_labels_it(self):
        code, ops, chain, out, cd = run_dry({"e2_start_fail": True})
        st = self.decisions(cd)["decisions"]["E2"]
        self.assertEqual(st["shm"]["verdict"], "startup_failed")
        self.assertTrue(st["noprefix"]["verdict"].startswith("DIAGNOSTIC ONLY"))
        self.assertTrue(any("extra '--no-enable-prefix-caching'" in c for c in ops.calls))
        self.assertEqual(len([c for c in ops.calls if c.startswith("measure ") and "noprefix" in c]), 1)

    def test_shm_exit_runs_the_diagnostic_arm(self):
        code, ops, chain, out, cd = run_dry({"e2_exit_pass": 2})
        st = self.decisions(cd)["decisions"]["E2"]
        self.assertEqual(st["shm"]["verdict"], "failed_exit")
        self.assertIn("DIAGNOSTIC ONLY", st["noprefix"]["verdict"])

    def test_low_preemptions_make_the_verdict_inconclusive_and_raise_the_next_level(self):
        code, ops, chain, out, cd = run_dry({"e2_low": {1}})
        st = self.decisions(cd)["decisions"]["E2"]
        self.assertEqual(st["shm"]["verdict"], "inconclusive_low_preemptions")
        levels = [int(c.split()[3]) for c in ops.calls if c.startswith("measure ") and "E2-shm" in c]
        self.assertEqual(levels, [56, 64, 56])

    def test_eq_mismatch_is_counted(self):
        code, ops, chain, out, cd = run_dry({"eq_mismatch": [3, 5]})
        compare = json.loads(next(out.glob("prelim-E2-eq-compare-shm-*.json")).read_text())
        self.assertEqual(compare["e1_vs_e2"]["byte_identical"], 14)

    def test_resume_skips_everything_finished(self):
        out = Path(tempfile.mkdtemp(prefix="edry-resume-"))
        run_dry(out=out)
        code, ops, chain, _, cd = run_dry(out=out)
        self.assertEqual(code, 0)
        self.assertFalse([c for c in ops.calls if c.startswith(("launch", "measure", "docker stop"))], ops.calls)

    def test_registry_edits_remove_the_stale_entries_and_add_the_chain(self):
        code, ops, chain, out, cd = run_dry()
        first = ops.calls[0]
        self.assertIn("prelim-E-chain", first)
        self.assertIn("prelim-chain", first)
        self.assertIn("sweep-armA-prelim-P2-mintok4096", first)


if __name__ == "__main__":
    unittest.main()
