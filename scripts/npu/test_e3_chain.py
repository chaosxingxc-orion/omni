"""Offline tests for e3_chain.py (CPU only: no docker, no card, no network).

Run: python3 -m unittest -v test_e3_chain   (from the scripts directory)   or   python3 -m pytest -q test_e3_chain.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import e3_chain as e3  # noqa: E402
import prelim_chain as pc  # noqa: E402

HERE = Path(__file__).resolve().parent


def make(out, scenario=None, gate="/dry/p3-status.json", ops=None):
    out = Path(out)
    chain_dir = out / "prelim-E3-chain-T"
    chain_dir.mkdir(parents=True, exist_ok=True)
    baseline = e3.seed_baseline_dir(out / "D3")
    ops = ops or e3.FakeOps(scenario, out)
    sleeps = []
    chain = e3.E3Chain(ops, chain_dir, "T", str(out), e3.FAKE_CAPTURE, gate, baseline_dir=baseline,
                       gate_poll=120, sleep=sleeps.append)
    return chain, ops, chain_dir, sleeps


def run(scenario=None, out=None, **kw):
    out = Path(out or tempfile.mkdtemp(prefix="e3test-"))
    chain, ops, chain_dir, sleeps = make(out, scenario, **kw)
    code = chain.run()
    status = pc.read_json(chain_dir / "chain-status.json")
    return code, status, ops, chain_dir, sleeps


def shm_pass(outcome="completed", index=1, exits=()):
    return {"pass": index, "outcome": outcome, "engine_exits": [{"name": n} for n in exits]}


AGREE = {"identical_s2": 16, "identical_s3": 16, "n": 16, "complete": True, "agrees": True}


class VerdictTests(unittest.TestCase):
    def test_adopted_requires_zero_exits_and_16_of_16(self):
        v = e3.e3_verdict({"passes": [shm_pass(index=1), shm_pass(index=2)]}, AGREE)
        self.assertEqual(v["verdict"], "ADOPTED")
        self.assertEqual(v["reasons"], [])

    def test_exit_in_a_shm_pass_drops(self):
        for which in (1, 2):
            passes = [shm_pass(index=1), shm_pass(index=2)]
            passes[which - 1] = shm_pass("exit", which, ["saea-prelim-e3-shm-1"])
            v = e3.e3_verdict({"passes": passes[:which]}, AGREE)
            self.assertEqual(v["verdict"], "DROPPED")
            self.assertIn("engine exit", v["reasons"][0])

    def test_probe_disagreement_drops_even_with_no_exit(self):
        probe = dict(AGREE, identical_s2=15, agrees=False)
        v = e3.e3_verdict({"passes": [shm_pass(index=1), shm_pass(index=2)]}, probe)
        self.assertEqual(v["verdict"], "DROPPED")
        self.assertIn("15/16", v["reasons"][0])

    def test_probe_s3_disagreement_drops(self):
        probe = dict(AGREE, identical_s3=14, agrees=False)
        self.assertEqual(e3.e3_verdict({"passes": [shm_pass(index=1), shm_pass(index=2)]}, probe)["verdict"], "DROPPED")

    def test_startup_failure_drops(self):
        v = e3.e3_verdict({"startup_failed": True, "startup_reason": "exited during startup", "passes": []}, None)
        self.assertEqual(v["verdict"], "DROPPED")

    def test_missing_evidence_is_incomplete_not_adopted(self):
        self.assertEqual(e3.e3_verdict(None, None)["verdict"], "INCOMPLETE")
        self.assertEqual(e3.e3_verdict({"passes": [shm_pass(index=1), shm_pass(index=2)]}, None)["verdict"], "INCOMPLETE")
        self.assertEqual(e3.e3_verdict({"passes": [shm_pass(index=1)]}, AGREE)["verdict"], "INCOMPLETE")
        self.assertEqual(e3.e3_verdict({"passes": [shm_pass(index=1), shm_pass(index=2)]},
                                       dict(AGREE, complete=False, agrees=False))["verdict"], "INCOMPLETE")

    def test_preemptions_alone_do_not_decide(self):
        p = shm_pass(index=1)
        p["preemptions_total"] = 40
        self.assertEqual(e3.e3_verdict({"passes": [p, shm_pass(index=2)]}, AGREE)["verdict"], "ADOPTED")

    def test_probe_agreement_needs_all_16_compared(self):
        full = {"byte_identical": 16, "requests": 16, "count_a": 16}
        self.assertTrue(e3.probe_agreement(full, full)["agrees"])
        self.assertFalse(e3.probe_agreement(dict(full, byte_identical=15), full)["agrees"])
        self.assertFalse(e3.probe_agreement(dict(full, requests=15, count_a=15), full)["agrees"])


class PureTests(unittest.TestCase):
    def test_engine_exits_from_container_and_log(self):
        names = [f"n{i}" for i in range(3)]
        out = e3.engine_exits(names, {"n0", "n1"}, {"n0": {"fatal_engine_error": True}, "n1": {}, "n2": {}})
        self.assertEqual({e["name"] for e in out}, {"n0", "n2"})
        self.assertIn("container_not_running", next(e for e in out if e["name"] == "n2")["reasons"])
        self.assertIn("fatal_engine_error_in_log", next(e for e in out if e["name"] == "n0")["reasons"])

    def test_pass_record_died_has_per_instance_preemptions(self):
        names = [f"n{i}" for i in range(4)]
        died = pc.EndpointDied("n1", {"dead_instances": ["n1"], "preemptions_first_polled": [0.0, 1.0, None, 3.0],
                                      "preemptions_last_polled": [4.0, 9.0, 2.0, 3.0], "kv_peak_polled": [1, 1, .8, .7]})
        row = e3.pass_record("shm", 1, "20261028", names, None, died, {"n0", "n2", "n3"}, {})
        self.assertEqual(row["outcome"], "exit")
        self.assertEqual([e["preemptions_delta"] for e in row["instances"]], [4.0, 8.0, None, 0.0])
        self.assertEqual([e["alive"] for e in row["instances"]], [True, False, True, True])
        self.assertEqual([e["name"] for e in row["engine_exits"]], ["n1"])

    def test_newest_gate_is_the_newest_p3_dir(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(e3.newest_gate(d))
            for n in ("prelim-chain-p3-20261029T150000Z", "prelim-chain-p3-20261029T170000Z", "prelim-chain-p2b-29999999"):
                (Path(d) / n).mkdir()
            self.assertTrue(e3.newest_gate(d).endswith("prelim-chain-p3-20261029T170000Z/chain-status.json"))

    def test_gate_final_needs_state_and_pid_gone(self):
        self.assertEqual(e3.gate_final({"state": "done", "pid": 1}, False)[0], True)
        self.assertEqual(e3.gate_final({"state": "done", "pid": 1}, True)[0], False)
        self.assertEqual(e3.gate_final({"state": "running", "pid": 1}, False)[0], False)
        self.assertEqual(e3.gate_final(None, False)[0], False)
        for state in ("failed", "interrupted"):
            self.assertTrue(e3.gate_final({"state": state, "pid": 1}, False)[0])


class GateTests(unittest.TestCase):
    def test_no_card_work_before_the_gate_is_final(self):
        code, status, ops, _, sleeps = run({"gate_waits": 3, "gate_pid_extra": 2})
        self.assertEqual(code, 0)
        self.assertGreaterEqual(ops.gate_reads, 6)  # 3 running reads + 2 final-but-pid-alive reads + the final read
        first = next(i for i, c in enumerate(ops.calls) if "remove=['vllm-arm-a-0'" in c)  # first side effect
        self.assertEqual(ops.calls[:first], ["jobs upsert=['prelim-E3-chain'] remove=[]", "sampler start"])
        self.assertEqual(ops.gate_reads, 3 + 2 + 1)  # 3 running, 2 final-but-pid-alive, then the final read
        self.assertTrue(all(s == 120 for s in sleeps))  # the status file is read at most every 2 minutes

    def test_no_p3_directory_stops_without_touching_anything(self):
        code, status, ops, _, sleeps = run({"gate_missing": True}, gate=None)
        self.assertEqual(code, 2)
        self.assertEqual(status["state"], "failed")
        self.assertIn("nothing was touched", status["reason"])
        self.assertFalse(any(c.startswith(("docker stop", "cards", "launch", "measure")) for c in ops.calls))
        self.assertEqual(ops.running, {f"saea-arm-a-{i}" for i in range(4)})
        self.assertGreaterEqual(sum(sleeps), e3.GATE_APPEAR_TIMEOUT_S)

    def test_failed_gate_chain_is_final(self):
        code, status, ops, _, _ = run({"gate_state": "failed"})
        self.assertEqual(code, 0)
        self.assertEqual(status["steps"]["gate"]["state"], "done")

    def test_arm_b_still_up_is_refused_before_any_stop(self):
        code, status, ops, _, _ = run({"arm_b_up": True})
        self.assertEqual(code, 2)
        self.assertIn("arm B", status["reason"])
        self.assertFalse(any(c.startswith("docker stop") for c in ops.calls))
        self.assertFalse(any(c.startswith("launch") for c in ops.calls))  # and no restore either

    def test_busy_arm_a_instance_is_not_stopped(self):
        code, status, ops, _, _ = run({"busy_arm_a": 3}, )
        self.assertEqual(code, 2)
        self.assertIn("refusing to stop", status["reason"])
        self.assertFalse(any(c.startswith("docker stop") for c in ops.calls))


class FlowTests(unittest.TestCase):
    def test_full_run_adopted_and_arm_a_restored(self):
        code, status, ops, chain_dir, _ = run()
        self.assertEqual(code, 0)
        self.assertEqual(status["state"], "done")
        dec = status["decisions"]["E3"]
        self.assertEqual(dec["verdict"], "ADOPTED")
        self.assertEqual(ops.running, {f"saea-arm-a-{i}" for i in range(4)})
        # baseline launched with empty EXTRA, shm with the flag, arm A with the original launcher
        launches = [c for c in ops.calls if c.startswith("launch_e")]
        self.assertEqual(len(launches), 8)
        self.assertTrue(all(c.endswith("extra ''") for c in launches[:4]))
        self.assertTrue(all(c.endswith("extra '--mm-processor-cache-type shm'") for c in launches[4:]))
        self.assertEqual(len([c for c in ops.calls if c.startswith("launch arm A")]), 4)
        # worst case 48, seeds, min_tokens
        self.assertEqual(ops.measured, [("base-pass1", 48, "20261028", 4096), ("shm-pass1", 48, "20261028", 4096),
                                        ("shm-pass2", 48, "20261029", 4096)])
        # the watchdog registry ends with arm A entries and no E3 instance entries
        self.assertTrue({f"vllm-arm-a-{i}" for i in range(4)} <= ops.jobs)
        self.assertFalse([j for j in ops.jobs if j.startswith("vllm-prelim-E3")])
        for sweep in ("sweep-prelim-E3-base-pass1", "sweep-prelim-E3-shm-pass1", "sweep-prelim-E3-shm-pass2",
                      "prelim-E3-chain"):
            self.assertIn(sweep, ops.jobs)
        result = pc.read_json(dec["result_file"])
        self.assertEqual(result["label"], "prelim-E3")
        self.assertEqual(len(result["shm"]["passes"]), 2)
        self.assertEqual(result["baseline"]["engine_exit_count"], 0)

    def test_probe_runs_before_any_shm_load(self):
        _, _, ops, _, _ = run()
        calls = ops.calls
        probe = [i for i, c in enumerate(calls) if c.startswith("probe send")]
        shm_load = next(i for i, c in enumerate(calls) if c.startswith("measure shm-pass1"))
        self.assertEqual(len(probe), 3)
        self.assertLess(max(probe), shm_load)

    def test_probe_mismatch_drops_with_reason_and_still_restores(self):
        code, status, ops, _, _ = run({"probe_mismatch": {3}})
        dec = status["decisions"]["E3"]
        self.assertEqual(code, 0)  # an experiment that finished with a DROPPED verdict is a completed chain
        self.assertEqual(dec["verdict"], "DROPPED")
        self.assertIn("warm probe disagrees", dec["reasons"][0])
        self.assertEqual(dec["shm"]["probe"]["identical_s2"], 15)
        self.assertEqual(ops.running, {f"saea-arm-a-{i}" for i in range(4)})

    def test_dead_instance_in_shm_pass_one_ends_e3_dropped(self):
        code, status, ops, _, _ = run({"shm_exit": 1})
        dec = status["decisions"]["E3"]
        self.assertEqual(dec["verdict"], "DROPPED")
        self.assertEqual([p["outcome"] for p in dec["shm"]["passes"]], ["exit"])  # no second pass
        self.assertEqual([m[0] for m in ops.measured], ["base-pass1", "shm-pass1"])
        self.assertEqual(ops.running, {f"saea-arm-a-{i}" for i in range(4)})
        self.assertEqual(status["state"], "done")

    def test_dead_instance_in_shm_pass_two_is_dropped(self):
        _, status, ops, _, _ = run({"shm_exit": 2})
        dec = status["decisions"]["E3"]
        self.assertEqual(dec["verdict"], "DROPPED")
        self.assertEqual([p["outcome"] for p in dec["shm"]["passes"]], ["completed", "exit"])
        exits = dec["shm"]["passes"][1]["engine_exits"]
        self.assertEqual([e["name"] for e in exits], ["saea-prelim-e3-shm-1"])
        self.assertTrue(exits[0]["k2_confirmed"])

    def test_dead_instance_in_baseline_is_informative_and_shm_still_runs(self):
        code, status, ops, _, _ = run({"base_exit": 1})
        dec = status["decisions"]["E3"]
        self.assertEqual(dec["verdict"], "ADOPTED")
        self.assertEqual(dec["baseline"]["engine_exit_count"], 1)
        self.assertEqual(dec["baseline"]["passes"][0]["outcome"], "exit")
        self.assertEqual([m[0] for m in ops.measured], ["base-pass1", "shm-pass1", "shm-pass2"])

    def test_shm_startup_failure_is_dropped(self):
        code, status, ops, _, _ = run({"shm_start_fail": True})
        dec = status["decisions"]["E3"]
        self.assertEqual(dec["verdict"], "DROPPED")
        self.assertTrue(dec["shm"]["startup_failed"])
        self.assertEqual([m[0] for m in ops.measured], ["base-pass1"])
        self.assertEqual(ops.running, {f"saea-arm-a-{i}" for i in range(4)})

    def test_baseline_startup_failure_does_not_block_shm(self):
        _, status, ops, _, _ = run({"base_start_fail": True})
        dec = status["decisions"]["E3"]
        self.assertTrue(dec["baseline"]["startup_failed"])
        self.assertEqual(dec["verdict"], "ADOPTED")

    def test_unexpected_error_restores_arm_a(self):
        out = Path(tempfile.mkdtemp(prefix="e3test-"))
        ops = e3.FakeOps({}, out)

        def boom(*a, **k):
            raise RuntimeError("scripted")
        ops.probe_send = boom
        chain, ops, chain_dir, _ = make(out, ops=ops)
        code = chain.run()
        status = pc.read_json(chain_dir / "chain-status.json")
        self.assertEqual(code, 4)
        self.assertEqual(status["state"], "failed")
        self.assertEqual(ops.running, {f"saea-arm-a-{i}" for i in range(4)})  # restored by the failure path


class ResumeTests(unittest.TestCase):
    def test_resume_skips_finished_steps_and_relaunches_nothing(self):
        out = Path(tempfile.mkdtemp(prefix="e3test-"))
        chain, ops, chain_dir, _ = make(out)
        self.assertEqual(chain.run(), 0)
        again, ops2, _, _ = make(out)
        ops2.running = {f"saea-arm-a-{i}" for i in range(4)}
        self.assertEqual(again.run(), 0)
        self.assertEqual(ops2.measured, [])
        self.assertFalse([c for c in ops2.calls if c.startswith(("launch", "docker stop", "probe"))])
        self.assertEqual(pc.read_json(chain_dir / "chain-status.json")["decisions"]["E3"]["verdict"], "ADOPTED")

    def test_resume_after_interruption_between_passes(self):
        out = Path(tempfile.mkdtemp(prefix="e3test-"))
        ops = e3.FakeOps({}, out)
        original = ops.measure_level

        def interrupt(step, *a, **k):
            if step == "shm-pass2":
                raise KeyboardInterrupt()
            return original(step, *a, **k)
        ops.measure_level = interrupt
        chain, ops, chain_dir, _ = make(out, ops=ops)
        self.assertEqual(chain.run(), 3)
        self.assertEqual(pc.read_json(chain_dir / "chain-status.json")["state"], "interrupted")
        self.assertNotEqual(ops.running & {f"saea-prelim-e3-shm-{i}" for i in range(4)}, set())  # left running
        again, ops2, _, _ = make(out)
        ops2.running = set(ops.running)
        ops2.extra = dict(ops.extra)
        self.assertEqual(again.run(), 0)
        self.assertEqual([m[0] for m in ops2.measured], ["shm-pass2"])  # base-pass1 and shm-pass1 were not repeated
        self.assertFalse([c for c in ops2.calls if c.startswith(("launch_e", "probe"))])
        self.assertEqual(ops2.running, {f"saea-arm-a-{i}" for i in range(4)})
        verdict = pc.read_json(chain_dir / "chain-status.json")["decisions"]["E3"]
        self.assertEqual(verdict["verdict"], "ADOPTED")
        self.assertEqual(len(verdict["shm"]["passes"]), 2)

    def test_resume_with_a_dead_instance_records_an_exit(self):
        out = Path(tempfile.mkdtemp(prefix="e3test-"))
        ops = e3.FakeOps({}, out)
        original = ops.measure_level

        def interrupt(step, *a, **k):
            if step == "shm-pass2":
                raise KeyboardInterrupt()
            return original(step, *a, **k)
        ops.measure_level = interrupt
        chain, ops, chain_dir, _ = make(out, ops=ops)
        chain.run()
        again, ops2, _, _ = make(out)
        ops2.running = set(ops.running) - {"saea-prelim-e3-shm-2"}  # the instance died while no chain was watching
        ops2.extra = dict(ops.extra)
        ops2.fatal = {"saea-prelim-e3-shm-2"}
        again.run()
        dec = pc.read_json(chain_dir / "chain-status.json")["decisions"]["E3"]
        self.assertEqual(dec["verdict"], "DROPPED")
        self.assertEqual(dec["shm"]["passes"][1]["outcome"], "exit")
        self.assertEqual(ops2.measured, [])  # no load was sent to a group with a dead instance


class LauncherTests(unittest.TestCase):
    def test_real_class_guards_container_names(self):
        for ok in ("saea-arm-a-0", "saea-arm-a-3", "saea-prelim-e3-base-2", "saea-prelim-e3-shm-0"):
            e3.E3Real._check_ours([ok])
        for bad in ("saea-arm-b-0", "saea-prelim-e1", "rl_ascend_20261026", "saea-prelim-e3-shm-9x", "saea-arm-a-0; rm"):
            with self.assertRaises(pc.ChainStop):
                e3.E3Real._check_ours([bad])

    def test_launcher_print_only_baseline_and_shm_argv(self):
        import os
        import subprocess
        base = dict(os.environ, E_PRINT_ONLY="1", NAME="saea-prelim-e3-base-0", CPUSET="192-223", HCCL_BASE_PORT="60000",
                    MAX_NUM_SEQS="64", EXTRA_ARGS="")
        shm = dict(base, NAME="saea-prelim-e3-shm-0", EXTRA_ARGS=e3.SHM_ARGS)
        out = []
        for env in (base, shm):
            done = subprocess.run(["bash", str(HERE / "npu_serve_e.sh"), "thinking", "0,1", "8200", "2"], env=env,
                                  capture_output=True, text=True, timeout=60)
            self.assertEqual(done.returncode, 0, done.stderr)
            out.append([l[len("E_PRINT_ARG "):] for l in done.stdout.splitlines() if l.startswith("E_PRINT_ARG ")])
        capture = pc.read_json(e3.DEFAULT_CAPTURE)
        if capture:  # the baseline argv equals the captured arm A argv (without the leading name and the model flag)
            cmd = capture["instances"][0]["cmd"]
            self.assertEqual(out[0][-len(cmd[1:]):][:3], cmd[1:][:3]) if len(out[0]) >= len(cmd) - 1 else None
        self.assertEqual(out[1][:len(out[0])], out[0])
        self.assertEqual(out[1][len(out[0]):], ["--mm-processor-cache-type", "shm"])


if __name__ == "__main__":
    unittest.main()
