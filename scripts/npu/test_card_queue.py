#!/usr/bin/env python3
"""Offline tests for card_queue.py (CPU only; a fake job script; npu-smi is never called: the card probe is injected, and the
CLI tests put a tripwire `npu-smi` first on PATH).

Run: python3 -m unittest -v test_card_queue   (from the scripts directory)   or   python3 -m pytest -q test_card_queue.py
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import card_queue as cq  # noqa: E402

FAKE_JOB = r'''#!/usr/bin/env python3
"""fake job: fake_job.py <trace> <name> <sleep_s> <exit> <lines...>   prints each line, sleeps, exits."""
import sys, time
trace, name, sleep_s, code = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
with open(trace, "a") as f:
    f.write("%s start %.3f\n" % (name, time.time()))
print("job %s begins" % name, flush=True)
for line in sys.argv[5:]:
    print(line, flush=True)
time.sleep(sleep_s)
with open(trace, "a") as f:
    f.write("%s end %.3f\n" % (name, time.time()))
sys.exit(code)
'''

SMI = """\
| NPU   Name                | Health        | Power(W)             Temp(C)                 Hugepages-Usage(page)   |
| Chip                      | Bus-Id        | AICore(%%)            Memory-Usage(MB)        HBM-Usage(MB)           |
| 0     910B3               | OK            | 115.9                44                      0    / 0                |
| 0                         | 0000:C1:00.0  | 11                   0    / 0                %d / 65536            |
| 1     910B3               | OK            | 97.1                 31                      0    / 0                |
| 0                         | 0000:C2:00.0  | 0                    0    / 0                %d / 65536            |
| NPU     Chip              | Process id    | Process name       | Process memory(MB)    | Process id in container |
%s"""


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="cq-test-")
        self.qdir = os.path.join(self.d, "q")
        os.makedirs(self.qdir)
        self.script = os.path.join(self.d, "fake_job.py")
        Path(self.script).write_text(FAKE_JOB)
        self.trace = os.path.join(self.d, "trace.txt")
        self.queue_file = os.path.join(self.d, "queue.json")

    def tearDown(self):
        # no job may outlive a test
        st = cq.read_json(os.path.join(self.qdir, "state.json")) or {}
        for rec in st.get("jobs", {}).values():
            if rec.get("pid") and cq.proc_alive(rec["pid"], rec.get("token")):
                cq.killpg(rec["pid"], signal.SIGKILL)
        shutil.rmtree(self.d, ignore_errors=True)

    def job(self, name, sleep_s=0.2, code=0, lines=("DONE-MARK",), **kw):
        j = {"id": name, "kind": "fake", "cmd": [sys.executable, self.script, self.trace, name, str(sleep_s), str(code)] + list(lines),
             "cwd": self.d, "log": os.path.join(self.d, name + ".log"), "done_marker": "DONE-MARK",
             "fail_marker": "BOOM", "timeout_s": 60, "requires_cards_free": False, "on_fail": "continue"}
        j.update(kw)
        return j

    def write_queue(self, *jobs):
        Path(self.queue_file).write_text(json.dumps({"schema": cq.QUEUE_SCHEMA, "id": "t", "jobs": list(jobs)}))

    def runner(self, probe=None, **kw):
        kw.setdefault("poll_s", 0.05)
        kw.setdefault("done_grace_s", 0.5)
        kw.setdefault("fail_grace_s", 0.5)
        kw.setdefault("kill_wait_s", 2.0)
        return cq.Runner(self.queue_file, self.qdir, probe or (lambda: self.fail("probe must not be called")), **kw)

    def state(self):
        return cq.read_json(os.path.join(self.qdir, "state.json"))

    def spans(self):
        out = {}
        for line in Path(self.trace).read_text().splitlines():
            name, what, t = line.split()
            out.setdefault(name, {})[what] = float(t)
        return out


class OrderAndVerdictTests(Base):
    def test_back_to_back_in_file_order(self):
        self.write_queue(self.job("j1"), self.job("j2"), self.job("j3"))
        self.assertEqual(self.runner().run(), 0)
        sp = self.spans()
        self.assertLessEqual(sp["j1"]["end"], sp["j2"]["start"])
        self.assertLessEqual(sp["j2"]["end"], sp["j3"]["start"])
        self.assertLess(sp["j2"]["start"] - sp["j1"]["end"], 2.0)  # next job starts at once
        st = self.state()
        self.assertEqual(st["state"], "done")
        self.assertEqual([st["jobs"][j]["status"] for j in ("j1", "j2", "j3")], ["done"] * 3)
        self.assertEqual(st["counts"]["done"], 3)
        self.assertIn("job j1 begins", Path(self.job("j1")["log"]).read_text())

    def test_job_is_detached_own_session_stdin_devnull(self):
        probe_script = os.path.join(self.d, "probe_job.py")
        Path(probe_script).write_text(
            "import os,sys\nprint('SID', os.getsid(0)==os.getpid(), 'PGID', os.getpgrp()==os.getpid(), "
            "'STDIN', os.readlink('/proc/self/fd/0'))\nprint('DONE-MARK')\n")
        j = self.job("p")
        j["cmd"] = [sys.executable, probe_script]
        self.write_queue(j)
        self.assertEqual(self.runner().run(), 0)
        text = Path(j["log"]).read_text()
        self.assertIn("SID True PGID True STDIN /dev/null", text)

    def test_fail_marker_with_continue(self):
        self.write_queue(self.job("bad", code=1, lines=("BOOM something", )), self.job("ok"))
        self.assertEqual(self.runner().run(), 4)
        st = self.state()
        self.assertEqual(st["jobs"]["bad"]["status"], "failed")
        self.assertIn("fail marker", st["jobs"]["bad"]["reason"])
        self.assertEqual(st["jobs"]["ok"]["status"], "done")
        self.assertEqual(st["state"], "failed")

    def test_fail_marker_wins_over_done_marker(self):
        self.write_queue(self.job("both", code=0, lines=("DONE-MARK", "BOOM late")))
        self.assertEqual(self.runner().run(), 4)
        self.assertEqual(self.state()["jobs"]["both"]["status"], "failed")

    def test_fail_marker_with_stop_leaves_later_jobs_pending(self):
        self.write_queue(self.job("bad", code=1, lines=("BOOM",), on_fail="stop"), self.job("later"))
        self.assertEqual(self.runner().run(), 2)
        st = self.state()
        self.assertEqual(st["state"], "stopped")
        self.assertEqual(st["jobs"]["later"]["status"], "pending")
        self.assertNotIn("later", self.spans())  # never launched

    def test_exit_without_done_marker_is_failure(self):
        self.write_queue(self.job("quiet", lines=()))
        self.assertEqual(self.runner().run(), 4)
        self.assertIn("without the done marker", self.state()["jobs"]["quiet"]["reason"])

    def test_no_done_marker_configured_uses_exit_code(self):
        self.write_queue(self.job("a", lines=(), done_marker=None, fail_marker=None),
                         self.job("b", code=3, lines=(), done_marker=None, fail_marker=None))
        self.assertEqual(self.runner().run(), 4)
        st = self.state()
        self.assertEqual((st["jobs"]["a"]["status"], st["jobs"]["b"]["status"]), ("done", "failed"))

    def test_timeout_terminates_the_job_group(self):
        self.write_queue(self.job("slow", sleep_s=30, lines=(), timeout_s=0.5), self.job("after"))
        self.assertEqual(self.runner().run(), 4)
        st = self.state()
        self.assertIn("timeout", st["jobs"]["slow"]["reason"])
        self.assertFalse(cq.proc_alive(st["jobs"]["slow"]["pid"], st["jobs"]["slow"]["token"]))
        self.assertEqual(st["jobs"]["after"]["status"], "done")

    def test_done_marker_with_lingering_process_is_done_then_terminated(self):
        self.write_queue(self.job("linger", sleep_s=30))
        t0 = time.time()
        self.assertEqual(self.runner().run(), 0)
        self.assertLess(time.time() - t0, 15)
        st = self.state()
        self.assertEqual(st["jobs"]["linger"]["status"], "done")
        self.assertFalse(cq.proc_alive(st["jobs"]["linger"]["pid"], st["jobs"]["linger"]["token"]))

    def test_disabled_job_is_skipped_and_queue_file_is_reread(self):
        self.write_queue(self.job("a", sleep_s=0.6), self.job("off", enabled=False))
        r = self.runner()
        t = threading.Thread(target=r.run)
        t.start()
        time.sleep(0.3)  # a is running; append a job to the file
        self.write_queue(self.job("a", sleep_s=0.6), self.job("off", enabled=False), self.job("appended"))
        t.join(30)
        st = self.state()
        self.assertEqual(st["jobs"]["off"]["status"], "skipped")
        self.assertEqual(st["jobs"]["appended"]["status"], "done")
        self.assertNotIn("off", self.spans())

    def test_queue_dir_placeholder_and_validation(self):
        self.write_queue(self.job("x", log="{queue_dir}/x.log"))
        self.assertEqual(self.runner().run(), 0)
        self.assertTrue(os.path.exists(os.path.join(self.qdir, "x.log")))
        for bad in ({"on_fail": "explode"}, {"timeout_s": 0}, {"cmd": "not a list"}, {"done_marker": "("},
                    {"cards_policy": "whatever"}):
            self.write_queue(self.job("y", **bad))
            with self.assertRaises(cq.QueueError, msg=str(bad)):
                cq.load_queue(self.queue_file, self.qdir)

    def test_existing_state_needs_resume(self):
        self.write_queue(self.job("a"))
        self.assertEqual(self.runner().run(), 0)
        self.assertEqual(self.runner().run(), 5)  # refuses: state exists, no --resume
        self.assertEqual(self.runner(resume=True).run(), 0)  # nothing left to do
        self.assertEqual(len(self.spans()), 1)

    def test_second_runner_on_the_same_queue_dir_is_refused(self):
        import fcntl
        self.write_queue(self.job("a"))
        with open(os.path.join(self.qdir, "runner.lock"), "w") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(self.runner().run(), 5)
        self.assertFalse(os.path.exists(self.trace))

    def test_status_lists_jobs_in_file_order(self):
        self.write_queue(self.job("zz"), self.job("aa"), self.job("mm"))
        self.assertEqual(self.runner().run(), 0)
        lines = [l.split()[1] for l in cq.format_status(self.state()).splitlines() if l.startswith("  ")]
        self.assertEqual(lines, ["zz", "aa", "mm"])

    def test_retry_failed_reruns_only_failed(self):
        self.write_queue(self.job("bad", code=1, lines=("BOOM",)), self.job("ok"))
        self.assertEqual(self.runner().run(), 4)
        self.assertEqual(self.runner(resume=True, retry_failed=True).run(), 4)
        starts = Path(self.trace).read_text().count("bad start")
        self.assertEqual(starts, 2)
        self.assertEqual(Path(self.trace).read_text().count("ok start"), 1)


class CardGateTests(Base):
    def cards(self, cls, procs=(), owners=(), hbm=60000):
        return {0: {"cls": cls, "owners": list(owners), "procs": list(procs), "hbm": hbm, "ai": 0},
                1: {"cls": "free", "owners": [], "procs": [], "hbm": 100, "ai": 0}}

    def test_waits_while_vllm_holds_a_card_then_launches(self):
        vllm = self.cards("ours", [(111, "VLLMWorker_TP", 57000)], ["saea-arm-a-0"])
        free = {n: {"cls": "free", "owners": [], "procs": [], "hbm": 100, "ai": 0} for n in (0, 1)}
        box = {"cards": vllm, "calls": 0}

        def probe():
            box["calls"] += 1
            return box["cards"]

        self.write_queue(self.job("gated", requires_cards_free=True))
        r = self.runner(probe=probe)
        t = threading.Thread(target=r.run)
        t.start()
        time.sleep(0.5)
        st = self.state()
        self.assertEqual(st["jobs"]["gated"]["status"], "waiting")
        self.assertIn("card 0 held (vllm)", st["waiting"]["reason"])
        self.assertFalse(os.path.exists(self.trace))  # not launched
        box["cards"] = free
        t.join(30)
        self.assertEqual(self.state()["jobs"]["gated"]["status"], "done")
        self.assertIsNone(self.state()["waiting"])
        self.assertGreater(box["calls"], 3)

    def test_no_other_policy_lets_the_resident_vllm_stay_but_never_a_foreign_card(self):
        ours = self.cards("ours", [(111, "VLLMWorker_TP", 57000)], ["saea-arm-a-0"])
        self.assertEqual(cq.card_blockers(ours, "no_other"), [])
        self.assertTrue(cq.card_blockers(ours, "all_free"))
        foreign = self.cards("other", [(222, "python3", 46000)], ["someone-else"])
        self.assertTrue(cq.card_blockers(foreign, "no_other"))
        unattributed = self.cards("other", [], [], hbm=30000)
        self.assertIn("unattributed HBM", cq.card_blockers(unattributed, "no_other")[0])
        ours_py = self.cards("ours", [(111, "python3", 5000)], ["saea-encoder-0"])
        self.assertTrue(cq.card_blockers(ours_py, "no_other"))  # an encoder job of ours is not a resident service
        resident_parent = self.cards("ours", [(111, "VLLMWorker_TP", 50000), (112, "python3", 1000)], ["saea-omni-i0"])
        for card in resident_parent.values(): card['resident_python_pids'] = [112]
        self.assertEqual(cq.card_blockers(resident_parent, "no_other"), [])
        for card in resident_parent.values(): card['procs'].append((113, 'python3', 1000))
        self.assertTrue(cq.card_blockers(resident_parent, "no_other"))
        self.assertEqual(cq.card_blockers({}, "no_other")[0][:8], "npu-smi ")  # unreadable: never launch blind

    def test_no_other_job_starts_with_vllm_resident_and_waits_for_foreign(self):
        box = {"cards": self.cards("other", [(222, "python3", 46000)], ["someone-else"])}
        self.write_queue(self.job("b", requires_cards_free=True, cards_policy="no_other"))
        r = self.runner(probe=lambda: box["cards"])
        t = threading.Thread(target=r.run)
        t.start()
        time.sleep(0.4)
        self.assertEqual(self.state()["jobs"]["b"]["status"], "waiting")
        box["cards"] = self.cards("ours", [(111, "VLLMWorker_TP", 57000)], ["saea-arm-a-0"])
        t.join(30)
        self.assertEqual(self.state()["jobs"]["b"]["status"], "done")

    def test_ungated_job_never_calls_the_probe(self):
        self.write_queue(self.job("free-running"))
        self.assertEqual(self.runner().run(), 0)  # default probe fails the test if called

    def test_waits_for_a_peer_queue_job(self):
        peer = os.path.join(self.d, "peer")
        os.makedirs(peer)
        Path(peer, "state.json").write_text(json.dumps({"queue_id": "peer", "runner_pid": os.getpid(), "current": "pj"}))
        free = {0: {"cls": "free", "owners": [], "procs": [], "hbm": 10, "ai": 0}}
        self.write_queue(self.job("g", requires_cards_free=True))
        r = self.runner(probe=lambda: free, peers=[peer])
        t = threading.Thread(target=r.run)
        t.start()
        time.sleep(0.4)
        self.assertIn("other queue peer runs job pj", self.state()["waiting"]["reason"])
        Path(peer, "state.json").write_text(json.dumps({"queue_id": "peer", "runner_pid": os.getpid(), "current": None}))
        t.join(30)
        self.assertEqual(self.state()["jobs"]["g"]["status"], "done")

    def test_classification_matches_the_watchdog_parse(self):
        txt = SMI % (60000, 3400, "| 0       0                 | 111           | VLLMWorker_TP      | 57000                 | NA |\n"
                                  "| 1       0                 | 222           | python3            | 3000                  | NA |\n")
        ps = [{"id": "a" * 64, "name": "saea-arm-a-0", "image": "i"}, {"id": "b" * 64, "name": "someone-else", "image": "i"}]
        cg = {111: "a" * 64, 222: "b" * 64}
        cards = cq.classify_cards(txt, ps, cg.get)
        self.assertEqual((cards[0]["cls"], cards[1]["cls"]), ("ours", "other"))
        self.assertEqual(cards[0]["owners"], ["saea-arm-a-0"])
        try:
            sys.path.insert(0, str(HERE.parent / "watchdog"))
            import watchdog as wd
        except ImportError:
            return
        self.assertEqual(wd.parse_smi(txt), cq.parse_smi(txt))


class WaitForFinalTests(Base):
    def setUp(self):
        super().setUp()
        self.gate = os.path.join(self.d, "chain-status.json")
        self.alive = {4321}

    def status(self, **kw):
        Path(self.gate).write_text(json.dumps(dict({"state": "running", "pid": 4321, "current_step": "gate"}, **kw)))

    def test_final_rule_matches_the_chain_gate(self):
        pa = lambda pid: pid in self.alive
        self.assertFalse(cq.final_gate(None, pa)[0])
        self.assertFalse(cq.final_gate({"state": "running", "pid": 1}, pa)[0])
        self.assertFalse(cq.final_gate({"state": "done", "pid": 4321}, pa)[0])  # state final but pid alive
        self.assertTrue(cq.final_gate({"state": "done", "pid": 99}, pa)[0])
        self.assertTrue(cq.final_gate({"state": "failed", "pid": 99}, pa)[0])
        self.assertTrue(cq.final_gate({"state": "interrupted", "pid": None}, pa)[0])
        self.assertFalse(cq.final_gate({"state": "weird", "pid": 99}, pa)[0])

    def test_job_waits_for_a_missing_then_running_then_final_status_file(self):
        probe_calls = []
        self.write_queue(self.job("after-chain", wait_for_final=self.gate))
        r = self.runner(probe=lambda: probe_calls.append(1) or {}, pid_alive=lambda pid: pid in self.alive)
        t = threading.Thread(target=r.run)
        t.start()
        time.sleep(0.4)  # file missing
        st = self.state()
        self.assertEqual(st["jobs"]["after-chain"]["status"], "waiting")
        self.assertIn("unreadable", st["waiting"]["reason"])
        self.status()  # running
        time.sleep(0.3)
        self.assertIn("state 'running'", self.state()["waiting"]["reason"])
        self.status(state="done")  # state final but the pid is still alive
        time.sleep(0.3)
        self.assertIn("still alive", self.state()["waiting"]["reason"])
        self.assertFalse(os.path.exists(self.trace))  # nothing launched yet
        self.alive.clear()  # the chain process exits
        t.join(30)
        self.assertEqual(self.state()["jobs"]["after-chain"]["status"], "done")
        self.assertEqual(probe_calls, [])  # no cards gate on this job: npu-smi never read

    def test_final_gate_comes_before_the_cards_gate(self):
        free = {0: {"cls": "free", "owners": [], "procs": [], "hbm": 10, "ai": 0}}
        calls = []
        self.status()
        self.write_queue(self.job("both", wait_for_final=self.gate, requires_cards_free=True, cards_policy="no_other"))
        r = self.runner(probe=lambda: calls.append(1) or free, pid_alive=lambda pid: False)
        t = threading.Thread(target=r.run)
        t.start()
        time.sleep(0.4)
        self.assertEqual(calls, [])  # the chain still owns the cards: not even read
        self.status(state="failed")
        t.join(30)
        self.assertEqual(self.state()["jobs"]["both"]["status"], "done")
        self.assertTrue(calls)

    def test_dry_run_shows_the_gate_and_validation(self):
        self.write_queue(self.job("g", wait_for_final=self.gate))
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cq.dry_run(self.queue_file, self.qdir)
        self.assertIn("WAIT for final " + self.gate, buf.getvalue())
        self.write_queue(self.job("g", wait_for_final=5))
        with self.assertRaises(cq.QueueError):
            cq.load_queue(self.queue_file, self.qdir)


class CliTests(Base):
    def cli(self, *args, path_extra=None, **kw):
        env = dict(os.environ)
        if path_extra:
            env["PATH"] = path_extra + os.pathsep + env["PATH"]
        return subprocess.run([sys.executable, str(HERE / "card_queue.py")] + list(args), capture_output=True, text=True,
                              env=env, **kw)

    def tripwire(self):
        """A fake npu-smi first on PATH: a real call would leave a file behind and fail the test."""
        d = os.path.join(self.d, "bin")
        os.makedirs(d)
        hit = os.path.join(self.d, "npu-smi-called")
        p = os.path.join(d, "npu-smi")
        Path(p).write_text("#!/bin/sh\necho called > %s\nexit 1\n" % hit)
        os.chmod(p, 0o755)
        return d, hit

    def test_dry_run_prints_every_command_and_runs_nothing(self):
        self.write_queue(self.job("one", requires_cards_free=True, cards_policy="no_other", on_fail="stop"),
                         self.job("two", log="{queue_dir}/two.log"))
        bindir, hit = self.tripwire()
        out = self.cli("run", "--queue", self.queue_file, "--queue-dir", self.qdir, "--dry-run", path_extra=bindir)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("[1/2] one", out.stdout)
        self.assertIn("[2/2] two", out.stdout)
        self.assertIn("WAIT for cards (policy no_other)", out.stdout)
        self.assertIn("on_fail=stop", out.stdout)
        self.assertIn(self.script, out.stdout)
        self.assertIn(os.path.join(self.qdir, "two.log"), out.stdout)
        self.assertFalse(os.path.exists(self.trace))
        self.assertFalse(os.path.exists(os.path.join(self.qdir, "state.json")))
        self.assertFalse(os.path.exists(hit))

    def test_resume_after_the_runner_is_killed_adopts_the_running_job(self):
        self.write_queue(self.job("long", sleep_s=3.0), self.job("next", sleep_s=0.1))
        bindir, hit = self.tripwire()
        args = ["run", "--queue", self.queue_file, "--queue-dir", self.qdir, "--poll-s", "0.1"]
        env = dict(os.environ, PATH=bindir + os.pathsep + os.environ["PATH"])
        runner = subprocess.Popen([sys.executable, str(HERE / "card_queue.py")] + args, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, start_new_session=True)
        deadline = time.time() + 10
        while time.time() < deadline and not (self.state() or {}).get("jobs", {}).get("long", {}).get("pid"):
            time.sleep(0.1)
        pid = self.state()["jobs"]["long"]["pid"]
        os.kill(runner.pid, signal.SIGKILL)
        runner.wait()
        time.sleep(0.3)
        self.assertTrue(cq.proc_alive(pid), "the job must survive the runner")
        self.assertEqual(self.state()["jobs"]["long"]["status"], "running")
        out = self.cli("run", *args[1:], "--resume", path_extra=bindir, timeout=60)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("adopted pid %d" % pid, out.stdout)
        st = self.state()
        self.assertEqual((st["jobs"]["long"]["status"], st["jobs"]["next"]["status"]), ("done", "done"))
        self.assertEqual(Path(self.trace).read_text().count("long start"), 1)  # never launched twice
        self.assertEqual(st["state"], "done")
        self.assertFalse(os.path.exists(hit))

    def test_resume_derives_the_verdict_of_a_job_that_ended_meanwhile(self):
        self.write_queue(self.job("quick", sleep_s=0.3), self.job("after", sleep_s=0.1))
        args = ["run", "--queue", self.queue_file, "--queue-dir", self.qdir, "--poll-s", "0.1"]
        runner = subprocess.Popen([sys.executable, str(HERE / "card_queue.py")] + args, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        deadline = time.time() + 10
        while time.time() < deadline and not (self.state() or {}).get("jobs", {}).get("quick", {}).get("pid"):
            time.sleep(0.05)
        os.kill(runner.pid, signal.SIGKILL)
        runner.wait()
        time.sleep(1.0)  # the job finishes while no runner exists
        out = self.cli("run", *args[1:], "--resume", timeout=60)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertEqual(self.state()["jobs"]["quick"]["status"], "done")
        self.assertEqual(self.state()["jobs"]["after"]["status"], "done")

    def test_status_subcommand(self):
        self.write_queue(self.job("a"), self.job("b", code=1, lines=("BOOM",)))
        self.assertEqual(self.runner().run(), 4)
        out = self.cli("status", "--queue-dir", self.qdir)
        self.assertEqual(out.returncode, 0)
        self.assertIn("state=failed", out.stdout)
        self.assertIn("done", out.stdout)
        self.assertIn("fail marker", out.stdout)
        js = json.loads(self.cli("status", "--queue-dir", self.qdir, "--json").stdout)
        self.assertEqual(js["counts"]["failed"], 1)
        self.assertEqual(self.cli("status", "--queue-dir", os.path.join(self.d, "nowhere")).returncode, 1)

    def test_bad_queue_file_exits_5(self):
        Path(self.queue_file).write_text("{}")
        out = self.cli("run", "--queue", self.queue_file, "--queue-dir", self.qdir)
        self.assertEqual(out.returncode, 5)


class RecoveryEvidenceTests(Base):
    def checkpoint_without_pid(self, *, finished=False):
        self.write_queue(self.job('one', on_fail='stop'))
        runner = self.runner()
        runner.load()
        job = runner.queue['jobs'][0]
        rec = runner.state['jobs']['one']
        rec.update(status='running', token='absent-token-for-crash-test', pid=None,
                   started_ts=time.time(), log_offset=0, job_spec=job)
        if finished:
            Path(job['log']).write_text('DONE-MARK\n')
        runner.save()
        return runner, job

    def test_uncertain_launch_is_not_replayed(self):
        self.checkpoint_without_pid()
        self.assertEqual(self.runner(resume=True).run(), 2)
        self.assertFalse(Path(self.trace).exists())
        self.assertIn('outcome unknown', self.state()['jobs']['one']['reason'])

    def test_finished_before_pid_save_is_recovered_without_replay(self):
        self.checkpoint_without_pid(finished=True)
        self.assertEqual(self.runner(resume=True).run(), 0)
        self.assertFalse(Path(self.trace).exists())
        self.assertEqual(self.state()['jobs']['one']['status'], 'done')

    def test_running_job_keeps_original_spec_after_queue_edit(self):
        runner, original = self.checkpoint_without_pid()
        self.write_queue(self.job('one', done_marker='OTHER', cmd=['false']))
        runner.reload_queue()
        self.assertEqual(runner.next_job(), original)

    def test_removed_running_job_is_supervised_before_next_job(self):
        runner, original = self.checkpoint_without_pid()
        self.write_queue(self.job('next'))
        runner.reload_queue()
        self.assertEqual(runner.next_job(), original)


if __name__ == "__main__":
    unittest.main()


def test_resident_python_classification_requires_same_known_container(monkeypatch):
    cards={0:{'hbm':55000,'ai':0}}
    procs={0:[(10,'VLLMWorker_TP',50000),(11,'python3',1000),(12,'python3',1000)]}
    monkeypatch.setattr(cq,'parse_smi',lambda _: (cards,procs))
    containers=[{'id':'resident','name':'saea-omni-i0'},{'id':'encoder','name':'saea-encoder'}]
    result=cq.classify_cards('smi',containers,lambda pid:'resident' if pid in (10,11) else 'encoder')
    assert result[0]['resident_python_pids']==[11]
    assert cq.card_blockers(result,'no_other')
    procs[0].pop()
    result=cq.classify_cards('smi',containers,lambda pid:'resident')
    assert cq.card_blockers(result,'no_other')==[]
