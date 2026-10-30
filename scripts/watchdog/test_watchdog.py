#!/usr/bin/env python3
"""Offline tests for watchdog.py: fake docker / npu-smi / HTTP readers and fixture logs in the real schemas.
Run: python3 -m pytest test_watchdog.py   or   python3 test_watchdog.py"""
import json, os, shutil, sys, tempfile, time, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import watchdog as wd


class FakeEnv:
    def __init__(self):
        self.t = time.time()
        self.containers, self.http, self.smi, self.cgroup = [], {}, "", {}

    def now(self):
        return self.t

    def docker_ps(self):
        return self.containers

    def http_get(self, url, timeout=5):
        return self.http.get(url)

    def npu_smi(self):
        return self.smi

    def pid_container(self, pid):
        return self.cgroup.get(pid)


def read(path):
    with open(path) as f:
        return f.read()


def load(path):
    return json.loads(read(path))


def level_json(path, n, tok=900.0, p10=30.0, lat=200.0, pre=0.0, err=0):
    """Per-level report in the schema of saea-synthetic-serving-sweep-v1 (only the fields the watchdog reads)."""
    lv = {"concurrency_per_instance": n, "output_tok_s": tok, "e2e_tok_s": {"p10": p10, "p50": 40.0},
          "latency_s": {"max": lat, "p50": 100.0}, "preemptions_delta": pre, "errors": err, "timeouts": 0, "ok": 96}
    write(path, json.dumps({"schema": "saea-synthetic-serving-sweep-v1", "levels": [lv], "settings": {"levels": [n]}}))


def write(path, text, mode="w"):
    with open(path, mode) as f:
        f.write(text)


def prom_text(running=0, waiting=0, kv=0.0, pre=0.0):
    return ('vllm:num_requests_running{engine="0"} %s\nvllm:num_requests_waiting{engine="0"} %s\n'
            'vllm:kv_cache_usage_perc{engine="0"} %s\nvllm:num_preemptions_total{engine="0"} %s\n'
            'vllm:num_preemptions_created{engine="0"} 1.7e9\n' % (running, waiting, kv, pre))


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.env = FakeEnv()
        self.jobs = []

    def tearDown(self):
        shutil.rmtree(self.d)

    def start(self, *jobs, host="h"):
        for j in jobs:
            j.setdefault("host", host)
        write(os.path.join(self.d, "jobs.json"), json.dumps(list(jobs)))
        self.w = wd.Watchdog(host, os.path.join(self.d, "jobs.json"), self.d, env=self.env)

    def cyc(self, advance=60):
        self.env.t += advance
        out, lines = self.w.cycle()
        return out, lines

    def bundles(self):
        return sorted(os.listdir(os.path.join(self.d, "bundles")))


class VllmTests(Base):
    def setUp(self):
        super().setUp()
        self.log = os.path.join(self.d, "srv.log")
        write(self.log, "(EngineCore pid=86) INFO started\nTraceback (most recent call last):\nold failure\n")
        self.start({"id": "srv", "kind": "vllm_server", "params": {"container": "c0", "port": 8200, "log": self.log}})
        self.env.containers = [{"id": "x" * 64, "name": "c0", "image": "img"}]
        self.up(0.0)

    def up(self, pre):
        self.env.http = {"http://127.0.0.1:8200/v1/models": (200, "{}"),
                         "http://127.0.0.1:8200/metrics": (200, prom_text(3, 1, 0.25, pre))}

    def test_running_and_old_errors_ignored_and_dedup(self):
        out, lines = self.cyc()
        self.assertEqual(out["srv"]["state"], "running")
        self.assertEqual(out["srv"]["metrics"]["running"], 3.0)
        self.assertEqual(len(lines), 1)  # one INFO for new -> running; historic Traceback is baseline
        self.assertEqual(self.cyc()[1], [])  # unchanged state: nothing repeated
        self.assertEqual(self.bundles(), [])

    def test_new_error_lines_bad_with_bundle(self):
        self.cyc()
        write(self.log, "(EngineCore pid=86) ERROR boom\nTraceback (most recent call last):\n", "a")
        _, lines = self.cyc()
        self.assertTrue(lines[0].split()[1] == "BAD" and "error line" in lines[0])
        b = load(os.path.join(self.d, "bundles", self.bundles()[0]))
        self.assertEqual(b["evidence"][0]["hit_lines"], [4, 5])  # exact new line numbers
        self.assertEqual(b["evidence"][1]["tail_lines"], 80)

    def test_info_lines_with_enginecore_are_not_errors(self):
        self.cyc()
        write(self.log, "(EngineCore pid=86) INFO all good\n", "a")
        self.assertEqual(self.cyc()[1], [])

    def test_preemption_increase_bad(self):
        self.cyc()
        self.up(2.0)
        _, lines = self.cyc()
        self.assertIn("preemptions increased 0 -> 2", lines[0])
        self.assertEqual(len(self.bundles()), 1)

    def test_container_gone_failed(self):
        self.cyc()
        self.env.containers = []
        out, lines = self.cyc()
        self.assertEqual(out["srv"]["state"], "failed")
        self.assertIn(" BAD srv ", lines[0])

    def test_endpoint_down_more_than_three_checks(self):
        self.cyc()
        self.env.http = {}
        for _ in range(3):
            self.assertEqual(self.cyc()[0]["srv"]["state"], "running")
        out, lines = self.cyc()
        self.assertEqual(out["srv"]["state"], "failed")
        self.assertIn("endpoint down", lines[0])
        self.up(0.0)
        out, lines = self.cyc()  # recovery is reported once
        self.assertEqual(out["srv"]["state"], "running")
        self.assertIn("INFO", lines[0])


class SweepTests(Base):
    def setUp(self):
        super().setUp()
        self.pre = os.path.join(self.d, "sweep-x")
        self.start({"id": "sw", "kind": "sweep", "params": {"run_prefix": self.pre}})

    def smi(self, level, used):
        write(self.pre + "-npu-smi.json", json.dumps({"samples": [
            {"level": level, "hbm_used_mib": {"0": used, "1": used - 10}, "hbm_total_mib": 65536}]}))

    def test_unknown_then_running_then_done(self):
        self.assertEqual(self.cyc()[0]["sw"]["state"], "unknown")
        level_json(self.pre + "-L8.json", 8)
        write(self.pre + "-L8.stderr.log", "level 8: 96/96 ok\n")
        write(self.pre + "-L16.stderr.log", "level 16/instance x 4 instance(s): fitting and running\n")
        self.smi(8, 61000)
        out, lines = self.cyc()
        s = out["sw"]
        self.assertEqual(s["state"], "running")
        self.assertIn("running L16", s["progress"])
        self.assertEqual(s["metrics"]["levels"][0], {"L": 8, "tok_s": 900.0, "p10": 30.0, "lat_max": 200.0, "pre": 0.0,
                                                     "err": 0, "hbm_free": 4536, "ok": True})
        self.assertTrue(any("level 8 passes" in l for l in lines))
        write(self.pre + ".json", json.dumps({"done": True, "stopped_at_level": None, "levels": []}))
        out, _ = self.cyc()
        self.assertEqual(out["sw"]["state"], "done")

    def test_threshold_failures_are_bad_with_bundle(self):
        level_json(self.pre + "-L16.json", 16, p10=8.5, lat=430.0, pre=2.0)
        self.smi(16, 64500)  # 1036 MiB free
        _, lines = self.cyc()
        bad = [l for l in lines if " BAD " in l]
        self.assertEqual(len(bad), 1)
        for frag in ("p10 8.5<10", "latency max 430>=420", "preemptions 2>0", "min HBM free 1036"):
            self.assertIn(frag, bad[0])
        b = load(os.path.join(self.d, "bundles", self.bundles()[0]))
        self.assertTrue(b["evidence"][0]["path"].endswith("-L16.json"))
        self.assertEqual(self.cyc()[1], [])  # same failure is not reported again

    def test_preemptions_unreadable_is_bad(self):
        level_json(self.pre + "-L8.json", 8, pre=None)
        self.assertIn("preemptions unreadable", self.cyc()[1][-1])

    def test_hbm_from_summary_preferred(self):
        level_json(self.pre + "-L8.json", 8)
        write(self.pre + ".json", json.dumps({"done": False, "levels": [
            {"concurrency_per_instance": 8, "min_hbm_free_mib_per_card": 1500}]}))
        self.assertIn("min HBM free 1500", self.cyc()[1][-1])

    def test_stderr_error_bad(self):
        write(self.pre + "-L8.stderr.log", "starting\nTraceback (most recent call last):\n")
        _, lines = self.cyc()
        self.assertTrue(any(" BAD " in l and "stderr error" in l for l in lines))

    def test_stalled_without_file_progress(self):
        level_json(self.pre + "-L8.json", 8)
        self.assertEqual(self.cyc()[0]["sw"]["state"], "running")
        out, lines = self.cyc(advance=21 * 60)
        self.assertEqual(out["sw"]["state"], "stalled")
        self.assertIn(" BAD ", [l for l in lines if "stalled" in l][0])
        write(self.pre + "-npu-smi.json", "{}")  # progress resumes (file mtime follows the fake clock)
        os.utime(self.pre + "-npu-smi.json", (self.env.t, self.env.t))
        self.assertEqual(self.cyc()[0]["sw"]["state"], "running")


class PytestTests(Base):
    def setUp(self):
        super().setUp()
        self.log = os.path.join(self.d, "p1.log")
        self.start({"id": "pt", "kind": "pytest", "params": {"logs": os.path.join(self.d, "p*.log")}})

    def test_progress_then_done_with_failures(self):
        write(self.log, "bringing up nodes...\n..F.......... [ 40%]\n.....E....... [ 60%]\n")
        out, _ = self.cyc()
        self.assertEqual(out["pt"]["state"], "running")
        self.assertEqual(out["pt"]["metrics"], {"passed": 24, "failed": 1, "error": 1})
        self.assertTrue(out["pt"]["progress"].startswith("60%"))
        write(self.log, "FAILED t.py::x\n2 failed, 30 passed, 1 error in 12.00s (0:00:12)\nPYTEST_EXIT=1\n", "a")
        out, lines = self.cyc()
        self.assertEqual(out["pt"]["state"], "done")
        self.assertEqual(out["pt"]["metrics"], {"passed": 30, "failed": 2, "error": 1})
        self.assertTrue(any(" WARN " in l and "finished with failures" in l for l in lines))
        self.assertEqual(self.cyc()[1], [])

    def test_partial_line_waits_for_newline(self):
        write(self.log, "....... [ 10%]\n....")
        self.assertEqual(self.cyc()[0]["pt"]["metrics"]["passed"], 7)
        write(self.log, "... [ 20%]\n", "a")
        self.assertEqual(self.cyc()[0]["pt"]["metrics"]["passed"], 14)

    def test_collection_error_bad(self):
        write(self.log, "ERROR collecting tests/unit/test_a.py\nInterrupted: 1 error during collection\n")
        _, lines = self.cyc()
        self.assertEqual(len([l for l in lines if " BAD " in l]), 1)
        self.assertEqual(len(self.bundles()), 1)

    def test_stalled(self):
        write(self.log, "..... [ 10%]\n")
        self.cyc()
        out, _ = self.cyc(advance=21 * 60)
        self.assertEqual(out["pt"]["state"], "stalled")


SMI = """\
| NPU   Name                | Health        | Power(W)             Temp(C)                 Hugepages-Usage(page)   |
| Chip                      | Bus-Id        | AICore(%%)            Memory-Usage(MB)        HBM-Usage(MB)           |
| 0     910B3               | OK            | 115.9                44                      0    / 0                |
| 0                         | 0000:C1:00.0  | 11                   0    / 0                %d / 65536            |
| 1     910B3               | OK            | 97.1                 31                      0    / 0                |
| 0                         | 0000:C2:00.0  | 0                    0    / 0                %d / 65536            |
| NPU     Chip              | Process id    | Process name       | Process memory(MB)    | Process id in container |
%s"""


class CardTests(Base):
    def setUp(self):
        super().setUp()
        self.start({"id": "cards", "kind": "npu_cards", "params": {}})
        self.env.containers = [{"id": "a" * 64, "name": "saea-arm-a-0", "image": "i"},
                               {"id": "b" * 64, "name": "someone-else", "image": "i"}]
        self.env.cgroup = {111: "a" * 64, 222: "b" * 64}

    def test_new_busy_card_not_ours_warns_once(self):
        self.env.smi = SMI % (3400, 3400, "| No running processes found in NPU 0 |\n")
        out, _ = self.cyc()
        self.assertEqual(out["cards"]["metrics"]["free"], [0, 1])
        self.assertEqual(out["cards"]["metrics"]["hbm_mib"], [3400, 3400])
        self.env.smi = SMI % (3400, 50000, "| 1       0                 | 222           | python3            | 46000                 | NA |\n")
        _, lines = self.cyc()
        self.assertEqual(len(lines), 1)
        self.assertIn("WARN", lines[0])
        self.assertIn("card 1 now busy by a non-owned job (someone-else", lines[0])
        self.assertEqual(self.cyc()[1], [])

    def test_own_container_is_ours_not_warning(self):
        self.env.smi = SMI % (3400, 3400, "")
        self.cyc()
        self.env.smi = SMI % (60000, 3400, "| 0       0                 | 111           | VLLMWorker_TP      | 57000                 | NA |\n")
        out, lines = self.cyc()
        self.assertEqual(out["cards"]["metrics"]["ours"], [0])
        self.assertTrue(all("INFO" in l for l in lines))

    def test_npu_smi_missing_is_unknown(self):
        self.env.smi = None
        self.assertEqual(self.cyc()[0]["cards"]["state"], "unknown")


class GenericAndShardTests(Base):
    def test_generic_log_done_and_bad(self):
        log = os.path.join(self.d, "dl.log")
        self.start({"id": "dl", "kind": "generic_log",
                    "params": {"path": log, "done_regex": "ALL DONE", "bad_regex": "ERROR|Killed"}})
        self.assertEqual(self.cyc()[0]["dl"]["state"], "unknown")
        write(log, "fetching 1/3\n")
        self.assertEqual(self.cyc()[0]["dl"]["state"], "running")
        write(log, "ERROR: checksum mismatch\n", "a")
        _, lines = self.cyc()
        self.assertTrue(any(" BAD " in l for l in lines))
        write(log, "ALL DONE\n", "a")
        self.assertEqual(self.cyc()[0]["dl"]["state"], "done")

    def test_shard_runs_progress_timeout_and_regression(self):
        for tree in ("head", "work"):
            d = os.path.join(self.d, "out", tree, "0")
            os.makedirs(d)
            write(os.path.join(d, "files.txt"), "tests/a.py\ntests/b.py\ntests/c.py\n")
            write(os.path.join(d, "progress.txt"), "tests/a.py rc=0 t=2\ntests/b.py rc=%d t=900\n" % (137 if tree == "head" else 0))
        write(os.path.join(self.d, "out", "work", "0", "progress.txt"), "tests/a.py rc=1 t=2\ntests/b.py rc=0 t=2\n")
        self.start({"id": "rg", "kind": "shard_runs", "params": {
            "dirs": os.path.join(self.d, "out", "*", "*"), "shards": 2, "base": "head", "cand": "work"}})
        out, lines = self.cyc()
        self.assertEqual(out["rg"]["state"], "running")
        self.assertIn("4/6 files", out["rg"]["progress"])
        self.assertTrue(any(" WARN " in l and "timeout" in l for l in lines))
        self.assertTrue(any(" BAD " in l and "fail in work but pass in head" in l and "tests/a.py" in l for l in lines))
        self.assertEqual(len(self.bundles()), 1)
        for tree in ("head", "work"):
            write(os.path.join(self.d, "out", tree, "0", "DONE"), "DONE\n")
        self.assertEqual(self.cyc()[0]["rg"]["state"], "done")


def rt_contact(row, arm, node="solver", trunc=False, think=900, used=(), outcome="ok", prompt=4000, compl=1200):
    """One contact line in the schema of saea-rollout-trace-v1 (only the fields the watchdog reads, plus the fixed envelope)."""
    return {"schema": "saea-rollout-trace-v1", "kind": "contact", "ts": "2026-10-30T10:00:00.000Z", "run_id": "RUN-1",
            "attempt_id": "attempt-1", "row_id": row, "arm": arm, "node": node, "outcome": outcome, "truncated": trunc,
            "thinking_tokens": think, "prompt_tokens": prompt, "completion_tokens": compl,
            "supply_items": [{"item_id": "s::%d" % i, "rank": i + 1, "score": 0.5, "used": u} for i, u in enumerate(used)]}


def rt_score(row, arm, f1, exact):
    return {"schema": "saea-rollout-trace-v1", "kind": "score", "run_id": "RUN-1", "attempt_id": "attempt-1", "row_id": row,
            "arm": arm, "scorer_id": "s", "scenario_action_exact": exact, "slu_f1": f1}


def rt_tfrl(row, reward, selected):
    return {"schema": "saea-rollout-trace-v1", "kind": "tfrl", "run_id": "RUN-1", "attempt_id": "attempt-1", "row_id": row,
            "arm": None, "trajectory_id": "t-%s" % row, "reward": reward, "selected": selected}


def rt_write(path, docs, mode="a"):
    write(path, "".join(json.dumps(d) + "\n" for d in docs), mode)


class RolloutTraceTests(Base):
    def setUp(self):
        super().setUp()
        self.f = os.path.join(self.d, "rollout-trace.jsonl")
        self.params = {"path": self.f, "min_contacts_for_rates": 4}
        self.start({"id": "rt", "kind": "rollout_trace", "params": self.params})

    def test_missing_file_is_unknown_then_summary_per_arm_node_supply_and_scores(self):
        out, _ = self.cyc()
        self.assertEqual(out["rt"]["state"], "unknown")
        rt_write(self.f, [
            rt_contact("r1", "lock", node="perception", think=2000),
            rt_contact("r1", "lock", trunc=True, think=3112, used=(True, False, None)),
            rt_contact("r2", "lock", think=1000, used=(False, False)),
            rt_contact("r1", "aa", trunc=False, think=800, used=(True,)),
            rt_score("r1", "lock", 0.5, True), rt_score("r2", "lock", 1.0, False), rt_score("r1", "aa", 0.0, False),
        ])
        out, lines = self.cyc()
        m = out["rt"]["metrics"]
        self.assertEqual(out["rt"]["state"], "running")
        self.assertEqual(m["contacts"], 4)
        self.assertEqual(m["arms"], {"aa": 1, "lock": 3})
        self.assertEqual(m["nodes"]["solver"]["n"], 3)
        self.assertAlmostEqual(m["nodes"]["solver"]["trunc_pct"], 33.3)
        self.assertEqual(m["nodes"]["solver"]["think_mean"], 1637)  # (3112 + 1000 + 800) / 3
        self.assertEqual(m["nodes"]["perception"]["think_mean"], 2000)
        self.assertEqual(m["supply"]["solver_contacts"], 3)
        self.assertEqual(m["supply"]["item_use_pct"], 40.0)  # 2 used of 5 checkable (the None item is not counted)
        self.assertAlmostEqual(m["supply"]["contact_use_pct"], 66.7)
        self.assertEqual(m["scored"]["lock"], {"rows": 2, "slu_f1_mean": 0.75, "sa_exact_pct": 50.0})
        self.assertEqual(m["scored"]["aa"]["rows"], 1)
        self.assertIn("contacts 4", out["rt"]["progress"])
        self.assertIn("scored 3", out["rt"]["progress"])
        self.assertEqual(len(lines), 1)  # only the unknown -> running state change

    def test_incremental_reads_count_each_line_once_and_partial_lines_wait(self):
        rt_write(self.f, [rt_contact("r1", "lock")])
        self.assertEqual(self.cyc()[0]["rt"]["metrics"]["contacts"], 1)
        rt_write(self.f, [rt_contact("r2", "lock")])
        write(self.f, json.dumps(rt_contact("r3", "lock"))[:40], "a")  # a line still being written
        self.assertEqual(self.cyc()[0]["rt"]["metrics"]["contacts"], 2)
        write(self.f, json.dumps(rt_contact("r3", "lock"))[40:] + "\n", "a")
        self.assertEqual(self.cyc()[0]["rt"]["metrics"]["contacts"], 3)
        self.assertEqual(self.cyc()[0]["rt"]["metrics"]["contacts"], 3)

    def test_tfrl_reward_distribution_and_acceptance(self):
        rt_write(self.f, [rt_tfrl("r1", 0.5, True), rt_tfrl("r2", -0.3, False), rt_tfrl("r3", 0.0, False),
                          rt_tfrl("r4", 0.25, True), rt_tfrl("r5", 0.02, False)])
        out, _ = self.cyc()
        t = out["rt"]["metrics"]["tfrl"]
        self.assertEqual((t["n"], t["accepted"], t["pos"], t["zero"], t["neg"]), (5, 2, 2, 2, 1))
        self.assertEqual((t["reward_min"], t["reward_max"]), (-0.3, 0.5))
        self.assertAlmostEqual(t["reward_mean"], 0.094, places=4)
        self.assertIn("tfrl n=5 acc=2", out["rt"]["progress"])

    def test_truncation_warning_once_and_error_bad_once(self):
        rt_write(self.f, [rt_contact("r%d" % i, "lock", trunc=True) for i in range(4)])
        _, lines = self.cyc()
        self.assertTrue(any(" WARN " in l and "solver truncation rate 100.0% over 4" in l for l in lines))
        rt_write(self.f, [rt_contact("e%d" % i, "lock", outcome="error") for i in range(2)])
        _, lines = self.cyc()
        self.assertEqual(len([l for l in lines if "truncation" in l]), 0)  # warned once already
        self.assertTrue(any(" BAD " in l and "2 of 6 contacts ended in error" in l for l in lines))
        self.assertEqual(len(self.bundles()), 1)
        rt_write(self.f, [rt_contact("e9", "lock", outcome="error")])
        self.assertEqual([l for l in self.cyc()[1] if "ended in error" in l], [])

    def test_invalid_lines_are_bad_with_exact_line_numbers(self):
        rt_write(self.f, [rt_contact("r1", "lock")])
        write(self.f, "{not json\n" + json.dumps({"schema": "old", "kind": "contact"}) + "\n", "a")
        out, lines = self.cyc()
        self.assertEqual(out["rt"]["metrics"]["invalid"], 2)
        self.assertEqual(out["rt"]["metrics"]["contacts"], 1)
        bad = [l for l in lines if " BAD " in l]
        self.assertEqual(len(bad), 1)
        b = load(os.path.join(self.d, "bundles", self.bundles()[0]))
        self.assertEqual(b["evidence"][0]["hit_lines"], [2, 3])

    def test_done_when_expected_counts_reached_and_status_stays_small(self):
        self.params.update(expected_contacts=3, expected_scores=1)
        write(os.path.join(self.d, "jobs.json"), json.dumps([{"id": "rt", "kind": "rollout_trace", "host": "h",
                                                              "params": self.params}]))
        rt_write(self.f, [rt_contact("r%d" % i, "lock") for i in range(3)])
        self.assertEqual(self.cyc()[0]["rt"]["state"], "running")
        rt_write(self.f, [rt_score("r1", "lock", 1.0, True)])
        out, lines = self.cyc()
        self.assertEqual(out["rt"]["state"], "done")
        self.assertTrue(any("running -> done" in l for l in lines))
        self.assertLess(os.path.getsize(os.path.join(self.d, "status-h.json")), 5000)

    def test_done_file_and_state_survives_restart_without_double_counting(self):
        rt_write(self.f, [rt_contact("r1", "lock"), rt_contact("r2", "lock")])
        self.cyc()
        w2 = wd.Watchdog("h", os.path.join(self.d, "jobs.json"), self.d, env=self.env)
        out, lines = w2.cycle()
        self.assertEqual(out["rt"]["metrics"]["contacts"], 2)
        self.assertEqual(lines, [])
        rt_write(self.f, [rt_contact("r3", "lock")])
        self.assertEqual(w2.cycle()[0]["rt"]["metrics"]["contacts"], 3)

    def test_stalled_when_file_stops_growing(self):
        rt_write(self.f, [rt_contact("r1", "lock")])
        self.cyc()
        out, lines = self.cyc(advance=2000)
        self.assertEqual(out["rt"]["state"], "stalled")
        self.assertTrue(any(" BAD " in l for l in lines))


class FrameworkTests(Base):
    def test_only_own_host_jobs_and_files_and_limits(self):
        log = os.path.join(self.d, "g.log")
        write(log, "x\n")
        self.start({"id": "mine", "kind": "generic_log", "params": {"path": log}},
                   {"id": "theirs", "kind": "generic_log", "host": "other", "params": {"path": log}}, host="h")
        out, _ = self.cyc()
        self.assertEqual(list(out), ["mine"])
        st = os.path.join(self.d, "status-h.json")
        self.assertLess(os.path.getsize(st), 50000)
        self.assertEqual(load(st)["jobs"]["mine"]["state"], "running")
        self.assertFalse([f for f in os.listdir(self.d) if ".tmp" in f])  # atomic write leaves no temp files

    def test_state_survives_restart_without_repeating_events(self):
        log = os.path.join(self.d, "g.log")
        write(log, "x\n")
        self.start({"id": "j", "kind": "generic_log", "params": {"path": log}})
        self.assertEqual(len(self.cyc()[1]), 1)
        w2 = wd.Watchdog("h", os.path.join(self.d, "jobs.json"), self.d, env=self.env)
        self.assertEqual(w2.cycle()[1], [])

    def test_event_lines_short_and_rotation(self):
        log = os.path.join(self.d, "g.log")
        write(log, "ERROR " + "y" * 900 + "\n")
        self.start({"id": "j", "kind": "generic_log", "params": {"path": log, "bad_regex": "ERROR"}})
        self.cyc()
        text = read(os.path.join(self.d, "events-h.log")).splitlines()
        self.assertTrue(all(len(l) <= 300 for l in text))
        self.assertTrue(text[1].split()[1] == "BAD" and "bundle=" in text[1])
        write(os.path.join(self.d, "events-h.log"), "z" * 5100000)
        write(log, "ERROR again\n", "a")
        self.cyc()
        self.assertTrue(os.path.exists(os.path.join(self.d, "events-h.log.1")))

    def test_broken_evaluator_does_not_crash(self):
        self.start({"id": "j", "kind": "generic_log", "params": {}})  # missing "path"
        out, lines = self.cyc()
        self.assertEqual(out["j"]["state"], "unknown")
        self.assertIn("evaluator error", out["j"]["note"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
