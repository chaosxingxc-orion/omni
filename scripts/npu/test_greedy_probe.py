#!/usr/bin/env python3
"""Unit tests for greedy_probe compare logic (synthetic fixtures, CPU only) plus the E1 reproduction check when files exist."""
import json
import os
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import greedy_probe as gp  # noqa: E402

LOGS = Path(__file__).resolve().parent.parent / "logs"


def res(text="abc", ids=(1, 2, 3), lps=(-0.1, -0.2, -0.3), top=None, finish="length", status="ok"):
    r = {"status": status, "reasoning": text, "content": "", "finish_reason": finish, "completion_tokens": len(lps),
         "logprobs": list(lps), "token_ids": list(ids) if ids is not None else None}
    if top is not None:
        r["top_logprobs"] = top
    return r


def cand(l1, l2):
    return [{"token": "token_id:1", "logprob": l1}, {"token": "token_id:2", "logprob": l2}]


class CompareTests(unittest.TestCase):
    def test_all_identical(self):
        a = [res(), res("x", (4,), (-1.0,))]
        c = gp.compare_runs(a, json.loads(json.dumps(a)))
        self.assertEqual((c["byte_identical"], c["mismatched_indices"], c["max_abs_logprob_diff"]), (2, [], 0.0))

    def test_token_divergence_and_gap(self):
        a = [res(), res("q", (7, 8, 9), (-0.1, -0.5, -0.6), top=[cand(-0.1, -3), cand(-0.5, -0.52), cand(-0.6, -2)])]
        b = [res(), res("r", (7, 9, 9), (-0.1, -0.52, -0.7), top=[cand(-0.1, -3), cand(-0.52, -0.5), cand(-0.7, -2)])]
        c = gp.compare_runs(a, b)
        self.assertEqual((c["byte_identical"], c["mismatched_indices"]), (1, [1]))
        m = c["mismatches"]["1"]
        self.assertEqual(m["first_divergent_token"], 1)
        self.assertEqual(m["divergence_basis"], "token_ids")
        self.assertEqual(m["tokens_at_divergence"], [8, 9])
        self.assertAlmostEqual(m["top2_gap_at_divergence"][0], 0.02)
        self.assertAlmostEqual(m["top2_gap_at_divergence"][1], 0.02)
        self.assertAlmostEqual(m["max_abs_logprob_diff_before_divergence"], 0.0)
        self.assertAlmostEqual(c["max_abs_logprob_diff"], 0.1)
        self.assertEqual(m["first_divergent_char"], 0)

    def test_length_divergence_is_prefix(self):
        a, b = [res(ids=(1, 2, 3))], [res(text="abd", ids=(1, 2))]
        b[0]["logprobs"] = [-0.1, -0.2]
        m = gp.compare_runs(a, b)["mismatches"]["0"]
        self.assertEqual(m["first_divergent_token"], 2)
        self.assertIsNone(gp.compare_runs(a, b)["max_abs_logprob_diff"])  # unequal lengths: legacy rule skips

    def test_error_status_is_a_mismatch(self):
        a, b = [res()], [res(status="error")]
        self.assertEqual(gp.compare_runs(a, b)["mismatched_indices"], [0])

    def test_legacy_logprob_proxy(self):
        a = [res(ids=None, lps=(-0.1, -0.2, -0.3))]
        b = [res(text="abd", ids=None, lps=(-0.1, -0.2005, -1.3))]
        for r in a + b:
            r["token_ids"] = None
        m = gp.compare_runs(a, b)["mismatches"]["0"]
        self.assertEqual(m["first_divergent_token"], 2)
        self.assertTrue(m["divergence_basis"].startswith("sampled_logprob_proxy"))
        self.assertIsNone(m["top2_gap_at_divergence"])
        self.assertAlmostEqual(m["max_abs_logprob_diff_before_divergence"], 0.0005, places=6)

    def test_matrix_and_classify(self):
        same = [res(), res("x", (4,), (-1.0,))]
        diff = [res(), res("y", (5,), (-1.5,))]
        runs = {"S1": same, "S2": same, "S3": same, "B1": same, "B2": diff, "B3": same}
        modes = {k: ("sequential" if k[0] == "S" else "burst") for k in runs}
        m = gp.matrix(runs, modes)
        self.assertEqual(len(m["pairs"]), 15)
        groups = {p["group"] for p in m["pairs"]}
        self.assertEqual(groups, {"within-sequential", "within-burst", "across-modes"})
        self.assertTrue(gp.classify(m["pairs"]).startswith("batch-composition"))
        runs["S2"] = diff
        m = gp.matrix(runs, modes)
        self.assertTrue(gp.classify(m["pairs"]).startswith("kernel-or-cache"))
        runs = {k: same for k in runs}
        self.assertTrue(gp.classify(gp.matrix(runs, modes)["pairs"]).startswith("deterministic"))

    def test_post_parses_token_ids(self):
        # exercise the response parsing path with a local stub server
        import http.server
        import threading
        payload = {"choices": [{"message": {"reasoning_content": "r", "content": "c"}, "finish_reason": "length",
                                "logprobs": {"content": [{"token": "token_id:11", "logprob": -0.5,
                                                          "top_logprobs": [{"token": "token_id:11", "logprob": -0.5},
                                                                           {"token": "token_id:12", "logprob": -0.9},
                                                                           {"token": "token_id:13", "logprob": -2}]}]}}],
                   "usage": {"completion_tokens": 1}}

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            r = gp.post(f"http://127.0.0.1:{srv.server_address[1]}", {"x": 1})
            self.assertEqual(r["token_ids"], [11])
            self.assertEqual(len(r["top_logprobs"][0]), 2)
            out = gp.send(f"http://127.0.0.1:{srv.server_address[1]}", [{"a": 1}, {"a": 2}, {"a": 3}], "burst", 2, True)
            self.assertEqual([x["index"] for x in out], [0, 1, 2])
            out = gp.send(f"http://127.0.0.1:{srv.server_address[1]}", [{"a": 1}, {"a": 2}], "sequential", 2, True)
            self.assertTrue(all(x["status"] == "ok" for x in out))
        finally:
            srv.shutdown()


@unittest.skipUnless((LOGS / "prelim-E1-eq-a-20261029T111900Z.json").exists(), "E1 probe outputs not present")
class ReproduceE1(unittest.TestCase):
    def test_9_of_16_and_1_97(self):
        a = gp.load_results(str(LOGS / "prelim-E1-eq-a-20261029T111900Z.json"))
        b = gp.load_results(str(LOGS / "prelim-E1-eq-b-20261029T111900Z.json"))
        c = gp.compare_runs(a, b)
        self.assertEqual(c["byte_identical"], 9)
        self.assertEqual(c["mismatched_indices"], [0, 2, 3, 4, 10, 12, 14])
        self.assertAlmostEqual(c["max_abs_logprob_diff"], 1.9739304929971695, places=9)

    def test_e2_matches_e1_a(self):
        a = gp.load_results(str(LOGS / "prelim-E1-eq-a-20261029T111900Z.json"))
        e2 = gp.load_results(str(LOGS / "prelim-E2-eq-shm-20261029T111900Z.json"))
        self.assertEqual(gp.compare_runs(a, e2)["byte_identical"], 16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
