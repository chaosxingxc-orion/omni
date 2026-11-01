import unittest

import card_queue


class QueueProgressTests(unittest.TestCase):
    def test_summary_uses_durable_state_without_probe(self):
        state = {
            "schema": card_queue.STATE_SCHEMA,
            "queue_id": "q1",
            "state": "running",
            "started": "2026-01-01T00:00:00Z",
            "current": "b",
            "waiting": {"since": "2026-01-01T00:00:30Z"},
            "jobs": {
                "a": {"status": "done", "started": "2026-01-01T00:00:00Z",
                      "finished": "2026-01-01T00:00:10Z"},
                "b": {"status": "waiting"},
                "c": {"status": "pending"},
            },
        }
        summary = card_queue.progress_summary(state, now=60)
        self.assertEqual(summary["terminal_jobs"], 1)
        self.assertEqual(summary["remaining_jobs"], 2)
        self.assertAlmostEqual(summary["throughput_jobs_per_second"], 1 / 60)
        self.assertAlmostEqual(summary["estimated_remaining_seconds"], 120)
        self.assertAlmostEqual(summary["waiting_seconds"], 30)

    def test_terminal_queue_does_not_invent_eta(self):
        state = {"queue_id": "q", "state": "stopped", "jobs": {
            "a": {"status": "failed"}, "b": {"status": "pending"}}}
        summary = card_queue.progress_summary(state, now=1)
        self.assertIsNone(summary["estimated_remaining_seconds"])

    def test_bad_state_fails_closed(self):
        with self.assertRaises(ValueError):
            card_queue.progress_summary({"jobs": []})


if __name__ == "__main__":
    unittest.main()
