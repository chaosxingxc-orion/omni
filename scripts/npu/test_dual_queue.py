import json
import os
from pathlib import Path
import tempfile
import unittest

import dual_queue


class DualQueueTests(unittest.TestCase):
    def queue(self, root, name, jobs):
        path = root / (name + ".json")
        path.write_text(json.dumps({"schema": "saea-card-queue-v1", "id": name, "jobs": jobs}))
        return path

    def job(self, name, **extra):
        value = {"id": name, "kind": "test", "cmd": ["true"], "cwd": ".", "log": "/tmp/%s.log" % name,
                 "done_marker": None, "timeout_s": 10, "on_fail": "continue"}
        value.update(extra)
        return value

    def test_cpu_jobs_are_ungated_and_queue_dirs_are_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npu = self.queue(root, "npu", [self.job("experiment", requires_cards_free=True, cards_policy="no_other")])
            cpu = self.queue(root, "cpu", [self.job("analysis")])
            config = root / "dual.json"
            config.write_text(json.dumps({"schema": "saea-dual-queue-v1", "npu_queue": str(npu), "cpu_queue": str(cpu),
                                          "npu_queue_dir": str(root / "npu-state"), "cpu_queue_dir": str(root / "cpu-state"),
                                          "guardian": str(root / "guardian.py")}))
            (root / "guardian.py").write_text("# fixture")
            value = dual_queue.validate_config(config)
            self.assertNotEqual(value["npu_queue_dir"], value["cpu_queue_dir"])

    def test_cpu_card_gate_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npu = self.queue(root, "npu", [self.job("experiment", requires_cards_free=True, cards_policy="no_other")])
            cpu = self.queue(root, "cpu", [self.job("analysis", requires_cards_free=True)])
            config = root / "dual.json"
            config.write_text(json.dumps({"schema": "saea-dual-queue-v1", "npu_queue": str(npu), "cpu_queue": str(cpu),
                                          "guardian": str(root / "guardian.py")}))
            (root / "guardian.py").write_text("# fixture")
            with self.assertRaises(dual_queue.DualQueueError): dual_queue.validate_config(config)

    def test_existing_terminal_queue_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npu = self.queue(root, "npu", [self.job("experiment", requires_cards_free=True, cards_policy="no_other")])
            cpu = self.queue(root, "cpu", [self.job("analysis")])
            for name in ("npu-state", "cpu-state"):
                d = root / name; d.mkdir(); (d / "state.json").write_text(json.dumps({"state": "done"}))
            config = root / "dual.json"
            config.write_text(json.dumps({"schema": "saea-dual-queue-v1", "npu_queue": str(npu), "cpu_queue": str(cpu),
                                          "npu_queue_dir": str(root / "npu-state"), "cpu_queue_dir": str(root / "cpu-state"),
                                          "guardian": str(root / "guardian.py")}))
            (root / "guardian.py").write_text("# fixture")
            out = dual_queue.start(config, root / "dual-state")
            self.assertEqual(out["queues"]["npu"]["state"], "terminal")
            self.assertEqual(out["queues"]["cpu"]["state"], "terminal")

    def test_npu_queue_must_be_card_gated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npu = self.queue(root, "npu", [self.job("experiment")])
            cpu = self.queue(root, "cpu", [self.job("analysis")])
            config = root / "dual.json"
            config.write_text(json.dumps({"schema": "saea-dual-queue-v1", "npu_queue": str(npu), "cpu_queue": str(cpu),
                                          "guardian": str(root / "guardian.py")}))
            (root / "guardian.py").write_text("# fixture")
            with self.assertRaises(dual_queue.DualQueueError): dual_queue.validate_config(config)


if __name__ == "__main__":
    unittest.main()
