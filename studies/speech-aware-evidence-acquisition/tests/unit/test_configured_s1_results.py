import base64
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from speech_aware_evidence_acquisition.rails import configured_s1 as shared
from speech_aware_evidence_acquisition.rails import configured_s1_entry as entry
from speech_aware_evidence_acquisition.rails import configured_s1_results as results
from speech_aware_evidence_acquisition.rails.configured_s1_host import readonly_dependency_mounts


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, name, value):
        path = self.root / name
        shared.save(path, value)
        return path

    def journal(self):
        run = self.root / 'job/run'
        self.write('job/runtime-config.json', dict(run_id='run', attempt_id='attempt'))
        folder = run / 'contact-journals/run/attempt'
        raw = shared.canonical(dict(choices=[dict(finish_reason='stop')], usage=dict(total_tokens=12)))
        self.write('job/run/contact-journals/run/attempt/request.intent.json', dict(identity={'version': 1}, request_id='request'))
        self.write('job/run/contact-journals/run/attempt/request.response.json', dict(identity={'version': 1}, request_id='request',
                   response_b64=base64.b64encode(raw).decode(), response_sha256=shared.sha(raw), latency_seconds=2))
        return dict(lane='A', rows=str(run / 'rows'), total_requests=1, cards=4, card_hours=.1, wall_seconds=90, status='COMPLETE'), folder

    def test_real_journal_binding_unknown_and_cost(self):
        lane, folder = self.journal()
        actual = results.census([lane], 1, 1)
        self.assertEqual(actual['states'], dict(observed=1, unknown=0, absent=0))
        self.assertEqual(actual['usage'], dict(total_tokens=12))
        (folder / 'request.response.json').unlink()
        self.assertEqual(results.census([lane], 1, 1)['states']['unknown'], 1)

    def test_corrupt_response_refused(self):
        lane, folder = self.journal()
        response = shared.read(folder / 'request.response.json')
        response['response_sha256'] = 'wrong'
        shared.save(folder / 'request.response.json', response)
        with self.assertRaisesRegex(ValueError, 'binding'):
            results.census([lane], 1, 1)

    def test_parent_success_and_rejections_never_charged_current(self):
        lane, _ = self.journal()
        current = results.census([lane], 1, 1)
        parent = self.write('parent.json', dict(states=dict(observed=66810), card_hours=4.96, usage=dict(total_tokens=34696975),
                            prior_attempts=[dict(requests=30, card_hours=.1)], all_attempt_requests=66840, all_attempt_card_hours=5.06))
        failed = self.write('failed.json', dict(model_calls=0, total_bind_wall_seconds=5.36))
        before = parent.read_bytes()
        cost = results.scope_costs(current, parent, failed)
        self.assertEqual(cost['all_attempt_requests'], 1)
        self.assertEqual(cost['all_attempt_card_hours'], .1)
        self.assertEqual(cost['parent_references'][0]['successful_requests'], 66810)
        self.assertEqual(cost['lineage_known_totals']['requests'], 66841)
        self.assertEqual(cost['pre_model_failures']['model_calls'], 0)
        self.assertEqual(parent.read_bytes(), before)

    def interface_inputs(self):
        lane, _ = self.journal()
        cost = results.census([lane], 1, 1)
        term = self.write('RESULTS.json', cost)
        summary = self.write('SUMMARY.json', dict(status='PARTIAL', cost=dict(runtime_receipt=cost)))
        successor = dict(summary=str(summary), scores=str(self.write('scores.jsonl', [])),
                         predictions=str(self.write('predictions.jsonl', [])),
                         manifests={l: str(self.write(l + '.jsonl', [])) for l in 'ABC'})
        return successor, term

    def test_interface_pins_all_inputs_accepts_incomplete_protocol(self):
        successor, term = self.interface_inputs()
        interface = results.result_interface(successor, term, True)
        self.assertEqual(interface['status'], 'READY')
        self.assertEqual(len(interface['pins']), 7)
        for path, digest in interface['pins'].items():
            self.assertEqual(shared.file_sha(path), digest)

    def test_unknown_not_ready_and_score_mismatch_refused(self):
        successor, term = self.interface_inputs()
        cost = shared.read(term)
        cost['states']['unknown'] = 1
        shared.save(term, cost)
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            results.result_interface(successor, term, True)
        cost['states']['unknown'] = 0
        cost['states']['observed'] = 0
        shared.save(term, cost)
        with self.assertRaisesRegex(ValueError, 'inventory'):
            results.result_interface(successor, term, True)

    def test_failed_terminal_with_known_inventory_can_be_analyzed(self):
        successor, term = self.interface_inputs()
        cost = shared.read(term)
        cost['lanes'][0]['status'] = 'FAILED'
        shared.save(term, cost)
        self.write('SUMMARY.json', dict(cost=dict(runtime_receipt=cost)))
        self.assertEqual(results.result_interface(successor, term, True)['terminal']['status'], 'FAILED_TERMINAL')

    def test_host_exception_persists_structured_failure(self):
        with patch.object(entry, '_host', side_effect=RuntimeError('missing model resource')):
            with self.assertRaisesRegex(RuntimeError, 'missing model'):
                entry.host(self.root)
        self.assertEqual(shared.read(self.root / 'ERROR.json')['status'], 'FAILED_HOST')

    def test_mount_missing_and_shadow_fail_before_launch(self):
        with self.assertRaises(ValueError):
            readonly_dependency_mounts([dict(source=str(self.root / 'missing'), target='/model')], [])
        with self.assertRaisesRegex(ValueError, 'shadows'):
            readonly_dependency_mounts([dict(source=str(self.root), target='/workspace')], ['/workspace'])
        mounts = readonly_dependency_mounts([dict(source=str(self.root), target='/model')], ['/workspace'])
        self.assertEqual(mounts, ['--mount', f'type=bind,src={self.root},dst=/model,readonly'])
        with self.assertRaisesRegex(ValueError, 'shadows'):
            readonly_dependency_mounts([dict(source=str(self.root), target='/model')] * 2, [])

    def test_signal_identity_mismatch_is_refused(self):
        receipt = self.write('watcher.json', dict(pid=os.getpid(),
                            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                            proc_start_ticks='wrong', argv=['wrong']))
        with self.assertRaisesRegex(ValueError, 'identity'):
            results.notify_watcher(receipt)

    def test_signal_to_verified_process(self):
        process = subprocess.Popen([sys.executable, '-c',
            "import signal,time; signal.signal(signal.SIGUSR1,lambda *_:exit(0)); print('READY',flush=True); time.sleep(20)"], stdout=subprocess.PIPE, text=True)
        self.addCleanup(lambda: process.poll() is None and process.kill())
        self.assertEqual(process.stdout.readline().strip(), 'READY')
        stat = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()
        receipt = self.write('watcher.json', dict(pid=process.pid, proc_start_ticks=stat[19],
                            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), argv=process.args))
        self.assertEqual(results.notify_watcher(receipt)['status'], 'SIGUSR1_SENT_IDENTITY_VERIFIED')
        self.assertEqual(process.wait(timeout=5), 0)
        process.stdout.close()
