import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import importlib.util
from contextlib import redirect_stdout
import io
MODULE_PATH = Path(__file__).resolve().parents[2] / 'scripts/analysis/verify_agent_delivery.py'
spec = importlib.util.spec_from_file_location('verify_agent_delivery', MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
verify, digest = module.verify, module.digest

class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);(self.root/'TASK.md').write_text('task A')
    def write(self,name,value):
        (self.root/name).write_text(json.dumps(value))
    def fixture(self):
        h=digest(self.root/'TASK.md');self.write('receipt.json',{'task_sha256':h})
        (self.root/'REPORT.md').write_text('actual report')
        r={'task_sha256':h,'status':'delivered','artifacts':[{'path':'REPORT.md','sha256':digest(self.root/'REPORT.md')}]}
        self.write('result.json',r);return r
    def test_missing_receipt_not_permission_diagnosis(self):
        self.assertEqual(verify(self.root)['state'],'no-receipt')
    def test_task_change_rejects_old_completion(self):
        self.fixture();(self.root/'TASK.md').write_text('task B')
        self.assertEqual(verify(self.root)['state'],'stale-receipt')
    def test_new_receipt_with_old_result(self):
        self.fixture();(self.root/'TASK.md').write_text('task B')
        self.write('receipt.json',{'task_sha256':digest(self.root/'TASK.md')})
        self.assertEqual(verify(self.root)['state'],'stale-result')
    def test_matching_result_does_not_accept_business_claim(self):
        self.fixture();r=verify(self.root)
        self.assertEqual(r['state'],'artifacts-verified');self.assertIn('pending',r['business_acceptance'])
    def test_changed_report_rejected(self):
        self.fixture();(self.root/'REPORT.md').write_text('edited')
        with self.assertRaisesRegex(ValueError,'bytes changed'):verify(self.root)
    def test_symlink_rejected(self):
        r=self.fixture();p=self.root/'REPORT.md';p.rename(self.root/'real.md');p.symlink_to('real.md')
        with self.assertRaisesRegex(ValueError,'symlink'):verify(self.root)
    def test_result_without_report_rejected(self):
        r=self.fixture();r['artifacts']=[{'path':'receipt.json','sha256':digest(self.root/'receipt.json')}];self.write('result.json',r)
        with self.assertRaisesRegex(ValueError,'report'):verify(self.root)
    def test_duplicate_artifact_rejected(self):
        r=self.fixture();r['artifacts']*=2;self.write('result.json',r)
        with self.assertRaisesRegex(ValueError,'unique'):verify(self.root)
    def test_duplicate_json_key_rejected(self):
        self.fixture();(self.root/'result.json').write_text('{"status":"delivered","status":"blocked"}')
        with self.assertRaisesRegex(ValueError,'duplicate'):verify(self.root)

    def test_task_is_a_valid_pinned_artifact(self):
        r = self.fixture()
        r['artifacts'].append({'path': 'TASK.md', 'sha256': digest(self.root/'TASK.md')})
        self.write('result.json', r)
        self.assertEqual(verify(self.root)['artifact_count'], 2)

    def test_result_self_reference_rejected(self):
        r = self.fixture()
        r['artifacts'].append({'path': 'result.json', 'sha256': digest(self.root/'result.json')})
        self.write('result.json', r)
        with self.assertRaisesRegex(ValueError, 'belong'): verify(self.root)

    def test_outside_artifact_rejected(self):
        r = self.fixture()
        r['artifacts'].append({'path': str(MODULE_PATH), 'sha256': digest(MODULE_PATH)})
        self.write('result.json', r)
        with self.assertRaisesRegex(ValueError, 'belong'): verify(self.root)

    def test_nested_symlink_rejected(self):
        r = self.fixture()
        (self.root/'sub').mkdir()
        (self.root/'sub/data.txt').write_text('data')
        (self.root/'linked').symlink_to('sub', target_is_directory=True)
        r['artifacts'].append({'path': 'linked/data.txt', 'sha256': digest(self.root/'sub/data.txt')})
        self.write('result.json', r)
        with self.assertRaisesRegex(ValueError, 'symlink'): verify(self.root)

    def test_missing_result_is_not_failure(self):
        self.fixture(); (self.root/'result.json').unlink()
        r = verify(self.root)
        self.assertEqual(r['state'], 'received-no-result')
        self.assertIn('inspect actual process', r['meaning'])

    def test_broken_receipt_symlink_is_invalid_not_missing(self):
        (self.root/'receipt.json').symlink_to('missing')
        with self.assertRaisesRegex(ValueError, 'symlink'): verify(self.root)

    def test_blocked_is_not_delivered(self):
        r = self.fixture(); r['status'] = 'blocked'; self.write('result.json', r)
        self.assertEqual(verify(self.root)['state'], 'blocked')

    def test_cli_json_and_exit_codes(self):
        for state, status, expected in [('missing', None, 3), ('delivered', 'delivered', 0), ('blocked', 'blocked', 4), ('invalid', 'unknown', 2)]:
            if status:
                r = self.fixture(); r['status'] = status; self.write('result.json', r)
            output = io.StringIO()
            with redirect_stdout(output): code = module.main([str(self.root)])
            self.assertEqual(code, expected, state)
            self.assertFalse(json.loads(output.getvalue())['automatic_retry'])

    def test_non_object_receipt(self):
        self.write('receipt.json', [])
        with self.assertRaisesRegex(ValueError, 'object'): verify(self.root)

    def test_parent_traversal_rejected(self):
        r = self.fixture(); r['artifacts'][0]['path'] = 'sub/../REPORT.md'; self.write('result.json', r)
        with self.assertRaisesRegex(ValueError, 'traversal'): verify(self.root)

if __name__ == '__main__': unittest.main()
