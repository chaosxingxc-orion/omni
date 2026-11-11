"""Configuration plumbing: real consumer reuse and fail-closed cache requirements."""
import unittest
from unittest.mock import patch
from speech_aware_evidence_acquisition.rails import configured_s1 as s
from pathlib import Path
class SupplyConfigTests(unittest.TestCase):
 def test_no_consumer_preserves_exact_constructor(self):
  with patch('speech_aware_evidence_acquisition.rails.slot_stages.StageRails') as rails:
   s.build_rails('/root',{}, {'cards':{}}, {'model':'m'})
   rails.assert_called_once_with({'cards':{}},{'model':'m'})
 def test_consumer_load_once_and_explicit_root(self):
  with patch('speech_aware_evidence_acquisition.rails.slot_stages.StageRails') as rails,patch.object(s,'pinned',return_value=Path('/cache')),patch.object(s,'read',return_value={'cache_id':'c'}) as rd:
   s.build_rails('/root',dict(retrieval_cache={'path':'cache','sha256':'x'},provenance_root='sources'),{},dict(s1_consumption={}))
   self.assertEqual(rd.call_count,1)
   self.assertEqual(rails.call_args.kwargs,dict(retrieval_cache={'cache_id':'c'},provenance_root=Path('/root/sources')))
 def test_missing_cache_or_root_rejected(self):
  for task in [{},{'retrieval_cache':{}},{'provenance_root':'.'}]:
   with self.assertRaises(ValueError):s.build_rails('/root',task,{},dict(s1_consumption={}))
 def test_supply_without_consumer_rejected(self):
  with self.assertRaises(ValueError):s.build_rails('/root',dict(retrieval_cache={}),{}, {})
if __name__=='__main__':unittest.main()
