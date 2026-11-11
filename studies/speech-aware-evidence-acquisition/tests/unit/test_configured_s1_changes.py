import copy,unittest
from speech_aware_evidence_acquisition.rails import configured_s1 as s
class ScientificContractTests(unittest.TestCase):
 def test_legacy_instruction_only_default(self):
  self.assertEqual(s.scientific_changes({},dict(s1_instruction='old'),dict(s1_instruction='new')),['s1_instruction'])
 def test_none_contract_rejects_extra_or_missing_change(self):
  task=dict(allowed_changes=['s1_none_label']);base=dict(s1_instruction='parent',seed=0)
  self.assertEqual(s.scientific_changes(task,base,dict(base,s1_none_label='NONE_OF_FOLD')),['s1_none_label'])
  for current in [base,dict(base,s1_none_label='NONE_OF_FOLD',seed=1),dict(s1_none_label='NONE_OF_FOLD',seed=0)]:
   with self.assertRaises(ValueError):s.scientific_changes(task,base,current)
 def test_none_schema_only_enum_may_change(self):
  old=dict(json_schema=dict(schema=dict(properties=dict(slot_types=dict(type='array',items=dict(type='string',enum=['real']))))))
  new=copy.deepcopy(old);new['json_schema']['schema']['properties']['slot_types']['items']['enum'].append('NONE_OF_FOLD')
  s.schema_change(old,new,{},dict(s1_none_label='NONE_OF_FOLD'),['s1_none_label'])
  new['json_schema']['schema']['properties']['slot_types']['uniqueItems']=True
  with self.assertRaises(ValueError):s.schema_change(old,new,{},dict(s1_none_label='NONE_OF_FOLD'),['s1_none_label'])
