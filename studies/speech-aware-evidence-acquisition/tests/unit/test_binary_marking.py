"""Positional protocol and failure semantics; no model accuracy fixtures."""
import copy,itertools,json,unittest
from test_slot_stages import knowledge,query,CONFIG
from speech_aware_evidence_acquisition.rails.slot_stages import StageRails
from speech_aware_evidence_acquisition.rails.configured_s1 import schema_change,protocol_wire_change
class BinaryTest(unittest.TestCase):
 def setUp(self):
  self.base={**CONFIG,'s1_candidate_presentation':'type_marking_v1','s1_instruction':'frozen semantics; JSON booleans'}
  self.policy={**self.base,'s1_candidate_presentation':'binary_marking_v1','s1_instruction':'frozen semantics; ordered binary marks'}
  self.rails=StageRails(knowledge(),self.policy);self.packet=self.rails.before_model_call('S1',query(),'f1')
 def reply(self,text,finish='stop'):return self.rails.after_model_call(self.packet,dict(text=text,finish_reason=finish))
 def test_mapping_and_all_assignments(self):
  for marks in itertools.product('01',repeat=2):
   result=self.reply(' '.join(marks));self.assertEqual(result['selected_types'],[t for t,b in zip(['person','time'],marks) if b=='1']);self.assertTrue(result['status'].startswith('valid_'))
  self.assertEqual(self.reply('0 0')['status'],'valid_empty');self.assertEqual(self.reply(' \n1 0\t')['selected_types'],['person'])
 def test_no_silent_false_or_repairs(self):
  for value in ['','0','0 1 0','01','0  1','0\t1','0\n1','1,0','true false','0 ２','```\n0 1\n```','"0 1"','{"bits":"0 1"}','0 1 because',None,10]:
   with self.subTest(value=value):self.assertEqual(self.reply(value)['status'],'protocol_error')
 def test_truncation_is_not_empty(self):
  for finish in ['length',None,'error']:
   self.assertEqual(self.reply('0 0',finish)['status'],'incomplete')
 def test_wire_order_text_regex_and_s2_compatibility(self):
  wire=self.packet['wire'];body=json.loads(wire['messages'][1]['content'][0]['text']);self.assertEqual(body['binary_marking']['ordered_types'],['person','time']);self.assertEqual(wire['response_format'],{'type':'text'});self.assertEqual(wire['structured_outputs'],{'regex':'[01]( [01]){1}'})
  self.assertEqual(self.rails.before_model_call('S2',query(),'time')['wire'],StageRails(knowledge(),self.base).before_model_call('S2',query(),'time')['wire'])
 def test_configured_protocol_rejects_extra_semantic_edits(self):
  before=StageRails(knowledge(),self.base).before_model_call('S1',query(),'f1')['wire'];after=self.packet['wire'];changed=['s1_candidate_presentation','s1_instruction']
  schema_change(before['response_format'],after['response_format'],self.base,self.policy,changed);protocol_wire_change(before,after,self.base,self.policy,changed)
  bad=copy.deepcopy(after);body=json.loads(bad['messages'][1]['content'][0]['text']);body['cards'][0]['definition']='drift';bad['messages'][1]['content'][0]['text']=json.dumps(body)
  with self.assertRaises(ValueError):protocol_wire_change(before,bad,self.base,self.policy,changed)
  bad=copy.deepcopy(after);bad['structured_outputs']['regex']='[01]+'
  with self.assertRaises(ValueError):protocol_wire_change(before,bad,self.base,self.policy,changed)
 def test_variable_fold_width_and_large_type_inventory(self):
  for n in [1,4,5,6,55,56]:
   k=knowledge();card=copy.deepcopy(k['cards']['time']);k['cards']={str(i):{**copy.deepcopy(card),'type':str(i)} for i in range(n)};k['folds']=[dict(fold_id='dynamic',types=list(k['cards']))]
   rails=StageRails(k,self.policy);packet=rails.before_model_call('S1',query(),'dynamic')
   self.assertEqual(packet['wire']['structured_outputs']['regex'],'[01]( [01]){'+str(n-1)+'}')
   marks=['1' if i%2 else '0' for i in range(n)]
   result=rails.after_model_call(packet,dict(text=' '.join(marks),finish_reason='stop'))
   self.assertEqual(result['selected_types'],[str(i) for i in range(n) if i%2])
   for malformed in [' '.join(marks+['0']),' '.join(marks[:-1])]:
    self.assertEqual(rails.after_model_call(packet,dict(text=malformed,finish_reason='stop'))['status'],'protocol_error')
 def test_u_list_parent_protocol_keeps_supplied_evidence(self):
  base={**CONFIG,'s1_instruction':'frozen U semantics; list format'}
  policy={**base,'s1_candidate_presentation':'binary_marking_v1','s1_instruction':'frozen U semantics; mark format'}
  a=StageRails(knowledge(),base).before_model_call('S1',query(),'f1')['wire']
  b=StageRails(knowledge(),policy).before_model_call('S1',query(),'f1')['wire']
  for wire in [a,b]:
   part=wire['messages'][1]['content'][0];body=json.loads(part['text']);body['retrieved_examples']=[{'frozen':'U supply'}];part['text']=json.dumps(body,sort_keys=True,separators=(',',':'))
  protocol_wire_change(a,b,base,policy,['s1_candidate_presentation','s1_instruction'])
  bad=copy.deepcopy(b);part=bad['messages'][1]['content'][0];body=json.loads(part['text']);body['retrieved_examples']=[];part['text']=json.dumps(body)
  with self.assertRaises(ValueError):protocol_wire_change(a,bad,base,policy,['s1_candidate_presentation','s1_instruction'])
 def test_legacy_and_json_bool_remain(self):
  for policy,text in [(CONFIG,'{"slot_types":["time"]}'),(self.base,'{"slot_type_presence":{"person":false,"time":true}}')]:
   rails=StageRails(knowledge(),policy);p=rails.before_model_call('S1',query(),'f1');self.assertEqual(rails.after_model_call(p,dict(text=text,finish_reason='stop'))['selected_types'],['time'])
if __name__=='__main__':unittest.main(verbosity=2)
