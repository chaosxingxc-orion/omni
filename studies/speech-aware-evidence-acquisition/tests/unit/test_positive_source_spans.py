"""Reference annotation trust and exact parent-wire compatibility."""
import copy,json,tempfile,unittest
from pathlib import Path
from speech_aware_evidence_acquisition.rails.slot_stage_source_spans import PositiveSourceSpanLayout,strip_source_span_layout,SPAN_SCOPE
from speech_aware_evidence_acquisition.rails.slot_stages import StageRails,canonical
from speech_aware_evidence_acquisition.rails import configured_s1 as s
from test_slot_stages import knowledge,query,CONFIG

class SpanTests(unittest.TestCase):
 def setUp(self):
  from speech_aware_evidence_acquisition.rails.slot_stage_selector import S1KnowledgeConsumer
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.row=dict(slurp_id=1,sentence='call amy tomorrow',tokens=[dict(surface=t,id=i) for i,t in enumerate(['call','amy','tomorrow'])],entities=[dict(type='person',span=[1]),dict(type='date',span=[2])])
  self.k=knowledge();self.k['source_components']={'1':'source-A'}
  self.k['cards']['person']['examples']=[dict(query=self.row['sentence'],source_id=1,component='source-A',values=['amy'])]
  self.c=S1KnowledgeConsumer(self.k,dict(schema='s1-consumption-policy-v1',max_examples=4,max_example_bytes=4096,max_rules=8,max_rule_bytes=4096),provenance_root=self.root)
  self.c.bank={'1':dict(query=self.row['sentence'])}
  self.body=dict(cards=[dict(type='person',examples=[])],retrieved_examples=[dict(query=self.row['sentence'],observed_support_types=['person'],annotation_scope='unlisted not negative')])
  self.receipt=dict(retrieval=dict(provenance=[dict(source_id=1,source_component='source-A',source_refs=[])]))
 def layout(self,rows=None):
  s.save(self.root/'train.json',rows or [self.row]);self.c._source_documents={}
  cfg=dict(layout='positive_source_spans_v1',annotation_source=dict(path='train.json',sha256=s.file_sha(self.root/'train.json')))
  return PositiveSourceSpanLayout(self.k,cfg,self.c)
 def apply(self,layout=None):
  body=copy.deepcopy(self.body);receipt=copy.deepcopy(self.receipt);(layout or self.layout()).apply(body,receipt,'target','target-id');return body,receipt
 def test_visible_positive_only_and_no_metadata_in_prompt(self):
  body,receipt=self.apply();self.assertEqual(body['retrieved_examples'][0]['positive_source_spans'],{'person':['amy']})
  text=json.dumps(body);self.assertNotIn('entities',text);self.assertNotIn('source-A',text);self.assertNotIn('date',text)
  refs=receipt['positive_source_spans']['examples'][0]['annotation_refs'];self.assertEqual(refs[0]['pointer'],'/0');self.assertEqual(refs[0]['source_component'],'source-A')
 def test_query_source_and_explicit_cached_exclusions(self):
  for mode in ['blocked','query','cache']:
   layout=self.layout();body=copy.deepcopy(self.body);receipt=copy.deepcopy(self.receipt)
   self.c.blocked={'source-A'} if mode=='blocked' else set();self.c.queries={'target-id':dict(excluded_source_components=['source-A'])} if mode=='cache' else {}
   layout.apply(body,receipt,'source-A' if mode=='query' else 'target','target-id');self.assertEqual(body,self.body)
 def test_missing_invalid_and_ambiguous_keep_original(self):
  for mutate in [lambda r:r.update(entities=[]),lambda r:r['entities'][0].update(span=[99]),lambda r:r['entities'][0].update(span=[True]),lambda r:r['tokens'][1].update(surface='other'),lambda r:r['entities'][0].update(span=[1,1]),lambda r:r['tokens'][0].update(id=4)]:
   row=copy.deepcopy(self.row);mutate(row);self.assertEqual(self.apply(self.layout([row]))[0],self.body)
  row=copy.deepcopy(self.row);row['entities'][0]['span']=[0]
  self.assertEqual(self.apply(self.layout([self.row,row]))[0],self.body)
  row=copy.deepcopy(self.row);row['slurp_id']=2
  self.assertEqual(self.apply(self.layout([row]))[0],self.body)
 def test_mismatched_component_and_missing_source_keep_original(self):
  self.k['source_components']['1']='different';self.assertEqual(self.apply()[0],self.body)
  self.receipt['retrieval']['provenance']=[];self.assertEqual(self.apply()[0],self.body)
 def test_static_positive_and_merged_supports(self):
  self.body['cards'][0]['examples']=[dict(query=self.row['sentence'],supports_type='person')];self.body['retrieved_examples']=[]
  body,_=self.apply();self.assertEqual(body['cards'][0]['examples'][0]['positive_source_spans'],{'person':['amy']})
 def test_unknown_values_unmarked_roles_not_negative(self):
  body,_=self.apply();self.assertIn('unlabelled, not negative',body['positive_source_span_scope']);self.assertIn('unknown or unrecoverable',SPAN_SCOPE);self.assertIn('matching these reference words is not required',SPAN_SCOPE)
  self.assertNotIn('negative_types',json.dumps(body))
 def test_binary_wire_inverse_and_semantic_drift_rejected(self):
  base={**CONFIG,'s1_candidate_presentation':'binary_marking_v1'};policy={**base,'s1_reference_layout':dict(layout='positive_source_spans_v1')}
  wire=StageRails(knowledge(),base).before_model_call('S1',query(),'f1')['wire'];after=copy.deepcopy(wire)
  part=after['messages'][1]['content'][0];part['text']=canonical(self.apply()[0]).decode()
  before=copy.deepcopy(wire);before['messages'][1]['content'][0]['text']=canonical(self.body).decode()
  self.assertEqual(strip_source_span_layout(after),before);s.schema_change(before['response_format'],after['response_format'],base,policy,['s1_reference_layout']);s.protocol_wire_change(before,after,base,policy,['s1_reference_layout'])
  bad=copy.deepcopy(after);bad['max_tokens']+=1
  with self.assertRaises(ValueError):s.protocol_wire_change(before,bad,base,policy,['s1_reference_layout'])
  body=self.apply()[0];body['retrieved_examples'][0]['positive_source_spans']['date']=['tomorrow'];after['messages'][1]['content'][0]['text']=canonical(body).decode()
  with self.assertRaises(ValueError):strip_source_span_layout(after)
 def test_default_wire_and_s2_unchanged(self):
  base={**CONFIG,'s1_candidate_presentation':'binary_marking_v1'};a=StageRails(knowledge(),base);b=StageRails(knowledge(),{**base,'s1_reference_layout':None})
  for stage,branch in [('S1','f1'),('S2','time')]:self.assertEqual(a.before_model_call(stage,query(),branch)['wire'],b.before_model_call(stage,query(),branch)['wire'])
 def test_annotation_pin_failure_and_traversal(self):
  with self.assertRaises((ValueError,FileNotFoundError)):PositiveSourceSpanLayout(self.k,dict(layout='positive_source_spans_v1',annotation_source=dict(path='../outside',sha256='bad')),self.c)
 def test_no_target_gold_api(self):
  from dataclasses import fields
  from speech_aware_evidence_acquisition.rails.slot_stages import Query
  self.assertEqual({f.name for f in fields(Query)},{'sample_id','audio','source_component','audio_sha256'})
if __name__=='__main__':unittest.main(verbosity=2)
