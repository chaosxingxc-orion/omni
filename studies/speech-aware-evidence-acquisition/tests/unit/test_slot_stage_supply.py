"""Source-bound optional S1 supply compiler; no model or query-label inputs."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from speech_aware_evidence_acquisition.rails.slot_stage_knowledge import build_s1_supply, source_document, digest


class SupplyTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.rows=[dict(slurp_id=1,sentence='name at noon',entities=[dict(type='person')],recordings=[dict(file='a.flac')]),
                   dict(slurp_id=2,sentence='blocked sentence',entities=[dict(type='time')],recordings=[])]
        self.text='Slot person.\nExtraction cues:\n- A name.\n- An adjacent time can coexist.\nMissing behavior: Omit a missing span.\nAmbiguity behavior: Prefer person instead of time.\nLimitations:\n- source scope'
        self.bundle=dict(snapshot=dict(values=[dict(value_id='x::slot-contract:person',content=self.text)]),
            source_components={'1':'self','2':'blocked'},original_sources={'x::slot-contract:person':['1']},bad_components=['blocked'],dev_components=[],
            policy=dict(coexistence='Types may coexist.',inventory='Allowed is not required.',demonstration='Examples are not audio evidence.',presence='Unknown values do not suppress types.'))
        self.k=dict(construction_split='train',cards={t:dict(definition=t,definition_components=['self'],examples=[]) for t in ('person','time')},folds=[dict(fold_id='f',types=['person','time'])],source_components=self.bundle['source_components'])
        docs=dict(bundle=self.bundle,train=self.rows,author_support=dict(output=dict(slot_support=[dict(utterance_id=1,entities=[dict(entity_type='person')])])),generic=dict(schema='saea-asrbias-slot-type-descriptions-v1',types=dict(person='name of a person')))
        refs={}
        for name,d in docs.items():
            raw=json.dumps(d).encode();p=self.root/(name+'.json');p.write_bytes(raw);refs[name]=dict(path=p.name,sha256=hashlib.sha256(raw).hexdigest(),pointer='')
        reviews={}
        for kind,text,status in [('extraction_boundary','An adjacent time can coexist.','applicable'),('presence','Omit a missing span.','inapplicable'),('ambiguity','Prefer person instead of time.','inapplicable')]:
            rid='s1-rule-'+digest(['person',kind,text])[:20];reviews[rid]=dict(status=status,description_sha256=hashlib.sha256(text.encode()).hexdigest(),reason='fixture reviewed task scope')
        self.policy=dict(schema='s1-knowledge-supply-policy-v1',construction_split='train',sources=refs,rule_scope_reviews=reviews)
    def tearDown(self):self.temp.cleanup()
    def compile(self):return build_s1_supply(self.k,self.bundle,self.rows,self.policy,self.root)
    def test_exact_literal_rule_and_conservative_dependency(self):
        out=self.compile();self.assertEqual(len(out['s1_discriminative_rules']),1)
        rule=out['s1_discriminative_rules'][0];self.assertEqual(rule['description'],'An adjacent time can coexist.')
        self.assertEqual(rule['source_components'],['self']);self.assertEqual(rule['related_types'],['time'])
        self.assertFalse(rule['scope']['final_adjudication'])
    def test_inapplicable_final_choice_never_supplied(self):
        out=self.compile();self.assertEqual(len(out['s1_rule_scope_audit']),3)
        self.assertNotIn('Prefer',json.dumps(out['s1_discriminative_rules']))
    def test_sha_tamper_rejected(self):
        (self.root/'generic.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):self.compile()
    def test_pointer_escape_and_relative_path(self):
        ref=self.policy['sources']['generic'];self.assertEqual(source_document(self.root,dict(ref,pointer='/types/person')),'name of a person')
        with self.assertRaises(ValueError):source_document(self.root,dict(ref,path='../generic.json'))
    def test_unreviewed_or_altered_rule_rejected(self):
        self.policy['rule_scope_reviews']={}
        with self.assertRaisesRegex(ValueError,'scope review'):self.compile()
    def test_query_self_components_remain_bound(self):
        out=self.compile();self.assertEqual(out['cards']['person']['definition_components'],['self'])
        self.assertEqual(out['cards']['person']['definition'],'person')
        self.assertEqual(out['cards']['person']['independent_definition']['source_components'],[])
    def test_retrieval_entire_sentence_positive_only_and_blocked_removed(self):
        out=self.compile();self.assertEqual(len(out['retrieval_examples']),1)
        ex=out['retrieval_examples'][0];self.assertEqual(ex['query'],'name at noon');self.assertEqual(ex['observed_types'],['person']);self.assertIsNone(ex['audio_sha256'])
        self.assertNotIn('intent',ex)
    def test_missing_generic_is_gap_no_fabrication(self):
        out=self.compile();self.assertNotIn('independent_definition',out['cards']['time'])
        self.assertEqual(out['knowledge_supply']['gaps']['independent_definition_missing'],['time'])
    def test_wrong_construction_inputs_rejected(self):
        with self.assertRaisesRegex(ValueError,'differ'):build_s1_supply(self.k,self.bundle,self.rows[:1],self.policy,self.root)
        self.policy['construction_split']='test'
        with self.assertRaisesRegex(ValueError,'TRAIN'):self.compile()
    def test_disable_sections_keeps_original_business_cards(self):
        self.policy.update(independent_definition_enabled=False,discriminative_rules_enabled=False,retrieval_examples_enabled=False)
        out=self.compile();self.assertEqual(out['cards'],self.k['cards']);self.assertEqual(out['folds'],self.k['folds'])
        self.assertEqual(out['retrieval_examples'],[]);self.assertEqual(out['s1_discriminative_rules'],[])
    def test_input_is_not_mutated_and_identity_is_deterministic(self):
        before=copy.deepcopy(self.k);a=self.compile();b=self.compile()
        self.assertEqual(self.k,before);self.assertEqual(a,b)
    def test_none_not_business_ontology_and_procedural_presence_locked(self):
        out=self.compile();self.assertNotIn('NONE_OF_FOLD',out['cards'])
        self.assertEqual(len(out['knowledge_supply']['procedural_rules']),4)
        self.assertIn('Unknown values',out['knowledge_supply']['procedural_rules'][-1]['description'])

if __name__=='__main__':unittest.main(verbosity=2)
