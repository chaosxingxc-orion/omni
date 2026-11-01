import argparse,hashlib,json
from pathlib import Path
from speech_aware_evidence_acquisition.rails.structured_experience_rules import RULE_SCHEMA,RULE_SET_SCHEMA,validate_rule_set,canonical_rule_set
parser=argparse.ArgumentParser()
parser.add_argument('--operations',type=Path,required=True)
parser.add_argument('--audit',type=Path,required=True)
parser.add_argument('--labels',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
ops=json.loads(args.operations.read_text())
audit=json.loads(args.audit.read_text())
labels=json.loads(args.labels.read_text())
observed_intents={row['intent'] for row in labels['rows'].values()}
observed_slots={slot for row in labels['rows'].values() for slot in row['slots']}
observed_relations={(row['intent'], slot) for row in labels['rows'].values() for slot in row['slots']}
# One source-bound operation per rule.  These are deliberately scoped to the
# operation's observed intent/slot contract; no Cartesian expansion is used.
selected=[
 ('a9ce4320','calendar_set',('person',),('person','calendar_set'),('preserve possessive surface form',),()),
 ('a50c4dfc','email_querycontact',('person',),('person',),('only when contact-name cue is explicit',),()),
 ('810ab363','weather_query',('place_name',),('place_name',),('reject metaphorical/non-physical references',),()),
 ('a89fe17f','weather_query',('place_name',),('place_name',),('do not invent an absent slot',),()),
 ('a60e87ea','recommendation_events',('place_name',),('place_name',),('only explicit event location',),()),
 ('e184bfda','calendar_query',('date','event_name'),('calendar_query',),('require personal scheduling activation',),()),
 ('e32554c1','lists_createoradd',('list_name',),('list_name',),('do not add event_name',),()),
 ('0e7a84b2','cooking_recipe',('cooking_type','food_type'),('cooking_type','food_type'),('verb must be followed by food cue',),()),
 ('54d12d26','iot_cleaning',('device_type',),('iot_cleaning',),('general conversation is an exception',),()),
 ('14b9e1db','calendar_set',('person','relation'),('person','relation'),('keep possessive on person; relation is role only',),()),
]
by_prefix={x['operation_sha256'][:8]:x for x in ops}
excluded_findings={'F1','F2','F3','F4','F5','F8'}
excluded_hashes={ref['operation_sha256'] for finding in audit['findings']
                 if finding['id'] in excluded_findings
                 for ref in finding.get('source_operations', ())}
rules=[];payloads={};source_index=[]
for i,(prefix,intent,slots,components,trigger,exceptions) in enumerate(selected,1):
 op=by_prefix[prefix]
 assert intent in observed_intents, (intent, 'intent absent from fixed H1 roster')
 assert set(slots) <= observed_slots and set(components) <= (observed_slots | observed_intents), (prefix, 'slot/component absent from fixed H1 roster')
 assert all((intent, slot) in observed_relations for slot in slots), (prefix, 'intent-slot relation absent from fixed H1 roster')
 assert op['operation_sha256'] not in excluded_hashes, (prefix, 'audited high-risk source')
 raw=json.dumps({'operation':op['operation'],'id':op['id'],'content':op['content']},ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
 assert hashlib.sha256(raw).hexdigest()==op['operation_sha256'],(prefix,hashlib.sha256(raw).hexdigest(),op['operation_sha256'])
 ref=f"h1-operation:{op['operation_sha256']}"
 payloads[ref]=raw
 rules.append({'schema':RULE_SCHEMA,'rule_id':f'conditioned-{i:02d}','kind':'conditional-rule',
   'scope':{'task_ids':[intent],'phases':['knowledge-consumption'],'components':list(components)},
   'trigger':list(trigger),'action':[f'apply only to {intent} with slots: {", ".join(slots)}',op['content']],
   'exceptions':list(exceptions),'source':{'ref_id':ref,'sha256':op['operation_sha256']},
   'conflict':{'state':'none','with_rule_ids':[],'resolution':None}})
 source_index.append({'ref_id':ref,'operation_sha256':op['operation_sha256'],'group_contact':op['group_contact'],'operation_index':op['operation_index']})
doc={'schema':RULE_SET_SCHEMA,'rules':rules}
validated=validate_rule_set(doc,source_payloads=payloads)
out=args.output
record={'schema':'saea-evidence-conditioned-candidate-cpu-v1','status':'cpu-validated-not-admitted','source_scope':{'source_operations':102,'source_groups':51,'source_file_sha256':hashlib.sha256(args.operations.read_bytes()).hexdigest(),'audit_file_sha256':hashlib.sha256(args.audit.read_bytes()).hexdigest()},'rules':doc,'source_index':source_index,'validation':{'rules_validated':len(validated),'new_model_calls':0,'new_scoring_jobs':0,'h2_read':False,'test_read':False,'cartesian_intent_slot_expansion':False,'observed_relation_count':len(observed_relations),'excluded_findings':sorted(excluded_findings),'excluded_operation_count':len(excluded_hashes)},'admission':{'eligible':False,'reason':'CPU structure/source validation only; semantic TRAIN counterexample review and fixed-consumer dry-run remain required before any NPU budget.'}}
out.write_text(json.dumps(record,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(out)
print('rules',len(validated),'sha',hashlib.sha256(out.read_bytes()).hexdigest())
