import argparse,json,math
from collections import defaultdict,Counter
from pathlib import Path
P=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--version',default='clean-v7');args=parser.parse_args()
allg=json.loads((P/args.version/'exact-multitype.private.json').read_text())
out={}
for split,groups in allg.items():
 cross=[g for g in groups if not g['within_intent']];within=[g for g in groups if g['within_intent']]
 mentions=[m for g in groups for m in g['examples']]
 byvi=defaultdict(Counter);byv=defaultdict(Counter)
 for m in mentions:byvi[(m['normalized'],m['intent'])][m['type']]+=1;byv[m['normalized']][m['type']]+=1
 def H(c):
  n=sum(c.values());return -sum(k/n*math.log2(k/n) for k in c.values())
 n=len(mentions)
 hv=sum(sum(c.values())*H(c) for c in byv.values())/n
 hvi=sum(sum(c.values())*H(c) for c in byvi.values())/n
 out[split]={'multi_type_values':len(groups),'only_between_intents':len(cross),'within_at_least_one_intent':len(within),'multi_value_entity_mentions':n,'value_intent_groups':len(byvi),'value_intent_groups_single_observation':sum(sum(c.values())==1 for c in byvi.values()),'value_intent_groups_multiple_types':sum(len(c)>1 for c in byvi.values()),'empirical_H_type_given_value_bits':hv,'empirical_H_type_given_value_intent_bits':hvi,'empirical_relative_entropy_reduction':1-hvi/hv,'limitation':'Descriptive in-sample entropy on multi-type values only. Sparse intent conditioning lowers entropy mechanically; not held-out accuracy, causality, or full-context dependence proof.'}
(P/args.version/'CONTEXT-SUMMARY.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))
