import argparse,json,re,unicodedata,hashlib,itertools
from pathlib import Path
from collections import Counter,defaultdict
from rapidfuzz import process,fuzz
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
paths={'train':ROOT/'delivery/recall-card-source-audit-v1/input/train-metadata.private.jsonl','test':ROOT/'delivery/s-large-cohort-v1/slu-task-audit/upstream-test.jsonl'}
parser=argparse.ArgumentParser()
parser.add_argument('--train',type=Path,default=paths['train'])
parser.add_argument('--test',type=Path,default=paths['test'])
parser.add_argument('--output',type=Path,default=OUT)
args=parser.parse_args()
paths={'train':args.train.resolve(),'test':args.test.resolve()};OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def dump(name,x): (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
rows={}; mentions={}; index={}; pairs={}; types={}
for split,p in paths.items():
 rows[split]=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
 for r in rows[split]:
  r['raw_intent']=r['intent'];r['intent']=r['scenario']+'_'+r['action']
 mentions[split]=[];index[split]=defaultdict(list);pairs[split]=defaultdict(list);types[split]=Counter()
 for r in rows[split]:
  tm={t['id']:t['surface'] for t in r['tokens']}
  for e in r['entities']:
   m={'id':r['slurp_id'],'intent':r['intent'],'type':e['type'],'value':' '.join(tm[i] for i in e['span']),'span':e['span'],'sentence':r['sentence'],'audio_count':len(r['recordings']),'split':split}
   m['normalized']=norm(m['value']); mentions[split].append(m);index[split][m['normalized']].append(m);pairs[split][(m['intent'],m['type'])].append(m);types[split][m['type']]+=1

def counts(ms):
 ids={(m['split'],m['id']) for m in ms}; audio={(m['split'],m['id']):m['audio_count'] for m in ms}
 return {'mentions':len(ms),'utterances':len(ids),'recordings':sum(audio.values())}
summary={'sources':{s:{'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'utterances':len(rows[s]),'recordings':sum(len(r['recordings']) for r in rows[s]),'entity_mentions':len(mentions[s]),'types':len(types[s]),'noncanonical_raw_intent_rows':sum(r['raw_intent']!=r['intent'] for r in rows[s]),'intent_type_pairs':len(pairs[s]),'unique_values':len(index[s])} for s,p in paths.items()},'test_types_absent_train':sorted(set(types['test'])-set(types['train'])),'train_types_absent_test':sorted(set(types['train'])-set(types['test']))}
unseen=[]
for pair,ms in pairs['test'].items():
 if pair not in pairs['train']:unseen.append({'intent':pair[0],'type':pair[1],**counts(ms),'examples':ms[:3]})
summary['test_unseen_intent_type_pairs']={'pairs':len(unseen),**counts([m for m in mentions['test'] if (m['intent'],m['type']) not in pairs['train']])}
dump('unseen-pairs.private.json',unseen)
allidx=defaultdict(list)
for s in index:
 for v,ms in index[s].items():allidx[v].extend(ms)
ambiguous={}; typepairs={}
for s,idx in [*index.items(),('pooled',allidx)]:
 groups=[];same=[];tp=defaultdict(set)
 for v,ms in idx.items():
  ts=sorted({m['type'] for m in ms})
  if len(ts)<2:continue
  ins=defaultdict(set)
  for m in ms:ins[m['intent']].add(m['type'])
  within={i:sorted(t) for i,t in ins.items() if len(t)>1}
  groups.append({'value':v,'types':ts,'within_intent':within,**counts(ms),'examples':ms})
  if within:same.append(v)
  for a,b in itertools.combinations(ts,2):tp[(a,b)].add(v)
 ambiguous[s]=groups
 typepairs[s]=[{'types':list(k),'values':len(v),'examples':sorted(v)[:12]} for k,v in sorted(tp.items(),key=lambda kv:-len(kv[1]))]
 summary.setdefault('exact_multitype_values',{})[s]={'groups':len(groups),'same_intent_groups':len(same),'type_pairs':len(tp),**counts([m for v,ms in idx.items() if len({m['type'] for m in ms})>1 for m in ms])}
dump('exact-multitype.private.json',ambiguous);dump('type-pairs.private.json',typepairs)
coverage=defaultdict(list)
for m in mentions['test']:
 prior=index['train'].get(m['normalized'],[])
 cat='same_value_same_type' if any(t['type']==m['type'] for t in prior) else 'same_value_other_type_only' if prior else 'unseen_value'
 coverage[cat].append(m)
summary['test_value_type_coverage']={k:counts(v) for k,v in coverage.items()}
dump('test-value-coverage.private.json',coverage)
# Exact normalized full-sentence + intent, comparing normalized value/type multisets.
si={}
for s in rows:
 idx=defaultdict(list)
 for r in rows[s]:
  tm={t['id']:t['surface'] for t in r['tokens']}
  sig=sorted((e['type'],norm(' '.join(tm[i] for i in e['span']))) for e in r['entities'])
  idx[(norm(r['sentence']),r['intent'])].append({'id':r['slurp_id'],'signature':sig,'split':s,'audio_count':len(r['recordings'])})
 si[s]=idx
conf=[]; shared=0
for key,trs in si['train'].items():
 tes=si['test'].get(key,[])
 if not tes:continue
 shared+=1
 if len({json.dumps(x['signature']) for x in trs+tes})>1:conf.append({'sentence':key[0],'intent':key[1],'train':trs,'test':tes})
summary['identical_sentence_intent_cross_split']={'shared_groups':shared,'conflicting_groups':len(conf)}
dump('sentence-conflicts.private.json',conf)
# Match TEST annotated token sequences in TRAIN. Exact boundaries; no punctuation deletion.
phrases=defaultdict(list)
for v,ms in index['test'].items():
 for m in ms:
  phrases[tuple(norm(x) for x in m['value'].split())].append(m)
lengths=sorted({len(p) for p in phrases})
findings=[]
for r in rows['train']:
 toks=[];ids=[]
 for t in r['tokens']:
  for part in norm(t['surface']).split():toks.append(part);ids.append(t['id'])
 for n in lengths:
  if n>len(toks):break
  for pos in range(len(toks)-n+1):
   key=tuple(toks[pos:pos+n]);refs=phrases.get(key)
   if not refs:continue
   span=set(ids[pos:pos+n]);over=[e for e in r['entities'] if span.intersection(e['span'])]
   for ty in sorted({m['type'] for m in refs}):
    if any(e['type']==ty and set(e['span'])==span for e in over):continue
    relevant=[m for m in refs if m['type']==ty]
    cat='unannotated_span' if not over else 'exact_span_other_type' if any(set(e['span'])==span for e in over) else 'partial_or_larger_span'
    findings.append({'train_id':r['slurp_id'],'train_intent':r['intent'],'sentence':r['sentence'],'value':' '.join(key),'test_type':ty,'category':cat,'same_intent':any(m['intent']==r['intent'] for m in relevant),'train_overlapping_entities':over,'test_examples':sorted(relevant,key=lambda m:m['intent']!=r['intent'])[:2],'audio_count':len(r['recordings'])})
summary['test_phrase_in_train_label_discrepancies']={}
for same in [True,False]:
 for cat in ['unannotated_span','exact_span_other_type','partial_or_larger_span']:
  fs=[f for f in findings if f['same_intent']==same and f['category']==cat]
  summary['test_phrase_in_train_label_discrepancies'][f'{"same" if same else "different"}_intent/{cat}']={'phrase_type_occurrences':len(fs),'train_utterances':len({f['train_id'] for f in fs}),'distinct_value_type_pairs':len({(f['value'],f['test_type']) for f in fs})}
dump('possible-omissions.private.json',findings)
# Explicit lexical-neighbor candidate screen, not semantic-equivalence claims.
values=sorted(v for v in allidx if len(v)>=4)
near=[]
for i,v in enumerate(values):
 for w,score,j in process.extract(v,values,scorer=fuzz.ratio,score_cutoff=90,limit=None):
  if j<=i:continue
  t1={m['type'] for m in allidx[v]};t2={m['type'] for m in allidx[w]}
  if not any(a!=b for a in t1 for b in t2):continue
  near.append({'left':v,'right':w,'ratio':score,'left_types':sorted(t1),'right_types':sorted(t2),'same_intent_cross_type':bool({m['intent'] for m in allidx[v]} & {m['intent'] for m in allidx[w]}) and any(a['intent']==b['intent'] and a['type']!=b['type'] for a in allidx[v] for b in allidx[w])})
summary['lexical_neighbors']={'metric':'RapidFuzz fuzz.ratio >=90; distinct NFKC/casefold/whitespace normalized values; each length>=4 characters; cross-type; unordered pairs','pairs':len(near),'distinct_values':len({v for n in near for v in [n['left'],n['right']]}),'same_intent_cross_type_pairs':sum(n['same_intent_cross_type'] for n in near)}
dump('lexical-neighbors.private.json',near)
dump('SUMMARY.json',summary)
print(json.dumps(summary,ensure_ascii=False,indent=2))
