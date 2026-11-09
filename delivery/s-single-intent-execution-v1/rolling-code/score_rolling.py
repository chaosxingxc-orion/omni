"""Terminal sidecar: frozen typed + strict/official; preserve TRAIN hold masks."""
import argparse,collections,json,sys,subprocess,os
# Use the fixed-image native extension before optional workstation dependencies.
import regex
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--state',required=True);p.add_argument('--workspace',required=True);a=p.parse_args();S=Path(a.state);W=Path(a.workspace);cfg=json.loads((S/'runtime-config.json').read_bytes());term=json.loads((S/'run/TERMINAL.json').read_bytes());assert term['status']=='COMPLETE' and term['unknown']==0
if cfg['expected_rows']==1635 and not cfg.get('dataset_partition') and S.parent.name=='jobs' and not (S/'score/SUMMARY.json').exists():
 r=subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('score_terminal.py')),'--state',str(S),'--workspace',str(W)],capture_output=True,text=True);assert r.returncode==0,r.stdout+r.stderr
sys.path.insert(0,str(W/'delivery/s-single-intent-fewshot-v1/code'));from common import read,lines,save,sha,canonical,load
from aggregate import summarize
T=Path(os.environ.get('OMNI_TYPED_SIDECAR',str(W/'delivery/s-typed-normalized-scoring-v1')));frozen=read(T/'CODE-CONFIG-FROZEN.json')
for path,pin in frozen['pins'].items():assert sha((T/path).read_bytes())==pin,path
sys.path.insert(0,str(T/'code'));from normalizer import Normalizer
normalizer=Normalizer(read(T/'CONFIG.json'));manifest=lines(W/cfg['manifest']);predictions=lines(S/'run/predictions.private.jsonl');assert len(predictions)==cfg['expected_rows'];summaries=summarize(lines(W/cfg['samples']),manifest,predictions);assert len(summaries)==cfg['full_denominator']
if not cfg.get('dataset_partition'):
 baseline={r['sample_id']:r for r in lines(T/'inputs/baseline545.private.jsonl')};mat=read(W/'delivery/s-correct-p-data-v1/config/materialize-locked.json');intents={r['sample_id']:r['gold_OFFLINE_ONLY'] for r in lines(W/mat['p_ledger']) if r['cohort']=='fit'};gold={sid:r['gold'] for sid,r in baseline.items()};mask={sid:dict(reviewed_hold=False,teacher_update_allowed=True,evaluation_only=False) for sid in gold}
else:
 ref=lines(W/cfg['reference_CPU_only']);gold={r['sample_id']:r.get('entities',r.get('gold')) for r in ref};intents={r['sample_id']:r.get('intent',r.get('gold_intent')) for r in ref};mask={r['sample_id']:dict(reviewed_hold=r['reviewed_hold'],teacher_update_allowed=r['teacher_update_allowed'],evaluation_only=r.get('evaluation_only',False)) for r in ref};assert all(m['teacher_update_allowed']==(not m['reviewed_hold'] and not m['evaluation_only']) for m in mask.values())
def pairs(es,typed):return collections.Counter((e['type'],normalizer.normalize(e['type'],e['filler']).key if typed else e['filler']) for e in es)
def counts(g,p,valid):
 tp=sum((g&p).values());fp=sum(p.values())-tp;fn=sum(g.values())-tp;den=2*tp+fp+fn
 return dict(tp=tp,fp=fp,fn=fn,F1=(2*tp/den if den else 1) if valid else 0,all_slots_exact=int(valid and g==p),invalid=int(not valid))
def aggregate(records,key):
 total=collections.Counter()
 for r in records:total.update({k:r[key][k] for k in ['tp','fp','fn','all_slots_exact','invalid']})
 tp,fp,fn=(total[k] for k in ['tp','fp','fn']);return dict(total,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1,denominator=len(records))
records=[];official_records=[]
for row in summaries:
 sid=row['sample_id'];branch=next(b for b in row['branches'] if b['candidate_id']==intents[sid]);valid=branch['status']=='valid';prediction=branch['result']['entities'] if valid else [];m=mask[sid]
 rec=dict(sample_id=sid,valid=valid,**m,normalized=counts(pairs(gold[sid],True),pairs(prediction,True),valid),strict=counts(pairs(gold[sid],False),pairs(prediction,False),valid),gold=gold[sid],prediction=prediction,invalid_branches=sum(b['status']!='valid' for b in row['branches']));records.append(rec)
 official_records.append(dict(sample_id=sid,reference_entities_CPU_only=gold[sid],gold_intent_CPU_only=intents[sid],outcome='legal' if valid else 'invalid',rails_parsed={intents[sid]:dict(entities=prediction)} if valid else None,**m))
from importlib import util
sp=util.spec_from_file_location('rolling_official',Path(__file__).with_name('official_rolling.py'));official=util.module_from_spec(sp);sp.loader.exec_module(official);official.ROOT=W
def official_subset(rows):return official.aggregate(dict(rails_src=cfg['rails_src']),rows)[0]
all_results={}
for name,subset in [('all',records),('teacher_eligible',[r for r in records if r['teacher_update_allowed']]),('hold',[r for r in records if r['reviewed_hold']]),('evaluation',[r for r in records if r['evaluation_only']])]:
 ids={r['sample_id'] for r in subset};of=official_subset([r for r in official_records if r['sample_id'] in ids]) if ids else {'official_conditional_SLU_F1':None,'denominator':0,'observed':0,'missing':0}
 all_results[name]=dict(normalized=aggregate(subset,'normalized'),strict=aggregate(subset,'strict'),official=of)
save(S/'score/samples.private.jsonl',summaries,jsonl=True)
_,official_counts=official.aggregate(dict(rails_src=cfg['rails_src']),official_records);save(S/'score/official-counts.private.jsonl',official_counts,jsonl=True)
metrics=dict(cost_note='historical C HTTP400 is one inherited contact, retained for traceability and never added once per scored arm',status='COMPLETE_TYPED_STRICT_OFFICIAL_SIDECAR',groups=all_results,denominator=len(records),scope='conditional gold intent branch; limited accepted typed normalization; not E2E',rule_config_sha256=sha((T/'CONFIG.json').read_bytes()),rule_freeze_sha256=sha((T/'CODE-CONFIG-FROZEN.json').read_bytes()),usage=term['usage'],model_calls=0,money=None)
save(S/'score/typed-normalized.json',metrics);save(S/'score/typed-samples.private.jsonl',records,jsonl=True);save(S/'score/teacher-eligible.private.jsonl',[r for r in records if r['teacher_update_allowed']],jsonl=True);save(S/'score/hold.private.jsonl',[r for r in records if r['reviewed_hold']],jsonl=True);save(S/'score/evaluation.private.jsonl',[r for r in records if r['evaluation_only']],jsonl=True);print(json.dumps(metrics))
