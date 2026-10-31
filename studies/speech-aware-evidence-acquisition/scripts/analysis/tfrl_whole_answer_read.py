"""One prospective whole-answer medoid screen on sealed construction625.

No generation: choose an existing candidate using outputs alone, then score the
entire625 once. The old family remains stopped regardless of this diagnostic.
"""
import argparse,copy,json,sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis'),str(ROOT/'scripts/checks')]
from tfrl_recovered_consumers import load,POOL
from tfrl_current_sources import read,sha,pinned
from tfrl_current_pool import pool
from tfrl_current_half_result import score_table,ARMS
from tfrl_h1_union import score_column
import stage3_rollout_score as s3
import paired_ci
RUN='SAEA-SLURP-TFRL-WHOLE-ANSWER-CONSTRUCTION625-READ-V1'
REL='docs/readiness/tfrl-whole-answer-construction625-v1/registration.json'
RULE='plurality-intent-lowest-index-tie;within-intent-entity-jaccard-medoid;empty-pair-one;lowest-index-medoid-tie;whole-answer-v1'
AUTH='docs/decisions/2026-10-30-owner-full-test-and-first-system-gain.md'


def select(answers):
    if len(answers)!=4:raise ValueError('exactly four real candidates required')
    for a in answers:
        if (not isinstance(a,dict) or set(a)!={'scenario','action','entities'}
                or not all(isinstance(a[k],str) for k in ('scenario','action')) or not isinstance(a['entities'],list)):
            raise ValueError('canonical candidate answer required')
        for e in a['entities']:
            if not isinstance(e,dict) or set(e)!={'type','filler'} or not all(isinstance(v,str) for v in e.values()):
                raise ValueError('canonical entity required')
    intents=[(a['scenario'],a['action']) for a in answers];counts=Counter(intents)
    winner=max(range(4),key=lambda i:counts[intents[i]])
    eligible=[i for i in range(4) if intents[i]==intents[winner]]
    entities=[{(' '.join(e['type'].split()).lower(),' '.join(e['filler'].split()).lower()) for e in a['entities']} for a in answers]
    def centrality(i):
        peers=[j for j in eligible if j!=i]
        if not peers:return Fraction(1)
        return sum((Fraction(len(entities[i]&entities[j]),len(entities[i]|entities[j]))
                    if entities[i]|entities[j] else Fraction(1)) for j in peers)/len(peers)
    chosen=max(eligible,key=centrality)
    return copy.deepcopy(answers[chosen]),chosen


def gate(metrics,control):
    gain=100*(metrics['slu_f1']-control['slu_f1']);intent=100*(metrics['intent_accuracy']-control['intent_accuracy'])
    passed=round(gain,6)>=1 and intent>=-1e-10
    return {'slu_gain_pp':gain,'intent_gain_pp':intent,'screen_passed':passed,
        'decision':'prepare-independent-comparison' if passed else 'stop-whole-answer-candidate',
        'old_family_stays_stopped':True,'system_accepted':False,'generation_authorized':False}


def sources(source,data):
    _,_,halves=load(source,data)
    verified=pool({v['plan']['half']:(v['plan'],v['results']) for v in halves.values()})
    p=data/'analysis'/POOL/'complete.json';record=read(p)
    if any(record.get(k)!=v for k,v in verified.items()) or record['sources']!={h:v['source'] for h,v in halves.items()}:
        raise ValueError('complete pool differs from exact sources')
    return halves,record,p


def register(repo,data,source):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    assert_execution_environment();require_clean_study_repo(repo);head=require_clean_study_repo(source)
    if (repo/REL).exists() or (data/'analysis'/RUN).exists():raise ValueError('fresh screen required')
    ledger=repo/'docs/exposure-ledger.md';text=ledger.read_text()
    if '| '+RUN+' |' in text:raise ValueError('screen already registered')
    _,record,p=sources(source,data)
    r={'schema':'saea-tfrl-whole-answer-read-registration-v1','read_run_id':RUN,'rows':625,'rule':RULE,
        'source_repo_relpath':str(source.resolve().relative_to(data.resolve())),'source_commit':head,
        'pool_sha256':sha(p),'sources':record['sources'],'implementation_sha256':sha(Path(__file__)),
        'official_scoring_jobs':1,'new_model_calls':0,'new_tokenizer_calls':0,
        'gate':{'slu_gain_pp_min':1.0,'intent_gain_pp_min':0.0},'authority':AUTH,
        'scope':'Prospective single fixed aggregation screen on already-exposed construction625; no acceptance claim',
        'old_family_stays_stopped':True,'generation_authorized':False}
    s3._immutable_json(repo/REL,r)
    row=['2026-10-31',RUN,'model-free-check','Single fixed whole-answer aggregation; full625','Official CPU scorer only',
        'slurp TRAIN construction625','discovery','exact existing313/312 sources','no','0','0','0',
        'All prior construction exposure retained; post-diagnosis hypothesis',REL,
        'Prospective fixed-rule registration; root countersigned under '+AUTH+'; coordinator Codex; SHA256 '+sha(repo/REL)+';1 official score; no generation admission']
    ledger.write_text(text+'\n## Whole-answer construction screen\n\n| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n| '+' | '.join(['---']*15)+' |\n| '+' | '.join(row)+' |\n')
    return r


def run(repo,data):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.data.lock import load_lock,umbrella_lock_path
    import flight_registration_check as reg
    assert_execution_environment();head=require_clean_study_repo(repo)
    r=read(repo/REL);entries=reg.parse_ledger(repo,Path('docs/exposure-ledger.md')).get(RUN,[])
    expected={'schema':'saea-tfrl-whole-answer-read-registration-v1','read_run_id':RUN,'rows':625,'rule':RULE,
        'implementation_sha256':sha(Path(__file__)),'official_scoring_jobs':1,'new_model_calls':0,'new_tokenizer_calls':0,
        'gate':{'slu_gain_pp_min':1.0,'intent_gain_pp_min':0.0},'old_family_stays_stopped':True,'generation_authorized':False}
    if (any(r.get(k)!=v for k,v in expected.items()) or len(entries)!=1 or not reg.is_countersigned(entries[0])
            or entries[0]['artifacts']!=REL or sha(repo/REL) not in entries[0].get('notes','')):
        raise ValueError('exact prospective registration required')
    source=(data/r['source_repo_relpath']).resolve()
    if not source.is_relative_to(data.resolve()) or require_clean_study_repo(source)!=r['source_commit']:
        raise ValueError('source tree changed')
    halves,record,p=sources(source,data)
    if sha(p)!=r['pool_sha256'] or record['sources']!=r['sources']:raise ValueError('registered results changed')
    output=[];chosen=Counter();pins={};ids=[]
    for h,half in halves.items():
        root=(data/half['source']['complete_relpath']).parent;complete=read(root/'complete.json');columns={}
        half_ids=[x['row_id'] for x in half['plan']['row_plan']];ids+=half_ids
        for arm in ARMS[:4]:
            rel=f'columns/arm-{arm}-'+('outputs.jsonl' if h=='h1' else 'first-visit-outputs.jsonl')
            path=pinned(root,{'relpath':rel,'sha256':complete['artifacts'][rel]});rows=list(s3.iter_jsonl(path))
            if len(rows)!=len(half_ids) or len({x['sample_id'] for x in rows})!=len(rows) or {x['sample_id'] for x in rows}!=set(half_ids):
                raise ValueError('exact candidate column required')
            columns[arm]={x['sample_id']:json.loads(x['text']) for x in rows};pins[h+'/'+arm]=sha(path)
        for sid in half_ids:
            answer,index=select([columns[a][sid] for a in ARMS[:4]])
            output.append({'sample_id':sid,'text':s3.render_answer(answer)});chosen[str(index+1)]+=1
    if len(ids)!=625 or len(set(ids))!=625:raise ValueError('unique625 required')
    out=data/'analysis'/RUN;out.mkdir(exist_ok=True)
    binding={'schema':'saea-tfrl-whole-answer-score-binding-v1','recipe_sha256':sha(repo/REL),
        'pool_sha256':sha(p),'candidate_file_hashes':pins,'rule':RULE,'implementation_commit':head}
    s3._immutable_json(out/'intent.json',binding)
    path=out/'outputs.jsonl';s3._immutable_jsonl(path,output)
    result=score_column(path,out/'scores.json',binding=binding,ids=ids,
        lock=load_lock(umbrella_lock_path(repo)),data=data)
    table=score_table(result,ids);metrics={'slu_f1':paired_ci.pooled_slu_f1(table,ids),'intent_accuracy':paired_ci.intent_mean(table,ids)}
    completion={'schema':'saea-tfrl-whole-answer-screen-complete-v1','rows':625,'metrics':metrics,
        'control':record['official_reaggregated']['lock'],'original_vote':record['official_reaggregated']['vote'],
        **gate(metrics,record['official_reaggregated']['lock']),'chosen_index_counts':dict(chosen),
        'recipe_sha256':sha(repo/REL),'source_pool_sha256':sha(p),'source_pins':pins,
        'output_sha256':sha(path),'score_sha256':sha(out/'scores.json'),'score_receipt_sha256':sha(out/'scores.receipt.json'),
        'implementation_commit':head,'new_model_calls':0,'new_tokenizer_calls':0,'official_scoring_jobs':1,
        'scope':'Construction mechanism screen; independent comparison required before system acceptance.'}
    s3._immutable_json(out/'complete.json',completion);return completion


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','run'])
    for k in ('repo','data'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--source',type=Path)
    a=vars(p.parse_args());mode=a.pop('mode')
    if mode=='run':a.pop('source');result=run(**a)
    else:result=register(**a)
    print(json.dumps(result))
