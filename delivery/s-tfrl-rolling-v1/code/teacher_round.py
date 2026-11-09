"""ACE dynamic evidence + pinned streaming executor. One dispatch per revision."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'delivery/s-tfrl-rolling-v1'
sys.path.insert(0,str(ROOT/'delivery/s-single-intent-fewshot-v1/code'))
from common import read,save,sha,canonical,require
EX=ROOT/'repro/codex-subprocess/global-topk-transition/recall_teacher_prepare/ace-recall-v1/vendor/base/ace/executor.py'
EXSHA='c9f9bbd333f038955945db82d946cb2757bafbc340b1a8ca476f52f3ab406bd4'
OLD=ROOT/'delivery/s-ace-teacher-v1/revision-v2'
TEMPLATE='''You are the ACE Reflector and Curator for frozen-parameter S slot extraction.
Use only the supplied bounded FIT diagnostics, legal TRAIN evidence and locked KB. Never request TEST/FoldA/FoldB feedback, whole trajectories, audio transcripts for current runtime, invented business knowledge, or weight updates.
Reflector: distinguish confirmed source/protocol contradictions, observed symptoms, causal hypotheses and counterevidence. Existing failed interventions are real feedback: retain their cost and retire/refine the causal claim. Parent identity is mandatory. Do not conflate different policies/seeds or partial diagnostics with full outcomes.
Curator: propose 1 to 4 distinct single-mechanism root-cause strategies, as structured data only. Prefer compact complete JSON serialization versus blank-loop termination, safe absence handling, knowledge span consumption, and relevance-constrained real example composition if evidence supports them. Avoid repeating rejected broad nonempty/schema or slot-name-equality rules. Preserve full real TRAIN examples/annotations. Do not claim finite bindings are missing: measured legal5598 has zero binding exclusions. Do not use a prior casing convention as semantic improvement; normalized scores pending acceptance may make it irrelevant.
No executable code, shell, eval, arbitrary source patch, credentials, endpoint edits, dataset resampling or business examples. Allowed patch targets are instruction append, a source-pinned card field replacement, or selector configuration already supported by the local compiler. Each proposal has exactly one mechanism, source locations/evidence IDs, applicability/exception, counterevidence, expected changes and falsification/rollback. Teacher text is a proposal; local bounded compiler must validate it before any run.
Output ONE JSON object with schema='s-rolling-ace-proposal-v1', parent_versions (array), reflector={confirmed,observed,hypotheses,failed_intervention_lessons,uncertainties}, curator={experience_entries:[{entry_id,operation:ADD|UPDATE|DELETE,parent_entry,mechanism,instruction,source_evidence_ids,applicability,counterevidence,expected_changes,falsification,rollback}],proposals:[{candidate_id,parent_version,mechanism,patch:{target:'instruction_append'|'card_field_replace'|'selector_config',value:string|object},source_evidence_ids,applicability,counterevidence,expected_changes,falsification,rollback}],retirements:[]}. At most 4 entries and proposals. All source references must exist in evidence. Use English instruction text for the model, Chinese explanations permitted. No observed gain exists without actual complete paired results.
'''

def executor():
    require(sha(EX.read_bytes())==EXSHA,'executor pin changed')
    s=importlib.util.spec_from_file_location('rolling_ace_paid_executor',EX);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def prepare(revision):
    folder=OUT/'teacher'/revision
    require(not folder.exists(),'immutable new teacher revision required')
    folder.mkdir(parents=True,mode=0o700)
    evidence=[]
    def add(eid,path,content):
        p=ROOT/path
        evidence.append(dict(evidence_id=eid,source=path,source_sha256=sha(p.read_bytes()),content=content))
    add('baseline545','delivery/s545-root-cause-v1/REPORT.md',
        dict(samples=545,exact=118,legal=530,invalid=15,strict_F1=0.291611185,official_conditional_SLU=0.5689150196,
             empty_filler=10,length=5,caseonly_whole_rows=59,caseonly_any_pairs=135,
             scope='FIT gold-intent-conditional slots; not E2E; no acoustic rootcause adjudication'))
    failed='delivery/s545-protocol-analysis-v1/receipts/FINAL-COMPLETE.json'
    add('failed-protocol545',failed,read(ROOT/failed)['summary'])
    scope='delivery/s-knowledge-scope-audit-v1/result-final/summary.json'
    add('legal-train-scope',scope,read(ROOT/scope))
    for eid,name in [('optional-slot-statistics','intent-slot-presence.jsonl'),('observed-combinations','slot-multiset-frequencies.jsonl')]:
        p='delivery/s-knowledge-scope-audit-v1/result-final/'+name
        rows=[json.loads(l) for l in (ROOT/p).read_text().splitlines() if l.strip()]
        # Bounded examples; aggregate complete scope supplied separately.
        add(eid,p,dict(total_rows=len(rows),representatives=rows[:20]))
    early='delivery/s-single-intent-execution-v1/MAIN-EARLY-ERRORS.private.json'
    d=read(ROOT/early)
    add('single-intent-early145',early,dict(snapshot_only=True,rows=d['rows'],counts=d['counts'],
        representatives=[{k:v for k,v in e.items() if k not in ('content','hooks')} |
                         dict(content_prefix=e.get('content','')[:650],content_chars=len(e.get('content','')))
                         for e in d['examples'][:6]]))
    length='delivery/s-single-intent-execution-v1/MAIN-LENGTH-DIAGNOSTIC.private.json'
    add('blank-loop-diagnostic',length,read(ROOT/length))
    for eid,p in [('single-intent-source-policy','delivery/s-single-intent-fewshot-v1/prepared-v5-dynamic/source-policy.json'),
                 ('complement-rule','delivery/s-tfrl-rolling-v1/config/complement-v1.json'),
                 ('timezone-diff','delivery/s-timezone-card-v1/prepared-v2/patch/time_zone.content.diff')]:
        add(eid,p,read(ROOT/p) if p.endswith('.json') else (ROOT/p).read_text())
    p='delivery/s-timezone-card-v1/prepared-v2/evidence/TRAIN-spans.private.jsonl'
    train=[json.loads(l) for l in (ROOT/p).read_text().splitlines() if l.strip()]
    add('timezone-legal-train',p,dict(records=[dict(slurp_id=r['slurp_id'],purpose=r['purpose'],
        sentence=r['original_record']['sentence'],annotation=r['original_record']['sentence_annotation'],spans=r['original_spans']) for r in train]))
    for eid,p in [('old-teacher-real-response','delivery/s-ace-teacher-v1/revision-v2/teacher/collected/curator.json'),
                 ('old-teacher-review','delivery/s-ace-teacher-v1/revision-v2/teacher/MAIN-REVIEW.json')]:
        add(eid,p,read(ROOT/p))
    code_refs={
      'project-single-intent':'delivery/s-single-intent-fewshot-v1/code/prepare.py',
      'safe-example-and-validator':'delivery/s-single-intent-fewshot-v1/code/common.py',
      'cached-signed-score-selector':'delivery/s-single-intent-fewshot-v1/code/retrieval.py',
      'complement-compiler':'delivery/s-tfrl-rolling-v1/code/prepare_candidates.py'}
    for eid,p in code_refs.items():
        text=(ROOT/p).read_text()
        add(eid,p,dict(source=text[:17000]))
    p='delivery/s-single-intent-fewshot-v1/config/prepare.json';cfg=read(ROOT/p)
    instr=cfg['instruction'];add('single-intent-instruction',instr,(ROOT/instr).read_text())
    # Bounded actual gold/pred excerpts: no P raw/reasoning/audio or full traces.
    casepaths=sorted((ROOT/'delivery/s545-root-cause-v1/audit-v5/private/cases').glob('*.json'))
    selected=[];seen=set()
    for p in casepaths:
        c=read(p); flags=c['overlapping_observations']
        key=c['outcome']+'|'+str(flags[:1])
        if key in seen:continue
        seen.add(key)
        selected.append(dict(evidence_id='fit-case-'+str(c['slurp_id']),source=str(p.relative_to(ROOT)),source_sha256=sha(p.read_bytes()),
            content={k:c[k] for k in ['sample_id','slurp_id','gold_intent','gold','differences','audit_conclusion','confirmation_boundary','next_discriminating_check']} |
            dict(S_gold_branch_pred=c['S']['gold_branch_pred'],finish_reason=c['S']['finish_reason'],listened=False)))
        if len(selected)>=10:break
    evidence.extend(selected)
    parents=['prepared-v5-dynamic/single_intent_no_examples','prepared-v5-dynamic/single_intent_fewshot',
             'timezone-single-v1','fewshot-complement-v1']
    context=dict(schema='s-rolling-ace-evidence-v1',parent_versions=parents,evidence=evidence,
        metrics=dict(normalized='PENDING_EXTERNAL_CPU_ACCEPTANCE; no normalized gain known',primary='frozen strict multiset + official conditional SLU',
                     current_545_conditional_is_all_history_best=False,all_history_same_ID_best='UNKNOWN; matching historical artifacts not yet verified'),
        old_teacher=dict(request_already_terminal=True,resend=False,cost=read(OLD/'teacher/collected/COST.json')['result']),
        active_round=dict(no_candidate_result_yet=True,parameters_frozen=True,TEST_feedback=False,FoldA_FoldB_feedback=False),
        feedback_protocol=dict(complete_paired_results_required_for_promotion=True,partial_failure_diagnostics_allowed=True,
            delete_failed_entries=True,max_same_mechanism_no_gain_rounds=2,next_teacher_input='parent version + advantage/repair/regression + retired entries + cost'))
    save(folder/'dynamic-context.private.json',context)
    (folder/'template.md').write_text(TEMPLATE)
    save(folder/'messages.private.json',[dict(role='user',content=json.dumps(context,ensure_ascii=False))])
    oldcfg=read(OLD/'teacher/executor-config.private.json')
    cfg=oldcfg | dict(output_root=str(folder/'response'))
    for field,name in [('messages','messages.private.json'),('research_prompt','template.md')]:
        cfg[field]=dict(path=str(folder/name),sha256=sha((folder/name).read_bytes()))
    cfg['analysis_budget']=cfg['analysis_budget'] | dict(max_hypotheses=4,max_reasoning_tokens_target=8192,representative_case_target=10)
    save(folder/'executor-config.private.json',cfg)
    save(folder/'REQUEST-PINS.json',{str(p.relative_to(folder)):sha(p.read_bytes()) for p in folder.iterdir() if p.is_file()})
    save(folder/'REQUEST_READY.json',dict(status='NEW_REVISION_READY_AUTHORIZED_SINGLE_REQUEST',revision=revision,old_request_resend=False,
         teacher_model=cfg['model'],max_tokens=131072,soft_thinking=8192,automatic_retry=0,max_proposals=4,
         evidence_count=len(evidence),context_bytes=len(canonical(context)),API_sent=False))
    print(json.dumps(read(folder/'REQUEST_READY.json')))

def collect(folder):
    result=read(folder/'response/result.json')
    save(folder/'ACTUAL-COST.json',dict(result=result,money=None,automatic_retry=0,old_cost_preserved=True))
    if result['state']!='response_received':
        save(folder/'COLLECTED.json',dict(status='NO_CANDIDATE_UNKNOWN_OR_FAILED_NO_RESEND',state=result['state']))
        return
    require(sha((folder/'response/response.sse').read_bytes())==result['response_sha256'],'SSE custody')
    response=read(folder/'response/response.json')
    require(response['choices'][0]['finish_reason']=='stop','truncated teacher; no resend')
    content=response['choices'][0]['message']['content']
    derived=content
    # Only an exact single complete Markdown envelope, never repair inner JSON.
    if content.startswith('```json\n') and content.rstrip().endswith('```'):
        derived=content[len('```json\n'):].rstrip()[:-3].rstrip()
        require('```' not in derived,'multiple code fences; no automatic extraction')
        save(folder/'ENVELOPE-DERIVATION.json',dict(operation='remove unique complete outer markdown json fence only',
            source_content_sha256=sha(content.encode()),derived_JSON_text_sha256=sha(derived.encode()),
            response_sha256=result['response_sha256'],semantic_repair=False,new_API_calls=0))
    obj=json.loads(derived)
    require(obj['schema']=='s-rolling-ace-proposal-v1','teacher schema')
    proposals=obj['curator']['proposals'];entries=obj['curator']['experience_entries']
    require(0<=len(proposals)<=4 and len(entries)<=4,'proposal bound; no forced strategy')
    known={e['evidence_id'] for e in read(folder/'dynamic-context.private.json')['evidence']}
    for p in proposals:
        require(p['patch']['target'] in ['instruction_append','card_field_replace','selector_config'],'bounded patch target')
        require(set(p['source_evidence_ids'])<=known,'unbound evidence')
        require(all(p.get(k) for k in ['parent_version','mechanism','applicability','counterevidence','expected_changes','falsification','rollback']),'proposal fields')
    save(folder/'teacher-final.json',obj)
    save(folder/'reflector.json',obj['reflector'])
    save(folder/'curator.json',obj['curator'])
    save(folder/'COLLECTED.json',dict(status='STRUCTURED_PROPOSALS_RECEIVED_NOT_EXECUTED',proposals=len(proposals),closed_loop_gain=False))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','dispatch','worker','collect']);ap.add_argument('--revision',default='revision-v3-rootcause545');a=ap.parse_args()
    os.umask(0o077);folder=OUT/'teacher'/a.revision
    if a.action=='prepare':return prepare(a.revision)
    if a.action=='collect':return collect(folder)
    cfg=read(folder/'executor-config.private.json')
    for p,pin in read(folder/'REQUEST-PINS.json').items():require(sha((folder/p).read_bytes())==pin,'request pin drift '+p)
    if a.action=='worker':
        result=executor().execute(folder/'executor-config.private.json')
        save(folder/'TEACHER-TERMINAL.json',dict(result=result,at=time.time(),automatic_retry=0))
        try:collect(folder)
        except Exception as e:save(folder/'COLLECT-ERROR.json',dict(error_type=type(e).__name__,message=str(e),resend=False))
        return
    with (folder/'launch.guard').open('a+b') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not (folder/'launch-intent.json').exists() and not (folder/'response').exists(),'prior dispatch or unknown; no resend')
        require(bool(os.environ.get('CUSTOM_API_KEY')),'CUSTOM_API_KEY absent; do not search or print key')
        save(folder/'launch-intent.json',dict(at=time.time(),revision=a.revision,automatic_retry=0,max_calls=1))
        with (folder/'executor.log').open('xb') as f:
            c=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'worker','--revision',a.revision],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        save(folder/'DETACHED-LAUNCH.json',dict(pid=c.pid,at=time.time(),agent=False,automatic_retry=0,max_calls=1))
        print(json.dumps(dict(status='DETACHED_TEACHER_SINGLE_REQUEST',pid=c.pid,revision=a.revision)))

if __name__=='__main__':main()
