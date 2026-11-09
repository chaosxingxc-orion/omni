"""Complete real feedback -> one paid ACE round -> bounded append -> CPU READY.

No service/lane actions; no retries. Validation/hold feedback is forbidden.
Unknown paid outcomes are retained and never re-dispatched.
"""
import copy,fcntl,json,os,re,subprocess,sys,time,traceback
from pathlib import Path
import teacher_round as t
from common import read,save,sha,canonical,require
import producer
ROOT,OUT=t.ROOT,t.OUT
CONTEXT_BYTE_LIMIT=100000
DIAGNOSTIC_BOUNDARY=('diagnostic_OFFLINE_ONLY: locked TRAIN reference transcription/annotation, '
    'not acoustic listening or ASR evidence; cannot establish ASR error. Never inject query '
    'text/gold into runtime packets or use query gold to select runtime examples. Rewards '
    'and positive/negative predictions refer only to the stated gold-intent branch, not '
    'the three-branch sample or end-to-end routing. Other branch indexes are scope only.')

def semantic_supplement(marker):
    if not marker.exists():return None
    try:m=read(marker)
    except (OSError,ValueError):return None
    if m.get('status') not in ('COMPLETE_KNOWN_FULL545','FULL545_THREE_PAIRS_READY_FOR_REVIEW'):return None
    if m.get('full_denominator',545)!=545 or m.get('unknown',0)!=0:return None
    path=m.get('bounded_teacher_summary');pin=m.get('bounded_teacher_summary_sha256')
    if not path or not pin:return None
    source=ROOT/path
    try:
        if not source.is_file() or sha(source.read_bytes())!=pin:return None
        return read(source)
    except (OSError,ValueError):return None

def dispatch_permission(folder,quota):
    if (folder/'launch-intent.json').exists():return False
    if quota.exists():
        require(read(quota)['revision']==folder.name,'persistent one-round quota already consumed')
        # Consumed intent with no dispatch evidence remains unresolved; never auto resend.
        return False
    return True

def group_context(data):
    """Attach actual runtime context, group/source isolation and paired rewards."""
    data=copy.deepcopy(data)
    from common import lines,source_id,annotations
    full=ROOT/'delivery/s-single-intent-execution-v1/rolling-state/group-feedback/group-advantages.private.jsonl'
    groups=lines(full);require(len(groups)==545 and all(len(g['members'])==8 for g in groups),'complete545 actual group members')
    sourcegroups=read(ROOT/'delivery/recall-card-source-audit-v1/input/train-source-group-index.json')['groups']
    native=read(OUT/'grpo-reserve/grpo-fewshot-seed2026110901/single_intent_fewshot/native-UNBOUND.json')
    custody=read(OUT/'config/grpo-policy-custody.private.json')
    nativepath=OUT/'grpo-reserve/grpo-fewshot-seed2026110901/single_intent_fewshot/native-UNBOUND.json'
    require(sha(nativepath.read_bytes())==custody['seeds']['2026110901']['config_sha256'], 'frozen seed config custody')
    require(sha((ROOT/native['manifest']).read_bytes())==native['manifest_sha256']==custody['seeds']['2026110901']['manifest_sha256'], 'frozen manifest custody')
    manifests=lines(ROOT/native['manifest']);by={}
    for row in manifests:by.setdefault(row['source_sample_id'],[]).append(row)
    cfg=read(ROOT/'delivery/s-single-intent-fewshot-v1/config/prepare.json')
    metapath=ROOT/cfg['metadata'];maskpath=ROOT/cfg['train_mask']
    for path in (metapath,maskpath):
        require(sha(path.read_bytes())==cfg['pins'][str(path.relative_to(ROOT))], 'locked TRAIN metadata/mask custody')
    metadata_rows=lines(metapath);metadata={r['slurp_id']:r for r in metadata_rows}
    require(len(metadata)==len(metadata_rows), 'unique TRAIN metadata IDs')
    holds={source_id(r['sample_id']) for r in lines(maskpath) if r['reviewed_hold']}
    frozen_ids=set(by)
    require(frozen_ids=={g['sample_id'] for g in groups},'actual frozen545 group roster')
    require(len(frozen_ids)==len(groups), 'unique actual group IDs')
    for sid in frozen_ids:source_id(sid)  # Fail closed for TEST/non-TRAIN, even std=0.
    valid_ids={r['sample_id'] for r in lines(ROOT/'delivery/s-train-dataset-v1/results/splits/S_valid.manifest.private.jsonl')}
    require(not frozen_ids&valid_ids,'S_valid excluded by frozen roster proof')
    chosen=[];seen_groups=set();mechanisms=set()
    counts=dict(variance_zero=0,reviewed_hold=0,eligible_nonzero=0,budget_overflow=0,
        source_group_duplicate=0,mechanism_duplicate=0)
    for group in sorted(groups,key=lambda r:(-r['normalized_std'],r['sample_id'])):
        if group['normalized_std']<=0:
            counts['variance_zero']+=1;continue
        uid=source_id(group['sample_id']);sg=sourcegroups[str(uid)]
        if uid in holds:
            counts['reviewed_hold']+=1;continue
        counts['eligible_nonzero']+=1
        if sg in seen_groups:
            counts['source_group_duplicate']+=1;continue
        members=sorted(group['members'],key=lambda r:(r['advantage'],r['seed']))
        def signature(member):return tuple(sorted(e['type'] for e in member['prediction']))
        mechanism=('protocol' if members[0]['rewards']['normalized']['valid']!=members[-1]['rewards']['normalized']['valid'] else
            'type_or_count' if signature(members[0])!=signature(members[-1]) else 'filler_content')
        if mechanism in mechanisms and len(chosen)>=3:
            counts['mechanism_duplicate']+=1;continue
        require(members[0]['advantage']<0<members[-1]['advantage'],'actual non-tied positive/negative group')
        record=metadata[uid];gold_intent=record['scenario']+'_'+record['action']
        rows=by[group['sample_id']]
        goldrows=[r for r in rows if r['candidate_ids']==[gold_intent]]
        require(len(rows)==3 and len(goldrows)==1, 'exactly one scored gold branch in actual P3')
        goldrow=goldrows[0]
        require(group['sample_id'].rsplit(':',1)[-1] in {Path(r['file']).stem for r in record['recordings']}, 'query recording TRAIN metadata identity')
        require(all(m['gold']==members[0]['gold'] for m in members), 'same gold unit across seeds')
        expected_gold=sorted((e['type'],e['filler'].lower()) for e in annotations(record))
        require(all(sorted((e['type'],e['filler']) for e in m['gold'])==expected_gold for m in members),
            'actual reward gold matches original TRAIN annotations')
        diagnostic=dict(usage='diagnostic_OFFLINE_ONLY',sample_id=group['sample_id'],
            original_intent=record['intent'],scenario=record['scenario'],action=record['action'],
            gold_intent_rule='scenario_action, same as accepted P ledger and prepare_train; original intent retained',
            sentence=record['sentence'],sentence_annotation=record['sentence_annotation'],
            entities=copy.deepcopy(record.get('entities',[])),tokens=copy.deepcopy(record.get('tokens',[])),
            span_unit='original token IDs; entities and tokens retained without span rewriting',
            source=str(metapath.relative_to(ROOT)),source_sha=cfg['pins'][cfg['metadata']],
            record_sha256=sha(canonical(record)),boundary=DIAGNOSTIC_BOUNDARY)
        contexts=[]
        for row in goldrows:
            raw=(ROOT/row['payload']).read_bytes();require(sha(raw)==row['packet_sha256'],'actual context packet custody')
            packet=json.loads(raw);blocks=packet['wire']['messages'][1]['content']
            normalized=copy.deepcopy(packet['wire']);normalized.pop('seed')
            require(sha(canonical(normalized))==custody['policy_wire_digests'][row['sample_id']], 'same-policy context across eight seeds')
            contexts.append(dict(candidate_index=row['candidate_index'],candidate_id=row['candidate_ids'][0],
                branch_id=row['sample_id'],purpose='actual scored gold-intent branch context; shared across8 seeds',
                instruction=packet['wire']['messages'][0]['content'],
                instruction_and_candidate_context=blocks[-1]['text'],
                knowledge=[dict(value_id=b['evidence_id'],source_uri=b['source_uri'],text=b['text']) for b in blocks if b.get('knowledge_role')=='value'],
                real_TRAIN_examples=[dict(example_id=e['example_id'],input=e['input'],annotation=e['annotation'],output=e['output']) for e in packet['examples']],
                packet_sha256=row['packet_sha256'],manifest_sha256=native['manifest_sha256'],
                audio_and_reasoning_excluded=True))
        representative=dict(sample_id=group['sample_id'],source_group=sg,group_id=group['group_id'],
            diagnostic_OFFLINE_ONLY=diagnostic,gold_intent_diagnostic_OFFLINE_ONLY=gold_intent,
            reward_scope=dict(unit='gold_intent_branch',branch_id=goldrow['sample_id'],candidate_id=gold_intent,
                group_size=8,context_seed=2026110901,all_seed_contexts_equal_without_seed=True,
                three_branch_sample_reward=False,end_to_end_reward=False),
            other_branch_indexes=[dict(branch_id=r['sample_id'],candidate_index=r['candidate_index'],
                candidate_id=r['candidate_ids'][0],packet_sha256=r['packet_sha256'],
                purpose='unscored scope index only; no reward or causal comparison supplied') for r in rows if r!=goldrow],
            normalized_mean=group['normalized_mean'],normalized_std=group['normalized_std'],
            positive=members[-1],negative=members[0],actual_input_context=contexts,teacher_update_allowed=True,
            reviewed_hold=False,evaluation_only=False,observed_error_mechanism=mechanism)
        if len(canonical(chosen+[representative]))>CONTEXT_BYTE_LIMIT:
            counts['budget_overflow']+=1;continue
        chosen.append(representative)
        mechanisms.add(mechanism);seen_groups.add(sg)
        if len(chosen)>=6:break
    data['representative_groups']=chosen
    data['selection']=dict(max_groups=6,distinct_source_groups=True,positive_negative_actual_only=True,
        variance_zero_not_winner=True,reviewed_hold_excluded=True,S_valid_excluded=True,
        source_groups_sha256=sha((ROOT/'delivery/recall-card-source-audit-v1/input/train-source-group-index.json').read_bytes()),
        frozen_manifest_sha256=native['manifest_sha256'],full_group_sha256=sha(full.read_bytes()),
        frozen_roster_teacher_eligible_proof=True,S_valid_ID_intersection=0)
    status=('REPRESENTATIVES_READY' if chosen else 'INPUT_BUDGET_INSUFFICIENT' if counts['budget_overflow'] else
        'NO_LEARNABLE_ELIGIBLE_VARIANCE_SIGNAL')
    data['selection'].update(context_bytes=len(canonical(chosen)),context_byte_limit=CONTEXT_BYTE_LIMIT,
        status=status,counts=counts,TEST_excluded=True,diagnostic_boundary=DIAGNOSTIC_BOUNDARY,
        TRAIN_metadata_sha256=cfg['pins'][cfg['metadata']],
        mechanism_classification='observed protocol/type-count/filler heuristic; causal mechanism not proven')
    # Keep bounded request; optional full semantics must supply a bounded summary.
    return data

def prepare_feedback(source,summary,kind):
    digest=sha(source.read_bytes());revision='revision-v5-'+kind+'-'+digest[:12];folder=OUT/'teacher'/revision
    if folder.exists():return revision
    require(len(canonical(summary))<=120000,'bounded feedback only, no whole trajectories')
    evidence=[]
    def add(eid,path,content,metadata):
        evidence.append(dict(evidence_id=eid,source=str(path.relative_to(ROOT)),source_sha256=sha(path.read_bytes()),content=content,metadata=metadata))
    meta=dict(architecture_version='s-single-intent-native-v1',parent_id='prepared-v5-dynamic',unit='sample',
        reward_unit='gold_intent_branch',group_unit='same_sample_same_gold_intent_eight_seeds',
        arm=kind,full_or_partial='full_known_terminal',total=545,time=dict(packaged_epoch=time.time()))
    add('actual-complete-feedback',source,summary,meta)
    for name in ['curator-v4-reviewed.json','reflector-v3-corrected.json']:
        p=OUT/'experience'/name;add(name,p,read(p),meta|dict(unit='experience',total=None))
    p=OUT/'config/grpo-v1.json';add('frozen-reward-policy',p,read(p),meta)
    p=ROOT/'delivery/s-single-intent-fewshot-v1/config/single-intent.txt';add('current-instruction',p,p.read_text(),meta)
    p=ROOT/'delivery/s-single-intent-fewshot-v1/prepared-v5-dynamic/source-policy.json';add('legal-source-policy',p,read(p),meta|dict(unit='TRAIN_source_policy'))
    for name in ('compact-json-v1','absent-slot-v1'):
        p=OUT/'candidates'/name/'experience.json';add(name,p,read(p),meta|dict(arm=name))
    context=dict(schema='s-rolling-ace-evidence-v1',evidence=evidence,
        parent_versions=['prepared-v5-dynamic/single_intent_fewshot'],
        feedback_source_kind=kind,TEST_feedback=False,FoldA_FoldB_feedback=False,S_valid_feedback=False,
        prior_teacher_cost_tokens=118691,metric='frozen accepted typed primary; strict/official supplementary; metric change not gain',
        local_vetoes=['casing-only','universal exact-spoken/verbatim fidelity from historical cases',
            'blanket type-name equality omission','repeated compact/absence/timezone/complement without new evidence'],
        rootcause_boundary='Only complete current parent contrasts or actual same-policy advantages; no historical batched cases supplied',
        runtime_patch_compiler=dict(allowed='instruction_append only, max1200 chars, one explicit current parent, no business knowledge or examples',
            no_code=True,no_runtime_gold=True,no_schema_or_scoring_change=True,parameters_frozen=True),
        pending=['If source reports no stable semantic gain, return0 proposals and reflect failed experiences'])
    folder.mkdir(parents=True,mode=0o700)
    template=(OUT/'config/training-free-grpo-template-v1.md').read_text()+'\n\n'+DIAGNOSTIC_BOUNDARY+'\n'
    p=OUT/'config/tfgrpo-operations-v1.json'
    add('bounded-operations',p,read(p),meta|dict(unit='fixed_generic_operation_catalog'))
    save(folder/'dynamic-context.private.json',context);(folder/'template.md').write_text(template)
    save(folder/'messages.private.json',[dict(role='user',content=json.dumps(context,ensure_ascii=False))])
    cfg=copy.deepcopy(read(OUT/'teacher/revision-v4-parent-unit-correction/executor-config.private.json'));cfg['output_root']=str(folder/'response')
    cfg['analysis_budget']['max_hypotheses']=2
    for field,name in [('messages','messages.private.json'),('research_prompt','template.md')]:cfg[field]=dict(path=str(folder/name),sha256=sha((folder/name).read_bytes()))
    save(folder/'executor-config.private.json',cfg)
    save(folder/'REQUEST-PINS.json',{str(p.relative_to(folder)):sha(p.read_bytes()) for p in folder.iterdir() if p.is_file()})
    save(folder/'REQUEST_READY.json',dict(at=time.time(),status='COMPLETE_FEEDBACK_NEW_REVISION',API_sent=False,
        source_sha256=digest,automatic_retry=0,source_kind=kind,TEST=False,S_valid=False))
    return revision

def compile_and_prepare(revision):
    folder=OUT/'teacher'/revision
    if (folder/'LOCAL-REVIEW.json').exists():return
    if not (folder/'teacher-final.json').exists():return
    result=read(folder/'teacher-final.json');known={e['evidence_id'] for e in read(folder/'dynamic-context.private.json')['evidence']}
    require(len(result['curator']['proposals'])<=2 and len(result['curator']['experience_entries'])<=2,'TFGRPO max2 proposals/entries; no admission')
    decisions=[]
    for i,proposal in enumerate(result['curator']['proposals']):
        reasons=[];patch=proposal.get('patch',{});value=patch.get('value');parent=proposal.get('parent_version')
        operations=read(OUT/'config/tfgrpo-operations-v1.json')
        opid=value.get('operation_id') if isinstance(value,dict) and set(value)=={'operation_id'} else None
        if patch.get('target')!='instruction_append' or opid not in operations['operations']:reasons.append('unsupported operation; no arbitrary prose/business knowledge accepted')
        if parent!=operations['parent_version']:reasons.append('not actual dynamic fewshot group parent')
        refs=set(proposal.get('source_evidence_ids',[]))
        if not refs<=known or 'actual-complete-feedback' not in refs:reasons.append('no current full feedback support')
        if opid in operations['preexisting_redundant_operations']:reasons.append('semantic mechanism already explicit in parent instruction')
        if opid in operations['operations'] and proposal.get('mechanism_family')!=operations['operations'][opid]['mechanism_family']:reasons.append('mechanism family mismatch')
        support=proposal.get('semantic_advantage')
        if not isinstance(support,dict):reasons.append('structured semantic advantage support required')
        else:
            groups=read(folder/'dynamic-context.private.json')['evidence'][0]['content']['representative_groups']
            ids={g['sample_id'] for g in groups}
            if not set(support.get('positive_sample_ids',[]))<=ids or not support.get('positive_sample_ids'):reasons.append('actual group sample support absent')
            if not support.get('counterexample_sample_id') in ids:reasons.append('actual group counterexample required')
            if not isinstance(support.get('explanation'),str) or not support['explanation'].strip():reasons.append('nonempty semantic explanation required; IDs alone are insufficient')
        if opid in operations['operations']:value=operations['operations'][opid]['instruction']
        if reasons:
            decisions.append(dict(index=i,status='NO_ADMISSION',reasons=reasons,proposal=proposal));continue
        name='tfgrpo-feedback-'+sha(canonical(proposal))[:12]
        entry=dict(mechanism=proposal['mechanism'],instruction=value,evidence=proposal['source_evidence_ids'],
            counterevidence=proposal['counterevidence'],falsification=proposal['falsification'],
            semantic_advantage=support,causal_status='BOUNDED_UNTESTED_HYPOTHESIS_NOT_PROVEN',
            rollback=proposal['rollback'],applicability=proposal['applicability'])
        save(OUT/'config'/('teacher-compiled-'+name+'.json'),dict(admission='ACCEPT_SINGLE_MECHANISM_HYPOTHESIS',
            target='instruction_append',parent_version=parent,experience=entry,teacher_final_sha256=sha((folder/'teacher-final.json').read_bytes())))
        task=dict(task_id=name,prepared='delivery/s-tfrl-rolling-v1/candidates/'+name,
            config='delivery/s-tfrl-rolling-v1/candidates/'+name+'/single_intent_fewshot/native-UNBOUND.json',
            preferred_lane='ABCD'[i%4],parent_version=parent,method='TFGRPO_actual_group_procedure_delta',
            purpose='test current full-feedback supported single mechanism; no weights update')
        try:
            producer.prepare(task);producer.verify(task);producer.publish(task)
            decisions.append(dict(index=i,status='CPU_READY_NOT_PROVEN_GAIN',task=task))
        except Exception as e:decisions.append(dict(index=i,status='CPU_FAILED_NOT_ADMITTED',error=str(e),task=task))
    save(folder/'LOCAL-REVIEW.json',dict(at=time.time(),status='BOUNDED_COMPLETE_FEEDBACK_REVIEW',decisions=decisions,
        no_arbitrary_code=True,no_weight_update=True,gain_proven=False))
    save(OUT/'experience'/('curator-'+revision+'.json'),dict(at=time.time(),parent_revision=revision,
        entries=result['curator']['experience_entries'],retirements=result['curator']['retirements'],local_decisions=decisions,
        source_feedback='actual known complete parent/group evidence',gain_proven=False))

def main():
    lock=(OUT/'feedback-teacher.guard').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    end=read(OUT/'REQUESTS/QUEUE-DEFICIT.json')['window']['end_epoch'];seen=set()
    save(OUT/'FEEDBACK-TEACHER-START.json',dict(pid=os.getpid(),ppid=os.getppid(),at=time.time(),interval_seconds=600,
        max_new_rounds=1,automatic_retry=0,source='actual8seed advantages with context; complete semantics optional supplementary',S_valid=False))
    while time.time()<end:
        try:
            sources=[]
            marker=OUT/'REQUESTS/SEMANTIC-FULL-FEEDBACK-READY.json'
            semantic_context=semantic_supplement(marker)
            group=ROOT/'delivery/s-single-intent-execution-v1/rolling-state/group-feedback/REFLECTOR-NEXT-INPUT.private.json'
            if group.exists():
                d=read(group);require(d['samples']==545 and d['group_size']==8 and d['cross_policy_pooled'] is False,'actual complete group')
                d=group_context(d)
                if not d['representative_groups']:
                    status=d['selection']['status']
                    name='GROUP-INPUT-BUDGET-INSUFFICIENT.json' if status=='INPUT_BUDGET_INSUFFICIENT' else 'GROUP-NO-SIGNAL.json'
                    save(OUT/name,dict(at=time.time(),status=status,selection=d['selection'],
                        group_source_sha256=sha(group.read_bytes()),new_API_calls=0))
                    seen.add('group');sources=[]
                elif semantic_context is not None and len(canonical(d))+len(canonical(semantic_context))<=120000:d['complete_parent_comparison']=semantic_context
                if d['representative_groups']:sources.append((group,d,'group'))
            for source,summary,kind in sources:
                digest=sha(source.read_bytes())
                if kind in seen:continue
                require(not summary.get('evaluation_only',False) and not summary.get('S_valid_feedback',False),'validation never teacher')
                quota=OUT/'teacher/WINDOW-GROUP-REQUEST-QUOTA.json'
                # Resume the consumed revision even if the upstream summary changed.
                revision=read(quota)['revision'] if quota.exists() else prepare_feedback(source,summary,kind)
                folder=OUT/'teacher'/revision
                prior_intents=list((OUT/'teacher').glob('revision-v5-*/launch-intent.json'))
                require(not prior_intents or prior_intents==[folder/'launch-intent.json'],'one new group teacher request in window')
                if dispatch_permission(folder,quota) and os.environ.get('S_TFRL_TEACHER_LOCAL_ONLY')!='1':
                    save(quota,dict(revision=revision,at=time.time(),max_new_requests=1,consumed=True,unknown_no_resend=True))
                    subprocess.run([sys.executable,'-B',str(OUT/'code/teacher_round.py'),'dispatch','--revision',revision],check=True)
                # Wait only from durable actual terminal; launch intent without terminal is never re-dispatched.
                if (folder/'TEACHER-TERMINAL.json').exists():
                    compile_and_prepare(revision)
                    # Terminal is written before collect; do not lose that short race.
                    if (folder/'LOCAL-REVIEW.json').exists() or (folder/'COLLECT-ERROR.json').exists():seen.add(kind)
            save(OUT/'FEEDBACK-TEACHER-STATE.json',dict(at=time.time(),processed_source_kinds=sorted(seen),
                status='WAIT_COMPLETE_GROUP_FEEDBACK_OR_TEACHER_TERMINAL',max_rounds=1,no_retry=True,S_valid=False,
                local_only=os.environ.get('S_TFRL_TEACHER_LOCAL_ONLY')=='1'))
        except Exception as e:save(OUT/'FEEDBACK-TEACHER-ERROR.json',dict(at=time.time(),error=str(e),traceback=traceback.format_exc(),retry_API=False))
        time.sleep(min(600,max(0,end-time.time())))

if __name__=='__main__':main()
