"""Prepare source-bound real K4 experience banks from verified sealed requests.

Uses the existing deterministic contrast distiller. No optional observer trace
or historical pin file supplies evidence. The caller supplies an exact token
measure and must register its tokenizer budget before executing it.
"""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis'),str(ROOT/'scripts/flights/nx')]
import stage3_rollout_score as s3
import tfrl_v2_distiller as distiller
import tfrl_v2_text as text
import tfrl_v2_sampler as sampler
import tfrl_v2_common as common
from speech_aware_evidence_acquisition.core.rollout_trace import usage_judgements,supply_items


def sealed_answers(bundle,plan,*,resolver=None,bank_context=None):
    """First-visit answers and exact evidence, keyed by real row and arm.

    Requires verify_bundle and the plan/request audit before invocation. The
    request/response hashes are checked again at this adapter boundary. No
    invented contact timestamp, optional trace or regenerated prompt is used.
    """
    expected=[r['row_id'] for r in plan['row_plan']]
    rows={r['sample_id']:r for r in bundle.outputs_rows}
    if len(rows)!=len(bundle.outputs_rows) or set(rows)!=set(expected):raise ValueError('whole-half answers required')
    answers={};texts={};provenance=[]
    if plan['iteration'] not in (1,2,3):raise ValueError('registered iteration1..3 required')
    if plan['iteration']>1 and bank_context is None:
        raise ValueError('banked distillation requires verified request context')
    if bank_context is not None:
        from speech_aware_evidence_acquisition.rails.tfrl_bank_request_audit import verify_group
        requests=[v['payload'] for k,v in bundle.trace_records.items() if k.endswith('/request')]
        for row in rows.values():
            if row.get('supply_variants') is None and any(v.get('stage')=='solver' for v in row['native_episode_receipt']['role_visits']):
                raise ValueError('banked Solver row lacks arm custody')
            if row.get('supply_variants') is not None:
                verify_group(row['sample_id'],row['supply_variants']['arms'],requests,**bank_context)

    for sid in expected:
        from tfrl_seed_request_audit import no_solver_abstention
        if (not any(v.get('stage')=='solver' for v in rows[sid]['native_episode_receipt']['role_visits'])
                and no_solver_abstention(rows[sid])):
            provenance.append({'row_id':sid,'availability':'no_solver_visit',
                'candidate_answers':0,'scoring_fallback':'canonical_empty',
                'distillation':'no_candidate_evidence'})
            continue
        episode=rows[sid]['native_episode_receipt']
        visits=[v for v in episode['role_visits'] if v.get('stage')=='solver']
        if len(visits)!=1:raise ValueError('current seven-contact program requires one Solver visit')
        contacts=visits[0]['execution_receipt']['contacts']
        if len(contacts)!=6:raise ValueError('complete K4 plus twin and lock required')
        normaliser=(episode.get('deliver_normaliser') or {}).get('normaliser_id')
        for arm,contact in zip(('s1','s2','s3','s4','aa','lock'),contacts):
            rid=contact['request_id'];request=bundle.trace_records[rid+'/request'];response=bundle.trace_records[rid+'/response']
            raw=response['text'];payload=request['payload']
            if (payload['sample_id']!=sid or request['payload_sha256']!=contact['request_sha256']
                    or hashlib.sha256(raw.encode()).hexdigest()!=contact['response_text_sha256']
                    or response['response_text_sha256']!=contact['response_text_sha256']):
                raise ValueError('sealed contact provenance differs')
            rendered,ok=s3.normalise_arm_answer(raw,normaliser)
            answer=json.loads(rendered);answer['parsed']=ok
            evidence={};inherited=[]
            for item in payload.get('supplied_evidence',[]):
                key,content=item['evidence_id'],item['content']
                from speech_aware_evidence_acquisition.rails.tfrl_v2_consumption import EXEMPLAR_EVIDENCE_ID
                if key==EXEMPLAR_EVIDENCE_ID:
                    if bank_context is None:raise ValueError('inherited exemplar lacks verified bank context')
                    inherited.append({'evidence_id':key,'source_uri':item['source_uri'],
                        'content_sha256':hashlib.sha256(content.encode()).hexdigest(),
                        'bank_receipt_sha256':bank_context['bank'].receipt_sha256,
                        'distillation':'inherited context; excluded from fresh evidence exemplars'})
                    continue
                if not isinstance(content,str) or not content:raise ValueError('empty evidence content')
                if key in evidence:raise ValueError('duplicate supplied evidence identity')
                if key in texts and texts[key]!=content:raise ValueError('one evidence identity has different actual text')
                evidence[key]=content;texts[key]=content
            judged=usage_judgements(evidence,answer)
            used={k:v[0] for k,v in judged.items()};bases={k:v[1] for k,v in judged.items()}
            answers[sid,arm]={'answer':answer,'supply_items':supply_items(evidence,used=used,bases=bases)}
            provenance.append({'row_id':sid,'arm':arm,'request_id':rid,'request_sha256':contact['request_sha256'],
                'response_text_sha256':contact['response_text_sha256'],'normalized_answer_sha256':hashlib.sha256(rendered.encode()).hexdigest(),
                **({'inherited_bank_context':inherited} if inherited else {})})
    if not texts and resolver is None:raise ValueError('verified knowledge resolver required without supplied evidence')
    if resolver is None:
        resolver=text.ItemTextResolver(texts,source={'kind':'verified-sealed-model-requests','run_id':plan['run_id'],
            'attempt_id':plan['attempt_id'],'raw_trace_sha256':bundle.manifest['raw_trace']['sha256']})
    else:
        for key,content in texts.items():
            if resolver.resolve(key).content!=content:
                raise ValueError('actual supplied text differs from verified knowledge binding')
    return answers,resolver,provenance


def answers_for_loaded(loaded,*,resolver=None):
    """Keep trace namespaces separate when a half is an explicit source union."""
    if 'segments' not in loaded:
        return sealed_answers(loaded['bundle'],loaded['plan'],resolver=resolver,bank_context=loaded.get('bank_context'))
    if resolver is None:raise ValueError('source union requires verified common knowledge resolver')
    ids=[r['row_id'] for r in loaded['plan']['row_plan']];seen=set();answers={};provenance=[]
    for segment in loaded['segments']:
        own=[r['row_id'] for r in segment['plan']['row_plan']]
        if len(own)!=len(set(own)) or seen.intersection(own):raise ValueError('source union repeats rows')
        seen.update(own)
        local,_,traces=sealed_answers(segment['rows'],segment['plan'],resolver=resolver)
        answers.update(local)
        provenance.extend({**r,'source_namespace':segment['namespace']} for r in traces)
    if len(ids)!=len(set(ids)) or seen!=set(ids):raise ValueError('source union omits full-half rows')
    return answers,resolver,provenance


def build(loaded,measure,*,resolver=None):
    """Build banks from load_half's verified results; no filesystem mutation.

    The injected vLLM measure may contact the registered tokenizer; this is
    accounted separately from generation. Estimates are refused before contact.
    """
    if measure.name!=common.MEASURE_VLLM:raise ValueError('exact registered tokenizer measure required')
    plan=loaded['plan'];half=plan['half'];iteration=plan['iteration']
    answers,resolver,provenance=answers_for_loaded(loaded,resolver=resolver)
    calls_before=measure.calls
    ids=[r['row_id'] for r in plan['row_plan']]
    tables={a:{r['sample_id']:r for r in result['per_sample']} for a,result in loaded['results'].items()}
    credit=sampler.credit_rows(ids,iteration,4,sampler.scores_from_counts(tables),twin_arm='aa')
    if credit!=loaded['credit']:raise ValueError('distillation credit differs from verified official read')
    abstentions={p['row_id'] for p in provenance if p.get('availability')=='no_solver_visit'}
    if any(r['positive'] for r in credit if r['row_id'] in abstentions):
        raise ValueError('no-Solver abstention cannot supply positive trajectory credit')
    exemplar,fate=distiller.build_exemplar_bank(credit,answers,iteration=iteration,source_half=half,resolver=resolver,measure=measure)
    pairs=distiller.build_contrast_pairs(credit,answers,iteration)
    rules,refused=distiller.build_rule_bank(pairs,iteration=iteration,source_half=half,measure=measure)
    source_rows=sorted({e['source_row_id'] for s in exemplar['by_scenario'].values() for e in s['entries']}
        | {r for rule in rules['rules'] for r in rule['support_rows']})
    receipt={'schema':distiller.RECEIPT_SCHEMA,'loop_id':plan['loop_id'],'iteration':iteration,'source_half':half,
        'kind':'half','measure':measure.name,'measure_detail':dict(measure.detail()),'exemplar_text':exemplar['exemplar_text'],
        'exemplar_text_source':resolver.identity(),'source_halves':[half],'consuming_half':common.other_half(half),
        'source_split_identities':{half:loaded['roster']['split_identity_hash']},
        'source_roster_file_sha256':plan['roster']['file_sha256'],'source_half_rows':len(ids),'source_row_ids':source_rows,
        'verified_source':loaded['source'],
        'exemplar_bank':{'content_sha256':common.content_sha256(exemplar),'scenarios':len(exemplar['by_scenario']),
            'entries':sum(len(s['entries']) for s in exemplar['by_scenario'].values()),
            'max_block_tokens':max((s['block_tokens'] for s in exemplar['by_scenario'].values()),default=0),
            'block_token_cap':common.EXEMPLAR_BLOCK_TOKEN_CAP,'measure':measure.name},
        'rule_bank':{'content_sha256':common.content_sha256(rules),'rules':len(rules['rules']),
            'rendered_tokens':rules['rendered_tokens'],'token_cap':common.RULE_BANK_TOKEN_CAP,
            'vote_min':common.VOTE_MIN_ROWS,'measure':measure.name},
        'credit_summary':sampler.credit_summary(credit),
        'counts':{'rows_in_half':len(ids),'rows_used':len(ids)-len(abstentions),'rows_skipped':len(abstentions),'no_solver_abstentions':len(abstentions),'scored_rows':len(ids),'contrast_pairs':len(pairs),'rules_refused':len(refused)},
        'method':'Existing deterministic answer-contrast rules and resolved evidence exemplars; verified current seed-only trajectories.' if iteration==1 else 'Verified banked K4 answers; fresh exemplars use original knowledge evidence only, inherited bank blocks retain provenance and are not recursively quoted.',
        'model_calls':0,'tokenizer_calls':measure.calls-calls_before,'next_iteration_authorized':False}
    # Actual timestamps are not available here; do not fabricate rollout contact/TFRL lines.
    return {'credit':credit,'pairs':pairs,'exemplar_bank':exemplar,'rule_bank':rules,'refused':refused,
        'provenance':provenance,'fate':fate,'receipt_core':receipt}


def common_resolver(halves,data):
    from speech_aware_evidence_acquisition.rails.knowledge_stores import load_knowledge_store_binding
    first,second=halves['h1']['config'].values,halves['h2']['config'].values
    if first['knowledge_stores']!=second['knowledge_stores']:
        raise ValueError('halves must use the same verified knowledge stores')
    binding=load_knowledge_store_binding(first,data_root=data)
    if binding is None:raise ValueError('resolved knowledge binding required')
    resolver=text.ItemTextResolver.from_binding(binding)
    # Check BOTH halves before any token contact, including evidence of non-positive rows.
    for loaded in halves.values():answers_for_loaded(loaded,resolver=resolver)
    return resolver


def write_bank(result,out):
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    files={'credit.jsonl':common.jsonl_bytes(result['credit']),
        'contrast-pairs.jsonl':common.jsonl_bytes(result['pairs']),
        'exemplar-bank.json':common.file_bytes(result['exemplar_bank']),
        'rule-bank.json':common.file_bytes(result['rule_bank']),
        'refused-rules.jsonl':common.jsonl_bytes(result['refused']),
        'trajectory-source.jsonl':common.jsonl_bytes(result['provenance']),
        'exemplar-fate.json':common.file_bytes(result['fate'])}
    receipt=dict(result['receipt_core'])
    receipt['files']={name:{'sha256':common.sha256_bytes(raw),'bytes':len(raw)} for name,raw in files.items()}
    for key in ('exemplar_bank','rule_bank'):
        receipt[key]=dict(receipt[key],file=key.replace('_','-')+'.json')
    files['bank-receipt.json']=common.file_bytes(receipt)
    out.mkdir(parents=True,exist_ok=True)
    for name,raw in files.items():
        path=out/name
        if path.exists():
            if path.read_bytes()!=raw:raise ValueError('existing bank artifact differs')
        else:publish_new(path,raw)
    return receipt
