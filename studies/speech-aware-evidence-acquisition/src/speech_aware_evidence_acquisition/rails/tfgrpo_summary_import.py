"""Verify historical summaries as learning inputs, never as fresh model calls."""
import base64
import hashlib
import json
from pathlib import Path
from ..contracts import ExecutionPlan
from ..core.contact_journal import ContactJournal
from .tfgrpo_learning import LearningError,sha,render
from .tfgrpo_projection import summary_trajectory


def load_import(spec, *, data_root, groups, common, model_id):
    root=Path(data_root).resolve()
    def pinned(pin):
        if set(pin)!={'relpath','sha256'}:raise LearningError('exact historical input pin required')
        path=(root/pin['relpath']).resolve()
        if not path.is_relative_to(root):raise LearningError('historical input escaped data root')
        blob=path.read_bytes()
        if hashlib.sha256(blob).hexdigest()!=pin['sha256']:raise LearningError('historical input changed')
        return path,json.loads(blob)
    _,doc=pinned(spec)
    if set(doc)!={'schema','source_plan','source_attempt','source_batch','source_bridge','records'} or doc['schema']!='saea-tfgrpo-summary-import-v1':
        raise LearningError('exact historical summary import required')
    plan=ExecutionPlan(**doc['source_plan']);plan.validate()
    ap,attempt=pinned(doc['source_attempt'])
    if (ap!=root/'attempts'/plan.run_id/(plan.attempt_id+'.json')
            or attempt.get('state')!='ABORTED' or attempt.get('contact_kind')!='prepared-text'
            or attempt.get('run_id')!=plan.run_id or attempt.get('attempt_id')!=plan.attempt_id
            or attempt.get('carrier_lock_key')!=plan.carrier_lock_key
            or attempt.get('reserved')!={'calls':plan.planned_model_calls,'audio_seconds':float(plan.planned_speech_audio_seconds)}):
        raise LearningError('historical learning source must be explicitly aborted')
    _,old=pinned(doc['source_batch']);oldbatch=old['batch']
    # The rollout policy, reward, source text and learning task remain identical.
    if oldbatch['model_id']!=model_id or oldbatch['groups']!=groups or oldbatch['common']!=common:
        raise LearningError('historical summary learning inputs differ')
    journal=ContactJournal(root,plan,'prepared-context-text')
    observed={};totals={'calls':0,'prompt_tokens':0,'completion_tokens':0}
    for path in journal.root.glob('*.intent.json'):
        intent=json.loads(path.read_bytes());record=journal.inspect(intent['request_id'])
        if record['state']!='observed':raise LearningError('historical generation outcome unknown')
        response=record['response'];body=json.loads(base64.b64decode(response['response_b64']))
        if intent['payload'].get('prepared_text')!='saea-prepared-text-v1':raise LearningError('historical text-only source required')
        observed[intent['request_id']]=(record,body)
        totals['calls']+=1
        for key in ('prompt_tokens','completion_tokens'):
            value=body['usage'][key]
            if type(value)is not int or value<0:raise LearningError('historical usage invalid')
            totals[key]+=value
    expected_responses={journal._path(rid,'response') for rid in observed}
    if set(journal.root.glob('*.response.json'))!=expected_responses or attempt['actual']!={'calls':totals['calls'],'audio_seconds':0.0}:
        raise LearningError('historical attempt cost differs from all observed contacts')
    if doc['source_bridge']!=str(Path('runs')/plan.run_id/plan.attempt_id/'tfgrpo/contacts'):
        raise LearningError('historical bridge path differs')
    lookup={r['trajectory_id']:(g,r) for g in groups for r in g['rollouts']}
    summaries={};used=set()
    if not isinstance(doc['records'],list) or not doc['records']:raise LearningError('nonempty historical summaries required')
    for row in doc['records']:
        if set(row)!={'trajectory_id','intent','response'}:raise LearningError('exact summary record required')
        tid=row['trajectory_id']
        if tid not in lookup or tid in summaries:raise LearningError('duplicate or foreign historical summary')
        group,roll=lookup[tid];ip,intent=pinned(row['intent']);rp,response=pinned(row['response'])
        key=sha({'stage':'summary','identity':tid});bridge=root/doc['source_bridge']
        if ip!=bridge/(key+'.intent.json') or rp!=bridge/(key+'.response.json'):raise LearningError('summary bridge filename differs')
        request=intent['request'];receipt=response['response']
        messages=[{'role':'system','content':render('SINGLE_ROLLOUT_SUMMARY_TEMPLATE_SP',**common)},
                  {'role':'user','content':render('SINGLE_ROLLOUT_SUMMARY_TEMPLATE_UP',question=group['query'],answer=group['answer'],
                    critique=roll['critique'],trajectory=summary_trajectory(group['query'],roll['trajectory']))}]
        if (request['stage']!='summary' or request['identity']!=tid or request['messages']!=messages
                or request['batch_sha256']!=sha(oldbatch) or request['learning_config_sha256']!=oldbatch['learning_config_sha256']
                or request['model_id']!=model_id or response['request_sha256']!=sha(request)):
            raise LearningError('historical summary prompt or generation configuration differs')
        prefix=f'{plan.run_id}/{plan.attempt_id}/';rid=receipt['receipt_id'].removeprefix(prefix)
        if receipt['receipt_id']!=prefix+rid or rid not in observed or rid in used:raise LearningError('historical response identity differs')
        record,body=observed[rid];payload=record['intent']['payload']
        if (receipt['model_id']!=model_id or receipt['text']!=body.get('text') or not isinstance(receipt['text'],str) or not receipt['text'].strip()
                or body.get('finish_reason')!='stop' or receipt['usage']!={'calls':1,**{k:body['usage'][k] for k in ('prompt_tokens','completion_tokens')}}
                or receipt['response_sha256']!=record['response']['response_sha256']
                or payload['sample_id']!=group['query_id']
                or {**intent['payload'],'request_id':rid}!=payload or receipt['request_sha256']!=record['intent']['request_sha256']):
            raise LearningError('historical summary response or usage differs')
        context=json.loads(payload['history'][0]['content'])
        if context.get('source_messages')!=messages or context.get('stage')!='summary':raise LearningError('historical native request messages differ')
        used.add(rid);summaries[tid]=receipt['text']
    return summaries,{'schema':'saea-tfgrpo-historical-summary-custody-v1','source':spec,
        'source_run_id':plan.run_id,'source_attempt_id':plan.attempt_id,'source_learning_config_sha256':oldbatch['learning_config_sha256'],
        'imported_summaries':len(summaries),'original_total_usage':totals,'source_all_responses_observed':True,
        'classification':'Historical learning inputs; excluded from fresh generation counters and receipts.'}
