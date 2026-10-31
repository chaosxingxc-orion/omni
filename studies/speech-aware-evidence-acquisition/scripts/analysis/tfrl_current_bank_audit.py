"""Audit actual K4 requests with one verified cross-fit bank, preserving controls.

Only exact declared rule suffix and the Perception-selected exemplar block are
projected out before the existing strict seed/context check. This does not
change iteration1 verification or authorize any generation/bank consumption.
"""
import copy,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/flights/nx')]
from speech_aware_evidence_acquisition.rails.trajectory_seeds import verify_recorded_group
from speech_aware_evidence_acquisition.rails import tfrl_v2_consumption as rail


def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def digest(s):return hashlib.sha256(s.encode()).hexdigest()


def checked_control(request):
    control=request['history'][-1]
    if (control.get('role')!='control' or digest(control['content'])!=control['content_sha256']):
        raise ValueError('control history custody differs')
    value=json.loads(control['content'])
    if (value['contact_id']!=control['contact_id'] or value['sample_id']!=request['sample_id']
            or value['messages'][0]!={'role':'system','content':request['task_instruction']}):
        raise ValueError('framework control and actual instruction differ')
    return value


def project_candidate(request,lock,bank,perception_phase):
    """Verify each augmentation against bank bytes before removing it for audit."""
    result=copy.deepcopy(request);control=checked_control(request);checked_control(lock)
    expected_instruction=lock['task_instruction']
    if bank.rule_text():expected_instruction+=rail.RULES_SEPARATOR+bank.rule_text()
    if request['task_instruction']!=expected_instruction:raise ValueError('candidate rule suffix differs from verified bank')
    states=[SimpleNamespace(**r) for r in request['history'] if r.get('role')=='agent_state']
    intent=rail.predicted_intent_id(states,'phase:'+perception_phase)
    block=bank.exemplar_block_for_intent(intent)
    evidence=result['supplied_evidence'];provenance=result['evidence_provenance']
    if len(evidence)!=len(provenance):raise ValueError('evidence and provenance cardinalities differ')
    expected=list(lock['supplied_evidence'])
    if block:
        expected.append({'evidence_id':rail.EXEMPLAR_EVIDENCE_ID,'source_uri':bank.source_uri,'content':block})
    if evidence!=expected:raise ValueError('candidate exemplar or ordinary evidence differs')
    if block:
        index=len(evidence)-1
        if provenance[-1]!={'lock_key':'rails-supply','relpath':f"rails-message/{request['request_id']}/{index}",'sha256':digest(block)}:
            raise ValueError('bank exemplar provenance differs')
        evidence.pop();provenance.pop()
    result['task_instruction']=lock['task_instruction']
    control['messages'][0]={'role':'system','content':lock['task_instruction']}
    result['history'][-1]['content']=canonical(control)
    result['history'][-1]['content_sha256']=digest(result['history'][-1]['content'])
    return result


def verify_group(row_id,arms,requests,*,bank,perception_phase='perception'):
    if bank.receipt_sha256!=rail.receipt_sha256(bank.receipt):raise ValueError('verified bank receipt changed')
    indexed={a['contact_id']:a for a in arms};own=[]
    for req in requests:
        if req.get('sample_id')!=row_id:continue
        history=req.get('history',[])
        if history and history[-1].get('contact_id') in indexed:
            checked_control(req);own.append(req)
    lock_rows=[r for r in own if r['history'][-1]['contact_id']==arms[-1]['contact_id']]
    if not lock_rows:raise ValueError('missing control lock request')
    lock=lock_rows[0]
    if any(e.get('evidence_id')==rail.EXEMPLAR_EVIDENCE_ID for e in lock['supplied_evidence']):
        raise ValueError('bank leaked into control evidence')
    projected=[];augmented=0
    for request in own:
        arm=indexed[request['history'][-1]['contact_id']]
        if arm['arm_id'] in ('s1','s2','s3','s4'):
            request=project_candidate(request,lock,bank,perception_phase);augmented+=1
        # Reask content is verified by its recorded request custody; initial
        # context equality remains the existing first-request seed check.
        projected.append(request)
    count=verify_recorded_group(row_id,arms,projected)
    # Protect all request fields outside the known routing/context/decoding
    # fields, not just the selected fields of the legacy seed check.
    exclusions={'request_id','history','evidence_provenance','decoding_params'}
    first={}
    for r in projected:first.setdefault(r['history'][-1]['contact_id'],r)
    expected={k:v for k,v in lock.items() if k not in exclusions}
    if any({k:v for k,v in r.items() if k not in exclusions}!=expected for r in first.values()):
        raise ValueError('unregistered request field changed')
    return {'solver_requests_verified':count,'bank_candidate_requests':augmented,'bank_receipt_sha256':bank.receipt_sha256,
            'control_bank_requests':0,'new_model_calls':0}


def verify(plan,outputs,trace,*,bank,perception_phase='perception'):
    """Require whole-half custody and exact planned previous-iteration bank."""
    import tfrl_v2_sampler as sampler
    from tfrl_seed_request_audit import no_solver_abstention
    columns=sampler.execution_columns(plan)
    expected={c['column_id']:c['decoding'] for c in columns};expected.update(aa=None,lock=None)
    ids=[r['sample_id'] for r in outputs]
    if len(ids)!=len(set(ids)) or set(ids)!={r['row_id'] for r in plan['row_plan']}:
        raise ValueError('bank audit requires exact planned full half')
    if (plan['iteration'] not in (2,3) or bank.receipt['iteration']!=plan['iteration']-1
            or plan['banks']['bank_receipt_sha256']!=sampler.cm.content_sha256(bank.receipt)
            or plan['banks']['rule_bank_sha256']!=bank.receipt['rule_bank']['content_sha256']
            or plan['banks']['exemplar_bank_sha256']!=bank.receipt['exemplar_bank']['content_sha256']):
        raise ValueError('planned previous-iteration bank differs')
    sampler.cm.check_cross_fit(bank.receipt,plan['half'],consuming_row_ids=ids,
                              consuming_split_identity=plan['roster']['split_identity_hash'])
    sampler.cm.check_bank_measured(bank.receipt)
    requests=[]
    for key,value in trace.items():
        if not key.endswith('/request'):continue
        payload=value['payload']
        if digest(canonical(payload))!=value['payload_sha256']:raise ValueError('request payload hash differs')
        requests.append(payload)
    total=0;candidates=0;abstentions=0
    for row in outputs:
        if no_solver_abstention(row):abstentions+=1;continue
        receipt=row['supply_variants'];arms=receipt['arms']
        if (receipt['schema']!='saea-supply-variant-arms-v6' or [a['arm_id'] for a in arms]!=list(expected)
                or any(a['decoding_override']!=expected[a['arm_id']] for a in arms)):
            raise ValueError('bank arm group differs from actual registered plan')
        checked=verify_group(row['sample_id'],arms,requests,bank=bank,perception_phase=perception_phase)
        total+=checked['solver_requests_verified'];candidates+=checked['bank_candidate_requests']
    return {'schema':'saea-tfrl-banked-plan-request-check-v1','rows':len(ids),'solver_requests_verified':total,
            'bank_candidate_requests':candidates,'no_solver_abstention_rows':abstentions,
            'bank_receipt_sha256':bank.receipt_sha256,'new_model_calls':0,
            'scope':'Exact bank augmentation plus original control context; independently verify source bundle and scores.'}


def verify_for_values(plan,outputs,trace,*,values,data):
    """Production reader entry: verify bank files and cross-fit before credit."""
    from tfrl_seed_request_audit import verify as plain_verify
    declaration=rail.declaration_of_values(values)
    if declaration is None:
        if plan['iteration']!=1 or plan['banks']['bank_receipt_sha256'] is not None:
            raise ValueError('later iteration lacks declared bank')
        return plain_verify(plan,outputs,trace)
    bank=rail.verify_for_flight(values,data,split_identity_hash=plan['roster']['split_identity_hash'],
                               consuming_row_ids=[r['row_id'] for r in plan['row_plan']])
    role=values['native_episode']['roles']['perception']
    phase=role.get('phase_id','perception')
    return verify(plan,outputs,trace,bank=bank,perception_phase=phase)
