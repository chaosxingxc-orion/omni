"""Pure verification of the exact bank augmentation on seed-only requests.

Callers must first bind the bank files and declaration to the composed config.
No model calls, filesystem mutation, or exception to ordinary control custody.
"""
import copy,hashlib,json
from types import SimpleNamespace
from .trajectory_seeds import verify_recorded_group
from . import tfrl_v2_consumption as rail

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


def project_candidate(request,lock,bank,perception_phase,allocation=None):
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
    original=list(lock['supplied_evidence'])
    indices=list(range(len(original)))
    if allocation is not None:
        from .tfrl_supply_allocation import keep_indices
        indices=keep_indices(original,allocation)
    expected=[original[i] for i in indices]
    if block:
        expected.append({'evidence_id':rail.EXEMPLAR_EVIDENCE_ID,'source_uri':bank.source_uri,'content':block})
    if evidence!=expected:raise ValueError('candidate exemplar or ordinary evidence differs')
    if block:
        index=len(evidence)-1
        if provenance[-1]!={'lock_key':'rails-supply','relpath':f"rails-message/{request['request_id']}/{index}",'sha256':digest(block)}:
            raise ValueError('bank exemplar provenance differs')
        evidence.pop();provenance.pop()
    if allocation is not None:
        # Verify retained source provenance before restoring the control view
        # on this audit-only copy. Dropped bytes are never sent to the model.
        def owned(source,request_id,index):
            if source.get('lock_key')=='rails-supply':
                if source.get('relpath')!=f'rails-message/{request_id}/{index}':
                    raise ValueError('supply provenance has wrong request ownership')
                return {**source,'relpath':'<request-owned>'}
            return source
        if len(lock['evidence_provenance'])!=len(original):raise ValueError('control provenance cardinality differs')
        for j,i in enumerate(indices):
            if owned(provenance[j],request['request_id'],j)!=owned(lock['evidence_provenance'][i],lock['request_id'],i):
                raise ValueError('allocated supply provenance differs')
        restored=[]
        for i,source in enumerate(lock['evidence_provenance']):
            owned(source,lock['request_id'],i)
            restored.append({**source,'relpath':f"rails-message/{request['request_id']}/{i}"}
                            if source.get('lock_key')=='rails-supply' else copy.deepcopy(source))
        result['supplied_evidence']=copy.deepcopy(original)
        result['evidence_provenance']=restored
    result['task_instruction']=lock['task_instruction']
    control['messages'][0]={'role':'system','content':lock['task_instruction']}
    result['history'][-1]['content']=canonical(control)
    result['history'][-1]['content_sha256']=digest(result['history'][-1]['content'])
    return result


def verify_group(row_id,arms,requests,*,bank,perception_phase='perception',allocation=None):
    if [a.get('arm_id') for a in arms] != ['s1','s2','s3','s4','aa','lock']:
        raise ValueError('banked seed audit requires the registered K4 arm order')
    if bank.receipt_sha256!=rail.receipt_sha256(bank.receipt):raise ValueError('verified bank receipt changed')
    if allocation is not None:
        from .tfrl_supply_allocation import validate, project_receipt
        allocation=validate(allocation)
        project_receipt({'arms':arms})
        if any(a['bank_supply_allocation']!=allocation for a in arms):
            raise ValueError('receipt allocation differs from sealed configuration')
    elif any('bank_supply_allocation' in a for a in arms):
        raise ValueError('allocation receipt lacks verified configuration')
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
    if allocation is not None:
        for request in own:
            arm=indexed[request['history'][-1]['contact_id']]
            ids=[e['evidence_id'] for e in request['supplied_evidence'] if e.get('knowledge_role')=='value']
            if ids!=arm['supplied_value_ids']:raise ValueError('arm supplied ids differ from actual request')
    projected=[];augmented=0
    for request in own:
        arm=indexed[request['history'][-1]['contact_id']]
        if arm['arm_id'] in ('s1','s2','s3','s4'):
            request=project_candidate(request,lock,bank,perception_phase,allocation);augmented+=1
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

