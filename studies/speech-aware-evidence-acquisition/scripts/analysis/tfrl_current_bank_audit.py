"""Audit actual K4 requests with one verified cross-fit bank, preserving controls.

Only exact declared rule suffix and the Perception-selected exemplar block are
projected out before the existing strict seed/context check. This does not
change iteration1 verification or authorize any generation/bank consumption.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/flights/nx')]
from speech_aware_evidence_acquisition.rails import tfrl_v2_consumption as rail
from speech_aware_evidence_acquisition.rails.tfrl_bank_request_audit import (
    canonical,digest,checked_control,project_candidate,verify_group,
)


def verify(plan,outputs,trace,*,bank,perception_phase='perception',allocation=None):
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
        schema='saea-supply-variant-arms-v7' if allocation is not None else 'saea-supply-variant-arms-v6'
        if (receipt['schema']!=schema or [a['arm_id'] for a in arms]!=list(expected)
                or any(a['decoding_override']!=expected[a['arm_id']] for a in arms)):
            raise ValueError('bank arm group differs from actual registered plan')
        checked=verify_group(row['sample_id'],arms,requests,bank=bank,perception_phase=perception_phase,allocation=allocation)
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
    from speech_aware_evidence_acquisition.rails.tfrl_supply_allocation import from_block
    allocation=from_block(values['native_episode']['consumption_module'])
    if allocation is not None and (plan['banks'].get('supply_allocation')!=allocation
            or plan['banks'].get('supply_bound_items')!=allocation['candidate_max_items']):
        raise ValueError('plan supply allocation differs from configuration')
    return verify(plan,outputs,trace,bank=bank,perception_phase=phase,allocation=allocation)
