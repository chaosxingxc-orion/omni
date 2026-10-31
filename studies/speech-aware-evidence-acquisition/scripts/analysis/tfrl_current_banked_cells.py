"""Derive held banked K4 cells from fixed source configuration and measured bank.

Pure document construction, no calls, writes, registration or admission. The
caller must bind full source results and enforce the existing iteration gate.
"""
import copy,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/flights/nx')]
import tfrl_v2_sampler as sampler
from speech_aware_evidence_acquisition.rails import consumption_module as cm,tfrl_v2_consumption as rail
from speech_aware_evidence_acquisition.rails.tfrl_supply_allocation import KEY,declaration,validate
from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks


def derive(repo,original,experiment,roster,plan,receipt,receipt_pin,source_scope):
    """Keep source roles/stores/prompts; vary seeds and declared bank allocation."""
    iteration=plan['iteration'];half=plan['half'];suffix=half.removeprefix('R01C-').lower()
    if type(iteration) is not int or iteration not in (2,3) or suffix not in ('h1','h2'):
        raise ValueError('banked iteration2/3 of a fixed half required')
    name=f'slurp-tfrl-current-bank-i{iteration}-{suffix}-v1'
    run=f'SAEA-SLURP-TFRL-CURRENT-BANK-I{iteration}-{suffix.upper()}-V1'
    ids=[r['sample_id'] for r in roster['items']];n=313 if suffix=='h1' else 312
    if (len(ids)!=n or len(set(ids))!=n or plan['run_id']!=run or plan['attempt_id']!='generation-v1'
            or [r['row_id'] for r in plan['row_plan']]!=ids or plan['roster']['split_identity_hash']!=roster['split_identity_hash']):
        raise ValueError('exact planned half required')
    sampler.cm.check_bank_measured(receipt)
    sampler.cm.check_cross_fit(receipt,half,consuming_row_ids=ids,consuming_split_identity=roster['split_identity_hash'])
    if receipt['iteration']!=iteration-1 or sampler.cm.content_sha256(receipt)!=plan['banks']['bank_receipt_sha256']:
        raise ValueError('previous opposite-half bank differs from plan')
    if receipt_pin['sha256']!=rail.receipt_sha256(receipt):raise ValueError('bank receipt content pin differs')
    base,exp=copy.deepcopy(original),copy.deepcopy(experiment)
    block=base['native_episode']['consumption_module'];bound=block['decoding_frame']['max_supplied_items']
    allocation=validate(plan['banks']['supply_allocation'])
    if allocation!=declaration(bound) or plan['banks']['supply_bound_items']!=bound-8:
        raise ValueError('actual source bound differs from planned allocation')
    cm.verify_declared_module(block,base,repo_root=repo)
    block['columns']=sampler.execution_columns(plan)
    block['tfrl_v2_bank']=rail.declaration_block(iteration=iteration-1,consumer=half,bank_kind='half',
        receipt_relpath=receipt_pin['relpath'],receipt_sha=receipt_pin['sha256'],source_scope=source_scope)
    block[KEY]=allocation
    compiled=cm.compile_module(block,carrier_module_hooks(repo_root=repo),repo_root=repo)
    values=compiled.values_on_roster_of(base)
    if compiled.program_ceiling!=7 or values['policy']!=base['policy']:
        raise ValueError('seven-contact shared program required')
    expected=copy.deepcopy(base['native_episode']['roles'])
    expected['solver']['decoding_variants']=values['native_episode']['roles']['solver']['decoding_variants']
    if values['native_episode']['roles']!=expected:raise ValueError('banked compilation changed unrelated roles')
    base['native_episode']['roles']=expected
    base.pop('prepared_phase_supply',None)
    base['native_contact_fit'].update(run_id=run+'-FIT',attempt_id='fit-v1',contact_cap=7*n)
    base['what_this_is']='Held cross-fit K4 bank consumption; explicit candidate supply allocation; unchanged aa/lock controls.'
    cm.verify_declared_module(block,base,repo_root=repo)
    cell,=exp['cells'];cell.update(cell_id=name,baseline_fragment=name,run_id=run,attempt_id='generation-v1',
        allowed_execution_stages=['check'],readiness={'state':'held','requirements':[],
        'reason':'Full source gate, actual bank/request proof, own supply, FIT, registration and admission required.'})
    exp.update(experiment_id=run,purpose=base['what_this_is'],population={'count':n,
        'dataset_fragment':cell['dataset_fragment'],'roster_receipt':sampler.cm.HALF_ROSTER_RELPATH[half],
        'split_identity_hash':roster['split_identity_hash']},
        populations=[{'dataset_fragment':cell['dataset_fragment'],'ids':ids}])
    exp['execution_plan'].update(planned_model_calls=14*n,
        planned_speech_audio_seconds=math.ceil(sum(r['audio_seconds'] for r in roster['items'])*14))
    exp['stage3_algorithm']={'id':'tfrl-v2','operators_used':['rollout','scoring','context_application'],
        'note':'Cross-fit measured rule/exemplar context on real K4 construction; unchanged aa/lock.'}
    exp['stage3_measurement'].update(rows=n,read_roster=roster['roster_id'],claims_benefit=False,
        note='Cross-fit banked construction only; independent system acceptance required.')
    return name,base,exp
