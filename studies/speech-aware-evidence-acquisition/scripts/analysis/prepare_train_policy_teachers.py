"""Offline ExternalTeacher preflight using the existing executor, never dispatch."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

p=argparse.ArgumentParser();p.add_argument('--mask-feedback',type=Path,required=True);p.add_argument('--mask-review',type=Path,required=True);p.add_argument('--source-feedback',type=Path,required=True);p.add_argument('--source-review',type=Path,required=True);p.add_argument('--supply-audit',type=Path,required=True);p.add_argument('--terminal-review',type=Path,required=True);p.add_argument('--package',type=Path,required=True);p.add_argument('--teacher-source',type=Path,required=True);p.add_argument('--teacher-config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
sys.dont_write_bytecode=True
sys.path.insert(0,str(args.teacher_source/'src'))
from speech_aware_evidence_acquisition.rails.external_teacher import ExternalTeacher
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import canonical,sha

def read(p):return json.loads(p.read_bytes())
def pin(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(p,obj):
    raw=(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode()
    if p.exists():
        assert p.read_bytes()==raw,'refusing overwrite of different prepared bytes'
        return
    with p.open('xb') as stream:
        os.chmod(p,0o600);stream.write(raw);stream.flush();os.fsync(stream.fileno())

def build_context(original, feedback, independent_review, supply_audit, terminal_review, mask_feedback, mask_review):
    """Correct the unsent dynamic evidence; never mutate the sent package."""
    ctx = copy.deepcopy(original)
    assert ctx['status'] == 'PREPARED_NOT_SENT' and len(ctx['representatives']) <= 16
    assert ctx['legal_only_policy_population']['n'] == 8554
    assert independent_review['status'] == 'MAIN_INDEPENDENT_GOLD_JOIN_VERIFIED'
    assert independent_review['n'] == 1080 and independent_review['all_freeze_hashes_verified']
    assert feedback['metrics'] == independent_review['metrics']
    assert feedback['paired'] == independent_review['pairs_vs_legacy_coverage']
    assert supply_audit['counts'] == dict(all_rows_have_3_distinct_label_cards=1080,
                                       gold_label_recalled=976, valid_index_outputs=1080)
    ctx['schema'] = 'teacher-contract-v2'
    for window in ctx['aggregate_consumption_feedback']['windows']:
        window['gold_covered_in_input'] = window.pop('input_supplied')
        window['all_rows_have_cards'] = window['n']
        window['cards_per_row'] = window['input_k']
        window['card_slot_unit'] = 'distinct label card, never question or source slot'
        window['rank_strata_definition'] = '0 = gold label absent from supplied cards, with nonempty cards and valid output; 1..K = gold label card rank'
    ctx['aggregate_consumption_feedback']['decision'] = 'Historical prompt-only evidence; fixed full8554 parent per window. Next retrieval mechanisms first CPU1080, no NPU launch.'
    ctx['supply_contract'] = dict(
        all_rows_have_cards=1080, gold_covered_in_input=976, gold_absent_in_input=104,
        input_k=3, output_k=1, card_slot_unit='distinct label',
        evidence_scope='A full8554 parent, FoldA1080, 3to1; audit checkpoint ffc029... belongs to A. C3to2 and B5to3 are separately scoped historical window evidence',
        gold_role='post-freeze scoring/error analysis only; never a supply gate or runtime feature',
        runtime_pipeline=['original audio query -> frozen cached audio vector',
                          'positive cosine to legitimate TRAIN question vectors',
                          'optional training-only mask before per-label aggregation',
                          'per-label mean5 and coverage with frozen original8554 class denominator',
                          'TopK distinct label cards with locked descriptions',
                          'P outputs ranking indices over supplied label cards'],
        runtime_query_text_required=False,
        text_query_role='bounded error-analysis evidence only, never a runtime transcript or retrieval input',
        training_text_role='legal original TRAIN question text and locked descriptions may define static features',
        audit=supply_audit)
    ctx['policy_scope'] = 'Eligible legal TRAIN8554 only for bank selection and feature/threshold fitting; excluded source-isolation rows are not bank candidates. Existing adapted FoldA1080 development aggregates may inform strategy feedback, but are not independent confirmation. TEST is never used for tuning or feedback. Source-isolation exclusions are not evidence of TEST pollution.'
    ctx['parent_contract'] = dict(
        legal_training_bank=8554, original_training_census=11514, full_census_descriptive_only=True,
        registered_labels=60, nonempty_question_labels=59,
        parent='full8554 positive-cosine coverage; fixed original exact-text per-class count',
        historical_7513='already completed selected-bank control, NOT the parent; reuse prior evidence, never rerun C3',
        frozen_denominator=True, evaluation_n=1080, dataset='original adapted FoldA1080',
        evaluation_is_blind=False, test_for_tuning=False,
        first_evaluation='CPU full1080 cached audio similarities: R1/R3/R5, MRR, macro60, paired repairs/regressions, retention and cost; freeze rankings before gold join',
        p_window_changed=False)
    ctx['source_distinct_counterevidence'] = dict(
        status=independent_review['status'], n=1080,
        metrics=independent_review['metrics'],
        paired_vs_full8554_coverage=independent_review['pairs_vs_legacy_coverage'],
        conclusions=feedback['teacher_next_round'],
        limitations=independent_review['limitations'],
        mechanism='per audio query and label: source-group max then top5 positive cosine; frozen original count and new source count are separate arms',
        diagnostic_difference='static representative/signed text LOO is descriptive; runtime group maximum depends on audio similarity, never transcript')
    ctx['selected_implementation'] = dict(
        algorithm='TFGROUP', candidate='C2 main adaptation',
        implementation_status='implementation_in_progress', reward=None, advantage=None,
        owner='A Terra PID922, independent operator worktree',
        operator_location='training-only mask before label aggregation',
        description_df=[8,10,12], source_distinct_same_label_peer_count=[2,3],
        small_class_rule='original legal class size <=32: keep entire class, both gates bypassed',
        empty_class_rule='if any originally nonempty class is filtered empty, restore its entire original legal class',
        coverage_denominator='original8554 class counts fixed, never filtered counts',
        first_evaluation='CPU all1080 cached audio, no encoder/API/NPU',
        teacher_instruction='Do not request or propose this same C2 adaptation again; reference as pending, do not invent reward. Do not implement operator code in this task.')
    assert mask_review['status'] == 'MAIN_FROZEN_RANKING_GOLD_JOIN_VERIFIED'
    assert mask_review['n'] == 1080 and mask_review['routes'] == 12
    assert len(mask_feedback['results']) == 12
    for arm, measured in mask_feedback['results'].items():
        assert measured['metrics'] == mask_review['metrics'][arm]
        if arm != 'parent':
            assert measured['metrics']['R3'] < mask_feedback['results']['parent']['metrics']['R3']
    ctx['selected_implementation'].update(
        implementation_status='completed_cpu_rejected',
        reward=dict(type=mask_feedback['reward_type'],n=1080,results=mask_feedback['results']),
        advantage=None, downstream_P_reward=None, npu_authorized=False,
        teacher_instruction='CPU11 arms completed and independently reviewed, all R3 below parent976. Do not repeat same C2 or tiny threshold scans. Prefer support-preserving/evidence-calibration mechanisms; lexical-overlap failures do not refute all semantic-quality methods.')
    ctx['mask_cpu_feedback'] = mask_feedback
    ctx['mask_cpu_feedback_review'] = mask_review
    ctx['statistical_contract'] = dict(
        significance='p>=.05 is not falsification, equivalence, or proof of no effect',
        advantage='compute only from genuine same-query same-policy same-budget reward groups; never zero advantage because of p',
        unrun_reward=None, unrun_advantage=None,
        equal_reward_group='actual equal rewards may yield zero relative advantage; this is not a significance rule',
        historical_windows='different input/output windows are not same-policy reward groups')
    ctx['previous_teachers'] = dict(
        total_tokens=terminal_review['tokens']['total_tokens'],
        usage=terminal_review['tokens'], currency_cost=terminal_review['currency_cost'],
        semantic_dispositions=copy.deepcopy(terminal_review['semantic_candidates']),
        no_retransmission=True)
    for disposition in ctx['previous_teachers']['semantic_dispositions']:
        if disposition['algorithm'] == 'tfgroup' and disposition['id'] == 'C2':
            disposition['status'] = 'completed_cpu_rejected'
            disposition['reason'] = 'CPU11 arms completed, independently verified and rejected: all R3 below parent976; actual retrieval reward attached, downstream P reward unmeasured. Do not propose duplicate work.'
    assert ctx['previous_teachers']['total_tokens'] == 104869
    assert 'input_supplied' not in json.dumps(ctx)
    return ctx


out=args.out.resolve();out.mkdir(mode=0o700,parents=True,exist_ok=True);os.chmod(out,0o700)
ctx=build_context(read(args.package/'teacher-context.json'),read(args.source_feedback),
                  read(args.source_review),read(args.supply_audit),read(args.terminal_review),
                  read(args.mask_feedback),read(args.mask_review))
ctx['evidence_sourcepins']=[pin(path) for path in [args.package/'teacher-context.json',
    args.source_feedback,args.source_review,args.supply_audit,args.terminal_review,args.mask_feedback,args.mask_review]]
save(out/'teacher-context.json',ctx)
results={}
for alg,template in [('ace','ace-retrieval-template.md'),('tfgroup','tfgroup-retrieval-template.md')]:
    d=out/alg;d.mkdir(mode=0o700,exist_ok=True);os.chmod(d,0o700)
    assert not list(d.glob('contacts/*/response/intent.json')),'sent requests are immutable'
    for src,name in [(Path(__file__).resolve().parents[2]/'configs'/'teacher-contract-v2'/template,'template.md'),(args.teacher_source/'scripts/environment/configured_external_learner.py','executor.py')]:
        dest=d/name
        if dest.exists():assert dest.read_bytes()==src.read_bytes()
        else:shutil.copyfile(src,dest);dest.chmod(0o600)
    config=read(args.teacher_config);config['research_prompt']=pin(d/'template.md')
    if 'analysis_budget' in config:
        config['analysis_budget'].update(max_hypotheses=4,max_mechanism_dimensions=4,representative_case_target=16)
    caller=ExternalTeacher(d/'contacts',config=config,executor=pin(d/'executor.py'))
    request=dict(schema='saea-tfgrpo-learning-request-v1',model_id=caller.model_id,
        learning_config_sha256=caller.learning_config_sha256,stage=alg+'-train-question-operator-feedback-v2',
        identity=sha(dict(algorithm=alg,context=ctx)),messages=[dict(role='user',content=canonical(ctx).decode())])
    binding=caller.preflight(request)
    save(d/'teacher.json',config);save(d/'request.json',request)
    contact_configs=list((d/'contacts').glob('*/config.json'));assert len(contact_configs)==1
    results[alg]=dict(status='PREFLIGHT_ONLY_NOT_SENT',request_identity=request['identity'],request=pin(d/'request.json'),
        config=pin(contact_configs[0]),executor=pin(d/'executor.py'),template=pin(d/'template.md'),
        binding=binding,max_tokens=config['max_tokens'],model=caller.model_id,
        input_bytes=sum(len(x['content'].encode()) for x in request['messages'])+len(caller.policy_text.encode()))
save(out/'result.json',dict(status='OFFLINE_PREFLIGHT_ONLY',new_api_calls=0,context=pin(out/'teacher-context.json'),
    dynamic_context=True,fixed_algorithm_templates=True,algorithms=results,
    dispatch_executed=False,full_train_table_attached=False,sourcepins=ctx['evidence_sourcepins'],
    preparation_source=pin(Path(__file__).resolve()),previous_teacher_tokens=104869,selected_implementation=ctx['selected_implementation']))
print(json.dumps(dict(status='OFFLINE_PREFLIGHT_ONLY',new_api_calls=0,algorithms={k:dict(request_identity=v['request_identity'],input_bytes=v['input_bytes']) for k,v in results.items()})))
