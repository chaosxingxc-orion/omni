"""Official SLURP count reuse for conditional gold-candidate slot evaluation.

No intent accuracy is reported: gold intent is an offline oracle projection.
All 545 rows contribute. Invalid/missing predict [], never fabricated FP.
Missing rows make corpus counts partial-with-empty-lower-bound, not a final run.
"""
import importlib
from importlib.metadata import version
from pathlib import Path
import sys
from common import ROOT, read, require


def aggregate(cfg, rows):
    # Rolling subset denominator; unchanged official distance/count implementation.
    denominator=len(rows)
    sys.path.insert(0,str(ROOT/cfg['rails_src']))
    module=importlib.import_module('speech_aware_evidence_acquisition.reproduction.slurp_scoring')
    import jiwer
    # Exercise the actual dependency; no alternate distance implementation.
    require(module._word_distance('five thirty','five') >= 0,'jiwer unavailable')
    fields=[f'{family}_{axis}' for family in ('slot_f1_exact','slu_f1_word','slu_f1_char') for axis in ('tp','fp','fn')]
    totals={k:0.0 for k in fields}; observed_totals={k:0.0 for k in fields}; by_slot={}; contributions=[]
    def compute(gold,pred,intent):
        scenario,action=intent.split('_',1)
        score=module.score_slu(gold_scenario=scenario,gold_action=action,
            gold_entities=[module.SlotEntity(**e) for e in gold],predicted_scenario=scenario,
            predicted_action=action,predicted_entities=[module.SlotEntity(**e) for e in pred])
        return {k:getattr(score,k) for k in fields}
    for row in rows:
        gold=row['reference_entities_CPU_only'];parsed=row['rails_parsed']
        pred=parsed[row['gold_intent_CPU_only']]['entities'] if row['outcome']=='legal' else []
        value=compute(gold,pred,row['gold_intent_CPU_only'])
        for k in fields:
            totals[k]+=value[k]
            if row['outcome']!='missing':observed_totals[k]+=value[k]
        contributions.append(dict(sample_id=row['sample_id'],status=row['outcome'],counts=value))
        for slot in sorted({e['type'] for e in gold+pred}):
            v=compute([e for e in gold if e['type']==slot],[e for e in pred if e['type']==slot],row['gold_intent_CPU_only'])
            acc=by_slot.setdefault(slot,{k:0.0 for k in fields})
            for k in fields:acc[k]+=v[k]
    def finish(v):
        result=dict(counts=v)
        for family in ('slot_f1_exact','slu_f1_word','slu_f1_char'):
            p,r,f=module._prf(*(v[f'{family}_{a}'] for a in ('tp','fp','fn')))
            result[family]=dict(precision=p,recall=r,F1=f)
        combined={a:v['slu_f1_word_'+a]+v['slu_f1_char_'+a] for a in ('tp','fp','fn')}
        p,r,f=module._prf(combined['tp'],combined['fp'],combined['fn'])
        result['SLU']=dict(counts=combined,precision=p,recall=r,F1=f)
        return result
    observed=sum(row['outcome']!='missing' for row in rows)
    final=finish(totals)
    return dict(status='NO_ARCHIVED_PREDICTIONS' if not observed else ('COMPLETE_COUNTS' if observed==denominator else 'PARTIAL_MISSING_AS_EMPTY_LOWER_BOUND'),
        metric_scope='conditional_slot_only_offline_oracle_gold_intent_projection_NOT_E2E',
        upstream_commit='8eb16545762be97ace75334109d73824217311f1',
        implementation=cfg['rails_src']+'/speech_aware_evidence_acquisition/reproduction/slurp_scoring.py',
        scorer_version=module.SLURP_SCORER_VERSION,jiwer_version=version('jiwer'),
        denominator=denominator,observed=observed,missing=denominator-observed,
        invalid=sum(row['outcome']=='invalid' for row in rows),
        gold_filler_rule='original tokens spans lowercase; prediction untouched; no normalization or repairs',
        invalid_missing_rule='empty entity sequence; all gold FN; no invented FP; legality and exactness separately fail',
        micro_full_manifest_missing_as_empty_lower_bound=final if observed else None,
        observed_subset_micro=finish(observed_totals) if observed else None,
        observed_subset_note='arrival-order partial 70/545 is not random or final; invalid kept; do not infer full-run quality',
        official_conditional_SLU_F1=final['SLU']['F1'] if observed==denominator else None,
        partial_lower_bound_SLU_F1=final['SLU']['F1'] if 0<observed<denominator else None,
        per_type={k:finish(v) for k,v in sorted(by_slot.items())} if observed else None,
        counts_only_even_without_predictions=totals,
        macro_SLU={k:sum(finish(v)['SLU'][k] for v in by_slot.values())/len(by_slot) for k in ('precision','recall','F1')} if observed and by_slot else None,
        intent_accuracy=None,intent_reason='oracle gold branch used only offline; S does not select final intent',
        TEST_read=False),contributions
