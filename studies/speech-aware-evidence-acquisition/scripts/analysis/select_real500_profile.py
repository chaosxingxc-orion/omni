"""Select the fastest complete real500 candidate, retaining full quality evidence.

TP8-384 remains conditional: if TP8-256 wins, refuse selection until its higher
capacity comparison is explicitly handled. No partial throughput extrapolation.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis')]
from register_real_engine_candidate import read,sha,write
from bind_a0_selected_profile import verify_selection

REQUIRED={f'SAEA-SLURP-REAL500-{suffix}-V1' for suffix in ('TP4-S96','TP4-S128','TP8-S256')}


def choose(results):
    if {r['run_id'] for r in results}!=REQUIRED or len(results)!=3:
        raise ValueError('exact complete96/128/256 comparison required')
    for r in results:
        if (r['schema']!='saea-real500-verified-result-v1' or r['rows']!=500
                or not r['official_finalized'] or not r['quality_identity_passed']
                or not r['checkpoint_cost_reconciled'] or not r['generation_sealed']):
            raise ValueError('verified complete candidate required')
        import math
        duration=r['generation_start_to_seal_seconds']
        if isinstance(duration,bool) or not math.isfinite(duration) or duration<=0:
            raise ValueError('finite positive generation duration required')
        for metric in ('slu_f1','intent_accuracy','span_f1','scenario_accuracy'):
            value=r['quality'][metric]
            if isinstance(value,bool) or not math.isfinite(value) or not 0<=value<=1:
                raise ValueError('complete finite quality metrics required')
    if len({r['split_identity_hash'] for r in results})!=1:
        raise ValueError('paired populations required')
    winner=min(results,key=lambda r:r['generation_start_to_seal_seconds'])
    if winner['run_id'].endswith('TP8-S256-V1'):
        raise ValueError('TP8-256 wins; evaluate conditional384 before final selection')
    return {'schema':'saea-real500-profile-selection-v1','complete_roster_comparison':True,
        'selected_run_id':winner['run_id'],'profile':winner['profile'],
        'reason':'Choose the lowest observed whole-generation wall time on the identical complete500 train roster. All official scores are retained below; this single comparison does not establish statistical quality equivalence. TP8-256 is slower, so conditional TP8-384 is not expanded; return to system-accuracy work.',
        'candidates':[r['selection_evidence'] for r in results],
        'full_results':[{'run_id':r['run_id'],'profile':r['profile'],
            'generation_seconds':r['generation_start_to_seal_seconds'],'quality':r['quality'],
            'calls':r['cost']['calls_used'],'prompt_tokens':r['cost']['prompt_tokens_total'],
            'completion_tokens':r['cost']['completion_tokens_total']} for r in results],
        'conditional384':'not-expanded: TP8-256 has no complete-roster throughput benefit',
        'new_model_calls':0,'claims_system_gain_locked':False}


def select(data,result_paths,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    if not out.resolve().is_relative_to((data/'work/codex').resolve()) or out.exists():
        raise ValueError('fresh owned selection required')
    decision=choose([read(p) for p in result_paths])
    # Reverify actual pinned finalized bundles, scores, roster and timings.
    verify_selection(data,decision)
    decision['input_results']=[{'relpath':str(p.resolve().relative_to(data.resolve())),'sha256':sha(p)} for p in result_paths]
    write(out,decision);return decision


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--result-path',type=Path,action='append',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(select(a.data,a.result_path,a.out)))
