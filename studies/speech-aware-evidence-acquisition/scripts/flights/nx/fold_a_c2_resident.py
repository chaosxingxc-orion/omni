"""Bind the complete Fold A c2 control to the measured S128 runtime configuration.

Preserve scientific decoding, all population IDs and the two-contact algorithm.
This stages files only; actual serving identity, repin and admission remain due.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(REPO/'src'),str(Path(__file__).parent)]
import fold_a_c2_baseline as baseline
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
read,sha,raw=baseline.read,baseline.sha,baseline.raw
MODEL='bf16-thinking-real500-tp4-s128-v1'
SESSION='serving-a0-remaining353-selected-resident-v1'
RECEIPT='receipts/runtime-a0-remaining353-selected-resident-v1.json'


def documents(exp, old_model, new_model):
    old,new=copy.deepcopy(old_model),copy.deepcopy(new_model)
    old.pop('runtime_receipt');new.pop('runtime_receipt')
    if old!=new or new_model['runtime_receipt']!=RECEIPT:
        raise ValueError('runtime-only change to scientific decoding required')
    result=copy.deepcopy(exp);cell,=result['cells']
    if (cell['cell_id']!=baseline.CELL or cell['run_id']!=baseline.RUN
            or cell['readiness']['state']!='held' or cell['allowed_execution_stages']!=['check']
            or result['stage3_algorithm']['id']!='none' or result['stage3_algorithm']['operators_used']!=[]
            or result['population']['count']!=1080):
        raise ValueError('held full Fold A A0 control required')
    cell['model_fragment']=MODEL
    cell['allowed_execution_stages']=['check','prepare-supply','worker','campaign']
    result['service_session']=SESSION
    return result


def stage(repo,data,source,binding,closure,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    assert_execution_environment();head=require_clean_study_repo(repo)
    baseline.comparator.confined(data,str(source));baseline.comparator.confined(data,str(out))
    if out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex'):
        raise ValueError('fresh owned stage output required')
    completed=read(binding);closed=read(closure)
    if (completed['schema']!='saea-fold-a-c2-supply-binding-v1' or completed['rows']!=1080
            or completed['new_model_calls']!=0 or completed['registered'] or completed['readiness']!='held'
            or not completed['checks'] or not all(c['identity_ok'] and c['seed_ok'] for c in completed['checks'])
            or closed['schema']!='saea-fold-a-c2-preparation-closure-v1' or closed['rows']!=1080
            or closed['generation_admitted'] or closed['new_model_contacts']!=0):
        raise ValueError('completed held supply and preparation closure required')
    ledger=(repo/'docs/exposure-ledger.md').read_text()
    if closed['closed_agreement_line'] not in ledger:
        raise ValueError('completed agreement ledger closure not installed')
    for row in [closed['encoder_ledger_row'],*closed['derivative_ledger_rows']]:
        if ledger.splitlines().count('| '+' | '.join(row)+' |')!=1:
            raise ValueError('completed preparation ledger closure not installed')
    supply=completed['supply']
    if sha(data/supply['relpath'])!=supply['sha256']:
        raise ValueError('bound supply changed')
    for rel,digest in completed['output_hashes'].items():
        if sha(repo/rel)!=digest:raise ValueError('bound configuration changed')
    for rel,digest in closed['pins'].items():
        if sha(data/rel)!=digest:raise ValueError('closed preparation changed')
    for run in (baseline.RUN,baseline.RUN+'-FIT'):
        if any((data/k/run).exists() for k in ('runs','attempts')):raise ValueError('existing spend; inspect before staging')
    exp_path=repo/f'configs/experiment/{baseline.CELL}.json';exp=read(exp_path);cell,=exp['cells']
    model_rel=f'configs/model/{MODEL}.json';receipt_rel='docs/'+RECEIPT
    model=read(source/model_rel)
    updated=documents(exp,read(repo/f"configs/model/{cell['model_fragment']}.json"),model)
    pins={rel:sha(source/rel) for rel in (model_rel,receipt_rel)}
    blobs={rel:(source/rel).read_bytes() for rel in pins}
    out.mkdir(parents=True)
    publish_new(out/'intent.json',raw({'source_commit':head,'source_runtime_hashes':pins,
        'binding_sha256':sha(binding),'closure_sha256':sha(closure),'new_model_calls':0}))
    for rel in (model_rel,receipt_rel,str(exp_path.relative_to(repo))):
        p=repo/rel
        if p.exists():
            backup=out/'before'/rel;backup.parent.mkdir(parents=True,exist_ok=True);publish_new(backup,p.read_bytes())
    for rel,blob in blobs.items():
        p=repo/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(blob)
    exp_path.write_bytes(raw(updated))
    result={'schema':'saea-fold-a-c2-resident-stage-v1','rows':1080,'source_commit':head,
        'binding_sha256':sha(binding),'closure_sha256':sha(closure),'source_runtime_hashes':pins,
        'output_hashes':{rel:sha(repo/rel) for rel in (*pins,str(exp_path.relative_to(repo)))},
        'new_model_calls':0,'readiness':'held','registered':False,'live_resident_verified':False,
        'remaining':['registration','shared supply repin','live serving verification','admission','preflight']}
    publish_new(out/'complete.json',raw(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','data','source','binding','closure','out'):p.add_argument('--'+k,type=Path,required=True)
    print(json.dumps(stage(**vars(p.parse_args()))))
