"""Prepare exact full Fold A first-phase supply through the shared runtime.

No inference, tokenizer, scoring, readiness promotion or serving deployment.
Retain failed preparation evidence; never overwrite a partial supply.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import fold_a_c2_baseline as baseline
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
read,sha,raw=baseline.read,baseline.sha,baseline.raw


def validate(repo,data,derivation):
    d=read(derivation)
    if (d.get('schema')!='saea-fold-a-c2-baseline-v1' or d.get('rows')!=1080
            or d.get('model_contacts')!=0 or d.get('readiness')!='held'):
        raise ValueError('completed held Fold A c2 derivation required')
    required={f'configs/{k}/{baseline.CELL}.json' for k in ('baseline','experiment')}
    if set(d['output_hashes'])!=required:raise ValueError('exact derived configuration pair required')
    pins=dict(d['output_hashes'])
    for rel,digest in pins.items():
        if sha(repo/rel)!=digest:raise ValueError('derived source changed')
    b=read(repo/f'configs/baseline/{baseline.CELL}.json');e=read(repo/f'configs/experiment/{baseline.CELL}.json');c,=e['cells']
    if (c['cell_id']!=baseline.CELL or c['run_id']!=baseline.RUN or c['baseline_fragment']!=baseline.CELL
            or c['allowed_execution_stages']!=['check'] or c['readiness']['state']!='held'
            or b['policy']!='phase-native-module-v1-c2' or 'prepared_phase_supply' in b):
        raise ValueError('unbound check-only c2 baseline required')
    for rid in (baseline.RUN,baseline.RUN+'-FIT'):
        if any((data/k/rid).exists() for k in ('runs','attempts')):raise ValueError('existing spend; inspect before continuation')
    roster_rel=baseline.comparator.inputs.ROSTER;dataset_rel=baseline.comparator.inputs.DATASET
    roster=read(repo/roster_rel);view=read(repo/dataset_rel);ids=[r['sample_id'] for r in roster['items']]
    if (len(ids)!=1080 or len(set(ids))!=1080 or ids!=e['populations'][0]['ids']
            or set(ids)!={r['sample_id'] for r in view['items']} or len(view['items'])!=1080
            or e['population']['split_identity_hash']!=roster['split_identity_hash']
            or c['dataset_fragment']!=Path(dataset_rel).stem):
        raise ValueError('exact complete Fold A population required')
    for row in view['items']:
        if not row['sample_id'].startswith('slurp:train:') or '/slurp_real/' not in row['media_relpath'] or not row['audio_sha256']:
            raise ValueError('real training audio identity required')
    data_pins={}
    for pin in b['preparation_source_receipts']:
        if pin['root'] not in ('data','repository'):raise ValueError('unknown preparation source')
        root=data if pin['root']=='data' else repo
        p=baseline.comparator.confined(root,pin['relpath'])
        if sha(p)!=pin['sha256']:raise ValueError('preparation receipt changed')
        (data_pins if pin['root']=='data' else pins)[pin['relpath']]=pin['sha256']
    for rel in (roster_rel,dataset_rel,f"configs/model/{c['model_fragment']}.json"):
        pins[rel]=sha(repo/rel)
    baseline.cm.verify_declared_module(b['native_episode']['consumption_module'],b,repo_root=repo)
    return e,ids,pins,data_pins


def verify_output(output,ids):
    if set(output)!={'facets','values'}:raise ValueError('exact shared supply shape required')
    if set(output['values'])!={'perception'}:raise ValueError('complete first-phase values required')
    for phases in output.values():
        if not phases:continue
        if set(phases)!={'perception'}:raise ValueError('first-phase perception only required')
        if set(phases['perception'])!=set(ids):raise ValueError('incomplete real1080 supply')


def prepare(repo,data,derivation,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    assert_execution_environment();head=require_clean_study_repo(repo)
    e,ids,pins,data_pins=validate(repo,data,derivation);c,=e['cells']
    target=data/'prepared'/baseline.RUN/'phase-supply-v1.json'
    if out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex') or target.parent.exists():
        raise ValueError('fresh owned output required; preserve existing preparation')
    out.mkdir(parents=True);dp=sha(derivation)
    publish_new(out/'intent.json',raw({'source_commit':head,'derivation_sha256':dp,'source_hashes':pins,
        'data_hashes':data_pins,'rows':1080,'new_model_calls':0,'automatic_replay':False}))
    try:
        runtime=build_flight_runtime(repo,data,model=c['model_fragment'],dataset=c['dataset_fragment'],
            baseline=c['baseline_fragment'],experiment=baseline.CELL,run_id=baseline.RUN,attempt_id=c['attempt_id'],
            runtime_root=data/'work/scratch/runtime-root',lock_path=repo.parents[1]/'docs/datasets.lock.json')
        runtime.prepare_phase_supply(target,max_workers=16)
        verify_output(read(target)['output'],ids)
        if require_clean_study_repo(repo)!=head or sha(derivation)!=dp or validate(repo,data,derivation)!=(e,ids,pins,data_pins):
            raise ValueError('source changed during preparation')
        result={'schema':'saea-fold-a-c2-supply-v1','source_commit':head,'rows':1080,'cell':baseline.CELL,
            'run_id':baseline.RUN,'derivation_sha256':dp,'source_hashes':pins,'data_hashes':data_pins,
            'supply':{'relpath':str(target.relative_to(data)),'sha256':sha(target)},
            'model_calls':0,'tokenizer_calls':0,'scoring_jobs':0,'readiness':'held','registered':False}
        publish_new(out/'complete.json',raw(result));return result
    except BaseException as error:
        publish_new(out/'failed.json',raw({'error':str(error),'automatic_replay':False}));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','data','derivation','out'):p.add_argument('--'+k,type=Path,required=True)
    print(json.dumps(prepare(**vars(p.parse_args()))))
