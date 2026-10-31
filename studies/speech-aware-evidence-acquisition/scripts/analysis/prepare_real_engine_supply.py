"""Prepare the real 500-row c2 workload offline, without admitting inference."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def run(repo,data,roster,out,lock):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    assert_execution_environment()
    if out.exists() or not out.resolve().is_relative_to((data/'work/codex').resolve()):raise ValueError('fresh owned scratch required')
    view=json.loads((roster/'dataset.json').read_bytes())
    if view['count']!=500:raise ValueError('exactly 500 real samples required')
    out.mkdir(parents=True)
    study=out/'studies/speech-aware-evidence-acquisition';study.mkdir(parents=True)
    (out/'docs').mkdir();shutil.copyfile(lock,out/'docs/datasets.lock.json')
    for name in ('configs','docs','scripts','src'):shutil.copytree(repo/name,study/name)
    source='slurp-native-stage3-l338-lever-column-fold-b-phase-width-v5-npu-v1'
    old=json.loads((repo/'configs/experiment'/f'{source}.json').read_bytes());old_cell,=old['cells']
    candidate=json.loads((data/'work/codex/host-recovery-client-v1/lock-budget-candidates-v2.json').read_bytes())
    baseline=candidate['candidate_baseline'];baseline.pop('prepared_phase_supply')
    baseline['native_contact_fit']={**baseline['native_contact_fit'],'run_id':'SAEA-SLURP-REAL-ENGINE-500-FIT-V1','attempt_id':'attempt-real-engine-500-fit-v1','contact_cap':1000}
    name='slurp-real-engine-500-c2-v1';dataset='slurp-real-engine-500-v1'
    plan={**old['execution_plan'],'planned_model_calls':2000,'planned_speech_audio_seconds':4*view['audio_seconds_total']}
    cell={k:v for k,v in old_cell.items() if k not in ('resume_from','checkpoint_compatibility','readiness')}
    cell.update(cell_id=name,run_id='SAEA-SLURP-REAL-ENGINE-500-C2-V1',attempt_id='attempt-real-engine-500-c2-v1',dataset_fragment=dataset,baseline_fragment=name)
    cell['readiness']={'state':'blocked','reason':'PREPARATION ONLY: serving/FIT and exposure registration pending','requirements':[]}
    exp={k:old[k] for k in ('date','execution_profile','fragment_kind','preserve_wire_responses','stage3_algorithm')}
    exp.update(experiment_id=cell['run_id'],execution_plan=plan,cells=[cell],split_id=view['split_name'],
        purpose='Owner-authorized real 500-row topology measurement; preparation only',
        service_session='saea-real-engine-500-pending-v1',
        population={'count':500,'dataset_fragment':dataset,'split_identity_hash':view['split_identity_hash']},
        populations=[{'dataset_fragment':dataset,'ids':[r['sample_id'] for r in view['items']]}])
    for kind,stem,doc in [('baseline',name,baseline),('dataset',dataset,view),('experiment',name,exp)]:
        (study/'configs'/kind/f'{stem}.json').write_text(json.dumps(doc,indent=2)+'\n')
    runtime=build_flight_runtime(study,data,model=cell['model_fragment'],dataset=dataset,baseline=name,experiment=name,
        run_id=cell['run_id'],attempt_id=cell['attempt_id'],runtime_root=data/'work/scratch/runtime-root',lock_path=lock)
    try:
        runtime.prepare_phase_supply(out/'phase-supply.json',max_workers=16)
        report={'rows':len(runtime._samples),'new_model_calls':0,'registered':False,'state':'prepared','split_identity_hash':view['split_identity_hash']}
        (out/'complete.json').write_text(json.dumps(report));return report
    except Exception as e:
        (out/'failed.json').write_text(json.dumps({'error':str(e),'new_model_calls':0}));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('repo','data','roster','out','lock'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.repo,a.data,a.roster,a.out,a.lock)))
