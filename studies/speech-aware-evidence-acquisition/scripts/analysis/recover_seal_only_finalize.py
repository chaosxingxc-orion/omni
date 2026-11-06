"""Resume a failed synchronous seal-only finalizer from pinned original evidence."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from speech_aware_evidence_acquisition.contracts import assert_execution_environment
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.core.bundle import verify_bundle
from speech_aware_evidence_acquisition.core.bundle.delivery import _recorded_delivery_config_hash
from speech_aware_evidence_acquisition.core.experiment_assets import resolve_asset
from speech_aware_evidence_acquisition.core.flight.receipts import read_flight_receipts
from speech_aware_evidence_acquisition.core.scoring_batch import ScoreJob, ScoringContext, run_score_batch
from speech_aware_evidence_acquisition.core.scoring_source import verify_separate_scoring_tool


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(pin):
    if set(pin) != {'path', 'sha256'} or not Path(pin['path']).is_absolute():
        raise ValueError('exact absolute input pin required')
    p = Path(pin['path'])
    if p.is_symlink() or not p.is_file() or sha(p) != pin['sha256']:
        raise ValueError('input pin differs: ' + str(p))
    return p


def validate_evidence(cfg, bundle, receipts, generation, failed, identity):
    m = bundle.manifest
    run, attempt, rows = cfg['run_id'], cfg['attempt_id'], cfg['rows']
    if (bundle.attempt_state != 'OPENED' or m.get('status') != 'COMPLETED'
            or m.get('scores') is not None or m.get('run_id') != run
            or m.get('attempt', {}).get('attempt_id') != attempt
            or len(bundle.outputs_rows) != rows or len(bundle.split_ids) != rows
            or m.get('study_commit') != identity['generation_commit']
            or m.get('cost', {}).get('calls_used') != 0):
        raise ValueError('complete opened unscored source identity differs')
    if set(receipts['stages']) != {'preflight', 'preprocess', 'serve', 'run_phases'}:
        raise ValueError('expected failed synchronous seal stage')
    pre = receipts['stages']['preflight']; phase = receipts['stages']['run_phases']
    resume = pre['resume']; predecessor = m.get('predecessor_attempt', {})
    if (pre.get('config_hash') != m['config']['config_hash'] or pre.get('protocol_hash') != m['protocol_hash']
            or pre.get('sample_count') != rows or not resume.get('seal_only')
            or resume.get('counts') != {'roster':rows, 'kept':rows, 'flown':0, 'partial':0, 'unstarted':0}
            or resume.get('worst_case_contacts') != 0 or phase.get('seal_only') is not True
            or phase.get('resume', {}).get('kept') != rows or phase['resume'].get('flown') != 0
            or predecessor.get('counts') != {'kept':rows,'flown':0,'partial':0}
            or predecessor.get('checkpoint_sha256') != cfg['checkpoint_sha256']):
        raise ValueError('sealed predecessor coverage or identity differs')
    if (generation.get('schema') != 'saea-seal-only-generation-manifest-v1'
            or generation.get('run_id') != run or generation.get('attempt_id') != attempt
            or generation.get('seal_only') is not True
            or generation.get('cost_provenance', {}).get('predecessor_attempt_ids') != [predecessor['attempt_id']]
            or generation.get('cost', {}).get('calls_used') != resume.get('predecessor_contacts')):
        raise ValueError('predecessor cost provenance differs')
    if ('delivery config differs from recorded run' not in failed
            or 'finalize_bundle' not in failed or 'score_bundle' not in failed):
        raise ValueError('expected synchronous finalizer failure evidence')


def recover(config_path, check_only=False):
    assert_execution_environment()
    cfg = json.loads(Path(config_path).read_bytes())
    required = {'schema','run_id','attempt_id','rows','checkpoint_sha256','producer_id','inputs','output','lock'}
    if set(cfg) != required or cfg['schema'] != 'saea-seal-only-finalize-recovery-v1':
        raise ValueError('exact recovery config required')
    rows = cfg['rows']
    lock_path = checked(cfg['lock'])
    source = Path('/opt/saea-release/studies/speech-aware-evidence-acquisition')
    identity = verify_separate_scoring_tool(source, REPO)
    inputs = cfg['inputs']
    if set(inputs) != {'manifest','generation','receipts','failed_stderr','original_launch','producer_container','runtime','admission','official_python'}:
        raise ValueError('exact recovery input set required')
    paths = {k:checked(v) for k,v in inputs.items()}
    launch = json.loads(paths['original_launch'].read_bytes()); container = json.loads(paths['producer_container'].read_bytes())
    runtime = inputs['runtime']; args = launch['argv']; env = launch['environment']
    if (len(container) != 1 or container[0]['Id'] != cfg['producer_id']
            or container[0]['Config']['Labels'].get('saea.launch.config') != hashlib.sha256(json.dumps(launch,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            or args.count('--runtime-config') != 1 or args[args.index('--runtime-config')+1] != runtime['path']
            or env.get('SAEA_RUNTIME_CONFIG_SHA256') != runtime['sha256']
            or os.environ.get('SAEA_RUNTIME_CONFIG') != runtime['path']
            or os.environ.get('SAEA_RUNTIME_CONFIG_SHA256') != runtime['sha256']
            or os.environ.get('SPEECHRL_DATA_DIR') != env.get('SPEECHRL_DATA_DIR')):
        raise ValueError('original producer runtime identity differs')
    target = Path(runtime['path']); mounts = [m for m in launch['mounts'] if target.is_relative_to(m['target'])]
    if not mounts: raise ValueError('runtime producer mount absent')
    mount = max(mounts,key=lambda m:len(m['target']))
    actual = [m for m in container[0]['Mounts'] if m['Destination'] == mount['target']]
    actual_env = dict(item.split('=', 1) for item in container[0]['Config']['Env'])
    host_runtime = Path(mount['source']) / target.relative_to(mount['target'])
    if (mount['readonly'] is not True or len(actual) != 1 or actual[0]['Source'] != mount['source']
            or actual[0]['RW'] is not False or actual[0]['Type'] != 'bind'
            or actual_env.get('SAEA_RUNTIME_CONFIG_SHA256') != runtime['sha256']
            or actual_env.get('SPEECHRL_DATA_DIR') != env['SPEECHRL_DATA_DIR']
            or not host_runtime.is_file() or sha(host_runtime) != runtime['sha256']):
        raise ValueError('runtime producer mount differs')
    if paths['manifest'].parent != paths['generation'].parent or paths['receipts'].parent != paths['manifest'].parent:
        raise ValueError('source files do not share original run directory')
    bundle = verify_bundle(paths['manifest'])
    receipts = read_flight_receipts(paths['receipts'])
    generation = json.loads(paths['generation'].read_bytes())
    validate_evidence(cfg, bundle, receipts, generation, paths['failed_stderr'].read_text(), identity)
    if _recorded_delivery_config_hash(source, bundle.manifest['config']['fragments']) != bundle.manifest['config']['config_hash']:
        raise ValueError('recorded runtime overlay composition differs')
    pre = receipts['stages']['preflight']; names = bundle.manifest['config']['fragments']; plan = bundle.manifest['plan']
    baseline = json.loads(resolve_asset(source, 'configs/baseline/' + names['baseline'] + '.json').read_bytes())
    model = json.loads(resolve_asset(source, 'configs/model/' + names['model'] + '.json').read_bytes())
    pending = 'pending post-run analysis'
    cells = {'experiment_id':str(baseline.get('experiment_id',cfg['run_id'])),
             'speech task/carrier':f"{plan['carrier_lock_key']}; split {pre['split_id']}; {cfg['rows']} samples",
             'changed axes':'model={model}; dataset={dataset}; baseline={baseline}; experiment={experiment}'.format(**names),
             'shared code revision':'none; this study declares no speechrl_common dependency',
             'model rev':str(model.get('model_lock_key','')), 'dataset rev':str(plan['split_identity_hash']),
             'MLflow run':'not used', **{k:pending for k in ('effectiveness','reasonableness','efficiency','deviations','decision')}}
    out=Path(cfg['output'])
    if out.parent != paths['manifest'].parent or out.exists(): raise ValueError('fresh adjacent recovery output required')
    if check_only:
        return {'validated': True, 'rows': rows, 'source_identities': identity, 'new_model_calls': 0}
    lock = out.parent / '.seal-only-finalize-recovery.guard'
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, 'a+b') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if out.exists(): raise ValueError('recovery already started')
        out.mkdir(mode=0o700)
        def publish(name,value): publish_new(out/name,(json.dumps(value,sort_keys=True,indent=2)+'\n').encode())
        publish('intent.json', {'schema':'saea-seal-only-finalize-recovery-intent-v1','config_sha256':sha(config_path),
            'source_identities':identity,'inputs':inputs,'generation_cost':generation['cost'],'rows':rows,'new_model_calls':0})
        publish('ledger-cells.json',cells)
        job=ScoreJob(attempt, paths['manifest'], ledger_cells=out/'ledger-cells.json')
        result=run_score_batch([job],output_dir=out/'batch',max_workers=1,operation='finalize',include_per_sample=True,
            context=ScoringContext(source,Path(env['SPEECHRL_DATA_DIR']),lock_path,tool_repo_root=REPO))
        if result.get('failed') or result.get('succeeded') != 1: raise ValueError('finalize failed; preserve intent and batch')
        after=verify_bundle(paths['manifest'])
        if after.attempt_state != 'FINALIZED' or len(after.outputs_rows) != rows: raise ValueError('finalized bundle incomplete')
        for key in ('generation','receipts','failed_stderr','original_launch','producer_container','runtime','admission','official_python'):
            checked(inputs[key])
        publish('recovery-receipt.json',{'schema':'saea-seal-only-finalize-recovery-v1','intent_sha256':sha(out/'intent.json'),
            'finalized_manifest_sha256':sha(paths['manifest']),'result':result,'new_model_calls':0})
        return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    print(json.dumps(recover(Path(args.config),check_only=args.check_only),sort_keys=True))
