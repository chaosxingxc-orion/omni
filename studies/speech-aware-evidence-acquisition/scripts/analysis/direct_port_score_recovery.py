"""Finalize sealed Direct outputs after a failed scorer without replaying inference.

Keep the failed flight receipt and score batch immutable. A separate registered
receipt binds the successful postprocessing to those exact historical bytes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.core.bundle import verify_bundle
from speech_aware_evidence_acquisition.core.flight.receipts import read_flight_receipts
from speech_aware_evidence_acquisition.core.flight.scoring import load_sealed_score_jobs
from speech_aware_evidence_acquisition.core.scoring_batch import ScoringContext, run_score_batch
from speech_aware_evidence_acquisition.contracts import assert_execution_environment

PACKAGE = 'docs/readiness/2026-10-31-direct-score-environment-recovery-v1'
REGISTRATION = PACKAGE + '/registration.json'
RECOVERY_ID = 'SAEA-SLURP-DIRECT-SEALED-SCORE-RECOVERY-NPU-V1'
FACTS = 'docs/readiness/2026-10-30-slurp-direct-only-fold-b-audiofix-npu-v1/registered-facts-v1.json'
REL = 'score-recovery-v1'
RECEIPT = REL + '/recovery-receipt.json'


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def publish(path, doc):
    publish_new(Path(path), (json.dumps(doc, indent=2, sort_keys=True) + '\n').encode())


def clean_head(repo):
    git = ['git', '-c', f'safe.directory={Path(repo).resolve()}']
    if subprocess.check_output(git + ['status', '--porcelain'], cwd=repo).strip():
        raise ValueError('recovery requires a clean source checkout')
    return subprocess.check_output(git + ['rev-parse', 'HEAD'], cwd=repo, text=True).strip()


def require_failed_score(receipts, directory):
    old = receipts['stages']['score']
    output = Path(old['output_dir']).resolve()
    if output != (directory / 'score-batch').resolve():
        raise ValueError('unexpected failed score directory')
    summary = read(output / 'batch-summary.json')
    if summary != old['result'] or (summary['job_count'], summary['succeeded'], summary['failed']) != (1, 0, 1):
        raise ValueError('recovery requires exactly one failed score and no successful score')
    job, = summary['jobs']
    if (old.get('operation') != 'finalize' or job['operation'] != 'finalize'
            or job['status'] != 'failed' or job['returncode'] == 0
            or job.get('result_file') is not None or job.get('complete_receipt_file') is not None):
        raise ValueError('failed batch contains an unexpected operation or result')
    if Path(job['manifest']).resolve() != Path(receipts['stages']['seal']['manifest_path']).resolve():
        raise ValueError('failed scorer targets a different manifest')
    stderr = output / job['stderr_log']
    if stderr.resolve().parent != output or sha(stderr) != job['stderr_sha256']:
        raise ValueError('failed stderr hash differs')
    # The official subprocess stderr is embedded with repr() in the wrapper
    # exception, so its quote characters may be escaped in the persisted log.
    error_text = stderr.read_text().replace("\\'", "'")
    if "ModuleNotFoundError: No module named 'progress'" not in error_text:
        raise ValueError('this recovery is restricted to the missing official environment')
    return old


def inspect_run(source_repo, data, key):
    spec = read(source_repo / FACTS)['identity'][key]
    directory = data / 'runs' / spec['run_id'] / spec['attempt_id']
    manifest = directory / (spec['run_id'] + '.run-manifest.json')
    bundle = verify_bundle(manifest)
    receipts = read_flight_receipts(directory / 'flight-receipts.json')
    require_failed_score(receipts, directory)
    seal = receipts['stages']['seal']
    jobs = load_sealed_score_jobs(seal)
    doc = read(source_repo / 'configs/experiment' / (spec['cell'] + '.json'))
    ids = doc['populations'][0]['ids']
    if (bundle.attempt_state != 'OPENED' or bundle.manifest['status'] != 'COMPLETED'
            or bundle.manifest.get('scores') is not None or len(jobs) != 1
            or jobs[0].job_id != spec['attempt_id'] or jobs[0].manifest != manifest
            or len(bundle.outputs_rows) != 1080 or len(ids) != 1080
            or set(bundle.split_ids) != set(ids) or len(set(ids)) != 1080
            or bundle.manifest['study_commit'] != clean_head(source_repo)):
        raise ValueError('only complete unscored original Direct bundles can be recovered')
    if (directory / REL).exists() or list(directory.glob('*.scores.json')):
        raise ValueError('recovery already started or scores already written; never replay')
    frozen = [p for p in directory.iterdir() if p.is_file() and p != manifest
              and not p.name.startswith('.')]
    frozen.extend(p for p in (directory / 'score-batch').iterdir() if p.is_file())
    return {'key': key, 'run_id': spec['run_id'], 'attempt_id': spec['attempt_id'],
        'experiment': spec['cell'], 'manifest': str(manifest), 'manifest_sha256': sha(manifest),
        'rows':1080, 'source_commit':bundle.manifest['study_commit'],
        'preserved_files': {str(p):sha(p) for p in sorted(frozen)}}


def register(tool_repo, source_repo, data, environment_receipt):
    assert_execution_environment()
    tool_commit = clean_head(tool_repo)
    smoke = read(environment_receipt)
    if smoke['status'] != 'PASS' or smoke['dataset_rows'] != 0:
        raise ValueError('synthetic evaluator verification is required first')
    doc = {'schema':'saea-direct-score-recovery-registration-v1', 'recovery_id':RECOVERY_ID,
        'created_utc':datetime.now(timezone.utc).isoformat(), 'tool_commit':tool_commit,
        'source_repo':str(source_repo), 'source_commit':clean_head(source_repo),
        'authority':'Owner 2026-10-30 D1/D2/D3: complete the registered Direct comparison; routine environment repair.',
        'scope':'Finalize only sealed lock and aa; preserve failed receipts; no inference, FIT, tokenizer or new roster.',
        'max_new_complete_score_jobs':2, 'new_model_calls':0,
        'environment_receipt':str(environment_receipt), 'environment_sha256':sha(environment_receipt),
        'implementation_sha256':sha(Path(__file__)),
        'runs':{k:inspect_run(source_repo,data,k) for k in ('lock','aa')}}
    path = tool_repo / REGISTRATION
    ledger = tool_repo / 'docs/exposure-ledger.md'
    if path.exists() or '| '+RECOVERY_ID+' |' in ledger.read_text():
        raise ValueError('recovery already registered')
    publish(path,doc)
    fields = ['2026-10-31', RECOVERY_ID, 'model-free-check', 'sealed Direct score environment recovery',
        'official evaluator only', 'SLURP TRAIN Fold B; same 1080 rows twice', 'discovery',
        read(source_repo / FACTS)['identity']['lock']['split_identity_hash'], 'no','0','0','0',
        'original generation and failed scoring preserved', REGISTRATION,
        'root countersigned under owner D1/D2/D3; Plan SHA256 '+sha(path)+'; two complete score jobs maximum; no model/tokenizer contact']
    with ledger.open('a') as f:
        f.write('\n## Direct sealed scoring environment recovery\n\n| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n| '+' | '.join(['---']*15)+' |\n| '+' | '.join(fields)+' |\n')
    return {'registration':str(path),'sha256':sha(path),'new_model_calls':0}


def require_registration(path):
    import direct_port_score_evidence as shared
    path = Path(path).resolve()
    tool_repo = path.parents[3]
    doc = read(path)
    if path != tool_repo / REGISTRATION or doc['schema'] != 'saea-direct-score-recovery-registration-v1':
        raise ValueError('invalid recovery registration location or schema')
    rows = shared.registration.parse_ledger(tool_repo,Path('docs/exposure-ledger.md')).get(RECOVERY_ID,[])
    if (len(rows)!=1 or not shared.registration.is_countersigned(rows[0])
            or rows[0]['artifacts']!=REGISTRATION or sha(path) not in rows[0].get('notes','')):
        raise ValueError('recovery is not uniquely registered with its exact hash')
    committed = subprocess.check_output(['git','-c',f'safe.directory={tool_repo.resolve()}','show','HEAD:'+REGISTRATION],cwd=tool_repo)
    if committed != path.read_bytes():
        raise ValueError('recovery registration is not committed')
    return doc


def preserved(entry):
    for name,digest in entry['preserved_files'].items():
        if sha(Path(name)) != digest:
            raise ValueError('preserved recovery source changed: '+name)


def execute(tool_repo, source_repo, data, key):
    assert_execution_environment()
    clean_head(tool_repo)
    path = tool_repo / REGISTRATION
    doc = require_registration(path)
    if sha(Path(__file__)) != doc['implementation_sha256'] or str(source_repo) != doc['source_repo']:
        raise ValueError('registered recovery implementation/source differs')
    if sha(Path(doc['environment_receipt'])) != doc['environment_sha256']:
        raise ValueError('registered evaluator smoke changed')
    smoke = read(doc['environment_receipt'])
    expected_python = smoke['official_command'][0]
    from speech_aware_evidence_acquisition.reproduction.slurp_official_eval import official_evaluator_python
    if official_evaluator_python() != expected_python:
        raise ValueError('isolated evaluator override was not passed')
    entry = inspect_run(source_repo,data,key)
    if entry != doc['runs'][key]:
        raise ValueError('sealed recovery inputs changed since registration')
    directory = Path(entry['manifest']).parent
    receipts = read_flight_receipts(directory/'flight-receipts.json')
    jobs = load_sealed_score_jobs(receipts['stages']['seal'])
    out = directory / REL
    out.mkdir()
    publish(out/'intent.json',{'registration':str(path),'registration_sha256':sha(path),
        'key':key,'created_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),
        'source_commit':clean_head(source_repo),'tool_commit':clean_head(tool_repo),
        'environment_sha256':doc['environment_sha256'],'new_model_calls':0})
    start = time.monotonic()
    summary = run_score_batch(jobs,output_dir=out/'batch',max_workers=1,operation='finalize',
        include_per_sample=True,context=ScoringContext(source_repo,data,source_repo.parents[1]/'docs/datasets.lock.json'))
    if summary['failed'] or summary['succeeded'] != 1:
        raise ValueError('recovery scoring failed; inspect partial effects, never replay')
    preserved(entry)
    duration = time.monotonic()-start
    score = {'job_count':1,'include_per_sample':True,'operation':'finalize',
        'output_dir':str(out/'batch'),'result':summary,'duration_seconds':duration,
        'finalized_manifest_sha256':sha(Path(entry['manifest'])),
        'source_generation_manifest_sha256':entry['manifest_sha256']}
    receipt = {'schema':'saea-direct-score-recovery-receipt-v1','recovery_id':RECOVERY_ID,
        'registration':str(path),'registration_sha256':sha(path),'key':key,
        'intent_sha256':sha(out/'intent.json'),'score':score,'new_model_calls':0,
        'created_utc':datetime.now(timezone.utc).isoformat()}
    publish(out/'recovery-receipt.json',receipt)
    return {'key':key,'status':'FINALIZED','rows':1080,'receipt':str(out/'recovery-receipt.json')}


def resolve(bundle, receipts, directory):
    """Return an independently bound score; never mutate a flight receipt."""
    path = directory / RECEIPT
    if not path.is_file():
        raise ValueError('failed score has no separately registered recovery receipt')
    recovery = read(path)
    if recovery['schema'] != 'saea-direct-score-recovery-receipt-v1' or recovery['recovery_id'] != RECOVERY_ID:
        raise ValueError('invalid recovery receipt')
    registration_path = Path(recovery['registration'])
    if sha(registration_path) != recovery['registration_sha256']:
        raise ValueError('recovery registration hash differs')
    doc = require_registration(registration_path)
    entry = doc['runs'][recovery['key']]
    if (Path(entry['manifest']).resolve()!=bundle.manifest_path.resolve()
            or receipts['run_id']!=entry['run_id'] or receipts['attempt_id']!=entry['attempt_id']):
        raise ValueError('recovery run identity differs')
    require_failed_score(receipts,directory)
    preserved(entry)
    intent_path = directory/REL/'intent.json'
    if sha(intent_path)!=recovery['intent_sha256']:
        raise ValueError('recovery intent changed')
    intent=read(intent_path)
    if (intent['registration_sha256']!=sha(registration_path) or intent['key']!=recovery['key']
            or intent['source_commit']!=doc['source_commit']):
        raise ValueError('recovery intent source differs')
    generation = Path(receipts['stages']['seal']['generation_manifest_path'])
    original = read(generation)
    current = dict(bundle.manifest)
    current['scores'] = None
    if original.get('scores') is None and 'scores' not in original:
        current.pop('scores')
    if sha(generation)!=entry['manifest_sha256'] or current!=original:
        raise ValueError('finalization altered generation fields')
    score = recovery['score']
    if (score['source_generation_manifest_sha256']!=entry['manifest_sha256']
            or Path(score['output_dir']).resolve()!=(directory/REL/'batch').resolve()):
        raise ValueError('recovery score source or directory differs')
    return score,{str(p):sha(p) for p in (path,registration_path,intent_path,generation)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('register','execute'))
    p.add_argument('--tool-repo',type=Path,default=REPO)
    p.add_argument('--source-repo',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--environment-receipt',type=Path)
    p.add_argument('--key',choices=('lock','aa'))
    a=p.parse_args()
    if a.stage=='register':
        result=register(a.tool_repo,a.source_repo,a.data_root,a.environment_receipt)
    else:
        result=execute(a.tool_repo,a.source_repo,a.data_root,a.key)
    print(json.dumps(result))


if __name__=='__main__':
    main()
