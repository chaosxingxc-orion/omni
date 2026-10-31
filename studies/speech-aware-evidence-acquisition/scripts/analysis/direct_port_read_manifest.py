"""Bind Direct smoke comparison inputs to sealed runs and actual serving receipts."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
for directory in (REPO/'src', REPO/'scripts/analysis', REPO/'scripts/flights/direct', REPO/'scripts/checks'):
    sys.path.insert(0, str(directory))
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.core.flight.receipts import read_flight_receipts
import flight_registration_check as registration
import direct_port_family as fam
import slurp_direct_fold_b_core_read as base


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError('read evidence escapes data root')
    return {'relpath':str(path.relative_to(root)), 'sha256':digest(path)}


def require_read(repo, read_id):
    rows = registration.parse_ledger(repo, Path('docs/exposure-ledger.md'))
    if not any(registration.is_countersigned(row) for row in rows.get(read_id, [])):
        raise ValueError('READ identity is not registered and countersigned')


def run_evidence(data, run, attempt, repo=REPO):
    paths = base.run_paths(data, run, attempt)
    manifest = json.loads(paths['manifest'].read_bytes())
    receipts_path = paths['dir']/'flight-receipts.json'
    receipts = read_flight_receipts(receipts_path)
    if (manifest['status'] != 'COMPLETED' or manifest['run_id'] != run or
            manifest['attempt']['attempt_id'] != attempt or
            (receipts['run_id'], receipts['attempt_id']) != (run, attempt) or
            'seal' not in receipts['stages']):
        raise ValueError('comparison requires the exact completed and sealed run')
    preflight = receipts['stages']['preflight']
    if (manifest['study_commit'] != preflight['study_commit'] or
            manifest['config']['config_hash'] != preflight['config_hash'] or
            manifest['protocol_hash'] != preflight['protocol_hash']):
        raise ValueError('run manifest and execution receipts disagree')
    for artifact, name in (('outputs','outputs'), ('raw_trace','trace')):
        if manifest[artifact]['sha256'] != digest(paths[name]):
            raise ValueError('sealed output/trace hash differs')
    ids = manifest['split']['ids']
    if len(ids) != 4 or len(set(ids)) != 4 or manifest['split']['count'] != 4:
        raise ValueError('comparison requires four unique sealed roster members')
    serve = receipts['stages']['serve']
    if serve['session_source'] != 'attached-multi-instance-serving-session':
        raise ValueError('comparison requires the registered resident deployment')
    deployed = serve['deployment_record']
    observed = []
    runtime_hashes = set()
    session_pins = []
    # The member records retain start identity; receipts prove which PIDs were used.
    for member in serve['instance_sessions']:
        path = data/'serving'/member['session_id']/'session.json'
        document = json.loads(path.read_bytes())
        if (document['session_id'] != member['session_id'] or document['pid'] != member['pid'] or
                document['identity']['port'] != member['port']):
            raise ValueError('run attached to a different serving process')
        from speech_aware_evidence_acquisition.core.config import read_fragment, fragment_hash
        fragment = read_fragment(repo, 'model', deployed['model_fragment'])
        if document['identity']['model_fragment_sha256'] != fragment_hash(fragment):
            raise ValueError('model fragment differs from attached identity')
        runtime_hashes.add(document['identity']['runtime_receipt_sha256'])
        observed.append([member['instance'], member['session_id'], member['pid'], document['started_at']])
        session_pins.append(pin(path, data))
    if len(runtime_hashes) != 1 or len(observed) != deployed['instance_count']:
        raise ValueError('incomplete or inconsistent deployment evidence')
    identity = {'model_fragment':deployed['model_fragment'],
                'runtime_receipt_sha256':next(iter(runtime_hashes)),
                'profile_sha256':deployed['profile_sha256'],
                'roster_sha256':hashlib.sha256(json.dumps(sorted(ids),separators=(',',':')).encode()).hexdigest(),
                'member_processes':sorted(observed), 'admitted_level':deployed['admitted_level']}
    return {'paths':paths, 'manifest':manifest, 'identity':identity, 'ids':ids,
            'pins':{'manifest':pin(paths['manifest'],data), 'receipts':pin(receipts_path,data)},
            'session_pins':session_pins}


def build(repo, data, *, direct_run, direct_attempt, agent_run, agent_attempt, agent_reference,
          allocation, read_id):
    require_read(repo, read_id)
    d = run_evidence(data, direct_run, direct_attempt, repo)
    a = run_evidence(data, agent_run, agent_attempt, repo)
    if d['identity'] != a['identity']:
        raise ValueError('Agent/Direct roster, model or actual deployment differs')
    model = repo/'configs/model'/f"{d['identity']['model_fragment']}.json"
    runtime = repo/'docs/receipts/runtime-vllm-bf16-thinking-npu.json'
    if digest(runtime) != d['identity']['runtime_receipt_sha256']:
        raise ValueError('current runtime receipt differs from execution')
    reference = json.loads(agent_reference.read_bytes())
    if (reference['diagnostic_identity']['run_id'] != agent_run or
            reference['diagnostic_identity']['attempt_id'] != agent_attempt or
            reference['sealed_hashes'] != {k:digest(a['paths'][k]) for k in ('outputs','trace')}):
        raise ValueError('Agent cost reference is not derived from the sealed run')
    experiment = d['manifest']['config']['fragments']['experiment']
    if not isinstance(experiment, str) or Path(experiment).name != experiment:
        raise ValueError('invalid executed experiment identity')
    direct_cell = json.loads((repo/'configs/experiment'/f'{experiment}.json').read_bytes())
    if (direct_cell['cells'][0]['run_id'],direct_cell['cells'][0]['attempt_id']) != (direct_run,direct_attempt):
        raise ValueError('Direct identity differs from registration')
    if set(direct_cell['populations'][0]['ids']) != set(d['ids']):
        raise ValueError('Direct executed roster differs from registration')
    cost = json.loads(allocation.read_bytes())
    from slurp_direct_port_smoke_read import allocation_cost
    devices = [card for member in json.loads(runtime.read_bytes())['deployment']['instances'] for card in member['device_ids']]
    ceiling = direct_cell['execution_plan']['planned_gpu_hours']
    allocation_cost(cost, commit=d['manifest']['study_commit'], devices=devices, ceiling=ceiling)
    identity = {**d['identity'], 'model_fragment_sha256':digest(model)}
    inputs = {name:pin(path,data) for name,path in {
        'trace':d['paths']['trace'], 'outputs':d['paths']['outputs'],
        'agent_reference':agent_reference, 'allocation':allocation}.items()}
    for side,evidence in (('direct',d),('agent',a)):
        inputs.update({side+'_'+k:v for k,v in evidence['pins'].items()})
        for i,p in enumerate(evidence['session_pins']): inputs[f'{side}_session_{i}']=p
    return {'schema':'saea-direct-port-smoke-read-inputs-v1', 'read_run_id':read_id,
            'study_commit':d['manifest']['study_commit'], 'sample_ids':d['ids'],
            'devices':devices, 'registered_card_hours':ceiling, 'inputs':inputs,
            'comparison_identity':{'agent':identity,'direct':identity},
            'provenance':'Derived from sealed manifests, hash-validated flight receipts and attached member processes.',
            'derivation':{'direct_run':direct_run,'direct_attempt':direct_attempt,'agent_run':agent_run,'agent_attempt':agent_attempt,
                          'agent_reference':str(agent_reference.relative_to(data)), 'allocation':str(allocation.relative_to(data)),
                          'read_id':read_id}}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo-root',type=Path,default=REPO)
    ap.add_argument('--data-root',type=Path,required=True)
    for key in ('direct-run','direct-attempt','agent-run','agent-attempt','read-id'):
        ap.add_argument('--'+key,required=True)
    for key in ('agent-reference','allocation','out'):
        ap.add_argument('--'+key,type=Path,required=True)
    args=ap.parse_args(argv)
    result=build(args.repo_root,args.data_root,direct_run=args.direct_run,direct_attempt=args.direct_attempt,
                 agent_run=args.agent_run,agent_attempt=args.agent_attempt,agent_reference=args.agent_reference,
                 allocation=args.allocation,read_id=args.read_id)
    publish_new(args.out,(json.dumps(result,sort_keys=True,indent=2)+'\n').encode())
    print(json.dumps({'written':str(args.out),'sha256':digest(args.out),'model_contacts':0}))

if __name__=='__main__':main()
