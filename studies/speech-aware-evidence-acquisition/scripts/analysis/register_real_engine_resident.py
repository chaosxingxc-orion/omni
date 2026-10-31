"""Register the next real500 resident after an observed owned topology switch.

Archive only exact stopped predecessor sessions. This never stops containers,
changes dataset records, or dispatches inference; guards and receipt verification
remain in the shared runtime.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def validate_predecessor(plan, originals, archived, sessions, previous):
    if previous['profile']!=plan['source']:raise ValueError('previous profile differs from transition')
    if len(archived)!=len(plan['before']):raise ValueError('archive count differs')
    by_name={d['Name'].lstrip('/'):d for d in originals}
    if set(by_name)!={c['name'] for c in plan['before']}:raise ValueError('original topology differs')
    coordinator=sessions[previous['service_session']]
    if coordinator['session_id']!=previous['service_session']:raise ValueError('previous coordinator differs')
    members=[c['name'] for c in plan['before']]
    if coordinator['deployment_instance_ids']!=members:raise ValueError('previous members differ')
    if set(sessions)!={previous['service_session'],*members}:raise ValueError('session set differs')
    for name,doc in sessions.items():
        if doc['identity']['model_fragment']!=previous['model_fragment']:raise ValueError('previous model differs')
        original=by_name[members[0] if name==previous['service_session'] else name]
        if doc['pid']!=original['State']['Pid']:raise ValueError('previous session PID differs from observed original')
    for c,archive in zip(plan['before'],archived):
        original=by_name[c['name']]
        if archive['Id']!=plan['original_ids'][c['name']] or archive['Id']!=original['Id']:raise ValueError('archive ID differs')
        if archive['Name']!='/'+plan['old_names'][c['name']]:raise ValueError('archive name differs')
        if archive['State']['Running'] or archive['RestartCount']:raise ValueError('archive running or restarted')
        if archive['Image']!=original['Image'] or archive['Config'].get('Labels',{}).get('saea.deployment.profile')!=plan['source']:raise ValueError('archive ownership differs')


def restore_receipt_addresses(data, archive):
    """Keep every archived receipt resolvable at its original content address.

    Whole serving directories move out of the resident scan. Their immutable
    receipt stores must also exist beside the old path referenced by bundles.
    Recreate only content-addressed bytes, never a resident session marker.
    """
    from speech_aware_evidence_acquisition.core.session.receipt import store_session_receipt_bytes
    manifest = read(archive / 'manifest.json')
    restored = 0
    for name, files in manifest.items():
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('unsafe archived session name')
        source = archive / name
        for rel, digest in files.items():
            path = (source / rel).resolve()
            if not path.is_relative_to(source.resolve()) or sha(path) != digest:
                raise ValueError('archived serving bytes differ')
        receipt = source / 'session-receipt.json'
        if 'session-receipt.json' not in files:
            raise ValueError('archived session receipt absent')
        candidates = [receipt] + [source / rel for rel in files
            if rel.startswith('session-receipt.json.archive/')]
        for path in candidates:
            raw = path.read_bytes()
            if path != receipt and path.name != hashlib.sha256(raw).hexdigest() + '.json':
                raise ValueError('archived content address differs')
            store_session_receipt_bytes(data / 'serving' / name / 'session-receipt.json', raw)
            restored += 1
    return restored


def register(repo,data,candidate,previous_candidate,runtime,transition,archived_inspect):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    from speech_aware_evidence_acquisition.core.config import read_fragment
    from speech_aware_evidence_acquisition.core.session.serving import find_resident_session
    from speech_aware_evidence_acquisition.core.session.vllm_resident import register_deployment
    assert_execution_environment()
    spec=read(repo/'docs/readiness'/candidate/'candidate.json')
    previous=read(repo/'docs/readiness'/previous_candidate/'candidate.json')
    plan=read(transition/'plan.json');completed=read(transition/'apply.complete.json')
    if plan['target']!=spec['profile'] or not completed['containers']:raise ValueError('target transition incomplete')
    receipt=runtime/'runtime-receipt.json'
    if sha(receipt)!=read(runtime/'complete.json')['runtime_receipt_sha256']:raise ValueError('runtime receipt differs')
    names=[c['name'] for c in plan['after']]
    observations=read(runtime/'observations.json')
    expected=read(transition/'candidate-container-ids.json')
    live=read(runtime/'docker-identity.json')
    if len(live)!=len(names):raise ValueError('observed container count differs')
    for c,d in zip(plan['after'],live):
        if d['Id']!=expected[c['name']] or not d['State']['Running'] or d['RestartCount']:raise ValueError('target identity changed')
        if d['State']['Pid']!=observations[c['name']]['pid']:raise ValueError('observed process changed')
    target=repo/'docs'/spec['runtime_receipt']
    if target.exists():
        if target.read_bytes()!=receipt.read_bytes():raise ValueError('receipt path occupied')
    else:publish_new(target,receipt.read_bytes())
    with (data/'locks/generation.guard').open('r+') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (data/'locks/flight.lock').exists():raise ValueError('flight lease exists')
        with (data/'.vllm-resident-registration.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            resident=find_resident_session(data)
            if resident is None:raise ValueError('previous resident absent; inspect interrupted transition')
            if resident[1]['session_id']!=previous['service_session']:raise ValueError('unexpected resident')
            old_names=[previous['service_session']]+resident[1]['deployment_instance_ids']
            sessions={n:read(data/'serving'/n/'session.json') for n in old_names}
            archives=read(archived_inspect)
            validate_predecessor(plan,read(transition/'original-docker-inspect.json'),archives,sessions,previous)
            for doc in sessions.values():
                if Path('/proc',str(doc['pid'])).exists():raise ValueError('predecessor PID occupied; inspect before archiving')
            archive=runtime/'previous-serving-archive';archive.mkdir(exist_ok=False)
            manifest={n:{str(p.relative_to(data/'serving'/n)):sha(p) for p in (data/'serving'/n).rglob('*') if p.is_file()} for n in old_names}
            publish_new(archive/'manifest.json',(json.dumps(manifest,sort_keys=True)+'\n').encode())
            for n in old_names:os.rename(data/'serving'/n,archive/n)
            restore_receipt_addresses(data, archive)
            publish_new(archive/'complete.json',b'{"archived":true}\n')
        paths=register_deployment(data_root=data,fragment=read_fragment(repo,'model',spec['model_fragment']),model_fragment=spec['model_fragment'],receipt_path=target,instances=observations,session_id=spec['service_session'])
        publish_new(runtime/'registration-complete.json',(json.dumps({'session':spec['service_session'],'files':[str(p) for p in paths],'model_calls':0})+'\n').encode())
    return {'registered':spec['service_session'],'model_calls':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','data','runtime','transition','archived-inspect'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('candidate','previous-candidate'):p.add_argument('--'+key,required=True)
    a=p.parse_args();print(json.dumps(register(a.repo,a.data,a.candidate,a.previous_candidate,a.runtime,a.transition,a.archived_inspect)))
