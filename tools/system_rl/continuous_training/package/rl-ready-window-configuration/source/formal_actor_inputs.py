"""Deliver the exact closed generation to the retained actor's read-only mounts."""

import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

from dependencies import local_path, require
from run_package import pin, read, write


def stage(template, config, directory):
    from transport import generation_runtime_inputs

    spec = config['formal_actor']
    mounts = template['capture_mounts']
    roots = {name: Path(path) for name, path in mounts['logical_roots'].items()}
    dynamic = set(mounts['formal_input_roles'])
    require(dynamic and dynamic <= roots.keys()
            and dynamic.isdisjoint({'receipts', 'cold', 'overlays', 'tools', 'index', 'policy_assets'}),
            'explicit formal input mount roles required')
    target = Path(mounts['formal_input_root'])
    prior = Path(template['restore_run'])
    require(target.parent == prior.parent, 'input custody must use the original local actor namespace')
    allowed = [Path(root) for root in spec['already_visible_roots']]
    if 'continuous_actor' in template:
        # Written by this same actor under its process-lifetime output mount.
        # These artifacts (including Adam shards) never travel back as inputs.
        owner_hot=Path(template['resident_owner_hot'])
        require(owner_hot.name=='hot' and owner_hot.parent.parent==prior.parent,'invalid actor output mount')
        allowed.append(owner_hot)
    require(all(root.is_absolute() and root != Path('/') and '..' not in root.parts for root in allowed),
            'invalid retained input roots')
    members = {}
    expanded = set()
    size = 0
    maximum = spec['input_max_bytes']
    require(type(maximum) is int and maximum > 0, 'invalid actor input byte budget')

    def collect(value, expand=True):
        nonlocal size
        if isinstance(value, list):
            for item in value:
                collect(item, expand)
        if not isinstance(value, dict):
            return
        if not {'path', 'sha256'} <= value.keys():
            for item in value.values():
                collect(item, expand)
            return
        path = Path(value['path'])
        require(path.is_absolute() and '..' not in path.parts, 'invalid logical actor input path')
        route = [name for name in dynamic if path.is_relative_to(roots[name]) and path != roots[name]]
        require(len(route) <= 1, 'overlapping actor input roots')
        if not route:
            require(any(path.is_relative_to(root) for root in [*allowed, *roots.values()]),
                    'actor input missing an explicit mount: '+str(path))
            return
        name = route[0]
        key = str(Path(name) / path.relative_to(roots[name]))
        ref = pin(value)
        if key in members:
            require(ref == members[key], 'actor input changed within the window')
        else:
            members[key] = ref
            size += ref['bytes']
            require(size <= maximum, 'actor input byte budget exceeded')
        if expand and path.suffix == '.json' and key not in expanded:
            expanded.add(key)
            document = read(ref)
            if isinstance(document,dict) and document.get('schema')=='omni-p-verified-replayed-tp4-v1':
                # bounded_tp4.replay_same_action reads these exact records. Embedded
                # worker reports contain provenance from inside the training image.
                for field in ('checkpoint_ref','activation_ref','precision_ref','cache_ref','profile_ref',
                              'reviewer_ref','paired_ref','response_ref','exit_ref','rank_refs',
                              'activated_publication_ref'):
                    collect(document[field],False)
            elif isinstance(document,dict) and 'activation_depth' in document and 'response_ref' in document:
                for field in ('activation_ref','precision_ref','cache_ref','response_ref'):
                    collect(document[field],False)
            elif isinstance(document, dict) and document.get('schema') == 'omni-dp8-publication-v1':
                # The actor restores its optimizer directly from /prior-checkpoint.
                # Inference provenance stays bound by its immutable endpoint receipt.
                collect(document['checkpoint_ref'], False)
                collect(document['assets'], False)
                if 'continuous_activation_ref' in document:
                    collect(document['continuous_activation_ref'])
                elif 'bounded_tp4_ref' in document:
                    collect(document['bounded_tp4_ref'])
            elif isinstance(document,dict) and document.get('schema')=='continuous-training-activation-v1':
                collect(document['checkpoint_ref'],False);collect(document['precision_lock_ref'])
                collect(document['profile_ref'],False);collect(document['lanes'])
            elif isinstance(document,dict) and document.get('purpose')=='precision_alignment':
                collect(document['accepted_checkpoint'],False)
                for key in ('activation_ref','precision_ref','profile_ref'):collect(document['assets'][key],False)
            elif isinstance(document,dict) and 'activation_depth' in document and 'adapter_sha256' in document:
                for key in ('activation_ref','precision_ref','cache_ref','switch_ref','previous_activation_ref'):
                    if key in document:collect(document[key],False)
            else:
                collect(document)

    generation = template['generation']
    if 'predecessor_step' in roots and ('continuous_actor' not in template or template['prior_step']==config.get('continuous_start_step')):
        commit=read(roots['predecessor_step']/'COMMITTED.json')
        for row in commit['ranks']:
            for name,digest in row['files'].items():
                if name.startswith('lora_adapter/'):
                    collect(dict(path=str(roots['predecessor_step']/name),sha256=digest),False)
    generation_runtime_inputs(dict(pending_capture=generation['pending_capture_ref'],
                                   generation_inventory=generation['generation_inventory_ref']), collect)
    native = read(generation['pending_capture_ref'])
    captured = read(read(native['pending_capture_ref'])['input_ref'])
    collect(captured['capture']['official_source_config_ref'])
    consumer = read(native['consumer_config_ref'])
    recording_map = read(consumer['recording_map_ref'])
    collect(consumer['recording_map_ref'], False)
    collect(recording_map['population_ref'], False)
    collect(template.get('zero_result_refs', []), False)
    require(members, 'empty actor input custody')
    runner_ref = pin(spec['input_receiver_ref'])
    runner = local_path(runner_ref['path']).read_text()
    require(hashlib.sha256(runner.encode()).hexdigest() == runner_ref['sha256'],
            'actor input receiver source changed')
    source = dict(run_id=template['run'], host_root=str(target), source_boot=spec['source_boot'])
    manifest = [dict(path=name, sha256=ref['sha256']) for name, ref in sorted(members.items())]
    metadata = dict(source=source, members=manifest, payload_bytes=size,
                    receiver_sha256=runner_ref['sha256'], roles=sorted(dynamic))
    expected = write(directory/'INPUT-INTENT.json', metadata)
    archive = directory/'actor-input.tar'
    if not archive.exists():
        from storage import storage
        storage.check_paths({'actor_input':directory}, min_free_bytes=20*1024**3+size)
        import os
        import tempfile
        with tempfile.NamedTemporaryFile(dir=directory) as pending:
            with tarfile.open(fileobj=pending,mode='w') as wire:
                for name, content in [('RETURN-MANIFEST.json', manifest), ('RETURN-SOURCE.json', source)]:
                    raw = json.dumps(content, sort_keys=True).encode()
                    info = tarfile.TarInfo(name)
                    info.size = len(raw)
                    wire.addfile(info, io.BytesIO(raw))
                for name, ref in sorted(members.items()):
                    info = tarfile.TarInfo(name)
                    info.size = ref['bytes']
                    with local_path(ref['path']).open('rb') as stream:
                        wire.addfile(info, stream)
            pending.flush()
            os.fsync(pending.fileno())
            os.link(pending.name, archive)
        from checkpoint_inventory import syncdir
        syncdir(directory)
    raw_pin = pin(archive)
    receiver = '''
root=Path(sys.argv[1])
if root.resolve()!=root or not root.parent.is_dir():
    raise ValueError('invalid local actor input namespace')
if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=sys.argv[3]:
    raise ValueError('actor boot changed')
marker=root/'INPUT-CLOSED.json'
if marker.exists():
    print(json.dumps(json.loads(marker.read_bytes())))
else:
    local_storage(root.parent)
    n=int(sys.argv[5])
    free=os.statvfs(root.parent)
    if free.f_bavail*free.f_frsize < 28*1024**3+2*n:
        raise ValueError('actor input storage exhausted')
    archive=sys.stdin.buffer.read(n+1)
    if len(archive)!=n or hashlib.sha256(archive).hexdigest()!=sys.argv[4]:
        raise ValueError('actor input archive changed')
    identity,rows=unpack_return(archive,root,run_id=sys.argv[2],host_root=str(root),expected_boot=sys.argv[3])
    for name in json.loads(sys.argv[6]):
        (root/name).mkdir(exist_ok=True)
    sync_directories([root])
    result=dict(identity=identity,files=len(rows),archive_sha256=sys.argv[4])
    save(marker,result)
    print(json.dumps(result))
'''
    # Load the existing receiver as a module, preserving the verified body
    # while excluding its CLI. Replays are input transfer only.
    # The receiver and its entry use the same namespace.
    code = ('namespace={"__name__":"actor_inputs","__file__":"/actor-input-receiver.py"}\n'
            'exec(compile('+repr(runner)+',"<input-receiver>","exec"),namespace)\n'
            'exec(compile('+repr(receiver)+',"<receive-inputs>","exec"),namespace)')
    command = shlex.join(['python3','-B','-c',code,str(target),template['run'],spec['source_boot'],
                          raw_pin['sha256'],str(raw_pin['bytes']),json.dumps(sorted(dynamic))])
    require(re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_-]*', template['host']),
            'invalid actor host binding')
    with archive.open('rb') as stream:
        reply = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',template['host'],command],
                               stdin=stream, capture_output=True, text=True, check=True,
                               timeout=spec['submission_timeout_seconds'])
    actual = json.loads(reply.stdout)
    require(actual == dict(identity=source, files=len(manifest), archive_sha256=raw_pin['sha256']),
            'actor input receipt differs')
    if 'continuous_actor' in template:
        # Publish from a sealed, private transfer into a single process-lifetime
        # read-only namespace. A conflicting logical path is never overwritten.
        policy=template['continuous_actor']
        projection=Path(mounts['resident_input_root'])
        require(projection.parent==target.parent and projection!=target,'invalid resident projection')
        module=local_path(pin(policy['input_projection_source_ref'])['path']).read_text()
        payload=json.dumps(dict(roster=manifest,roles=sorted(dynamic),identity=actual))
        call=module+"\nimport sys\na=json.load(sys.stdin)\nprint(json.dumps(publish(a['roster'],Path(sys.argv[1]),Path(sys.argv[2]),a['roles'],a['identity'])))"
        remote=shlex.join(['python3','-B','-c',call,str(target),str(projection)])
        reply=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',template['host'],remote],
            input=payload,capture_output=True,text=True,check=True,timeout=spec['submission_timeout_seconds'])
        proof=json.loads(reply.stdout)
        require(proof==dict(files=len(manifest),archive_sha256=raw_pin['sha256'],projection=str(projection)),
                'resident input projection not closed')
        write(directory/'PROJECTION.json',dict(proof,source_intent_ref=expected))
    return write(directory/'INPUT-CLOSED.json', dict(intent_ref=expected, remote=actual, archive_ref=raw_pin))
