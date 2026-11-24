"""Recoverable run-local immutable object inventory, never a permanent backup.

Objects commit before graph publication. Published graphs hardlink objects and
remain recoverable independently of inventory receipts. No automatic eviction:
retention is permitted only after independently verified permanent custody and
closure of all original consumers. Partial objects are never advertised.
"""
import hashlib
import json
import os
import tarfile
import io
import uuid
from pathlib import Path
from dependencies import require
from storage import check_running, environment, claim_object
from run_package import read, write, pin


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''): h.update(b)
    return h.hexdigest()


def syncdir(path):
    fd = os.open(path,os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def object_path(cache, sha):
    require(len(sha)==64 and all(c in '0123456789abcdef' for c in sha),'invalid immutable object hash')
    return Path(cache)/'objects'/sha


def verified(cache, sha, size):
    target=object_path(cache,sha); marker=Path(cache)/'receipts'/(sha+'.json')
    if not target.is_file() or not marker.is_file() or target.is_symlink(): return False
    try:
        receipt=json.loads(marker.read_text())
        return receipt==dict(schema='closed-object-v1',sha256=sha,bytes=size) and target.stat().st_size==size and digest(target)==sha
    except (OSError,ValueError): return False


def inventory(cache):
    """Revalidate complete content on restart/negotiation; corrupt means miss."""
    cache=Path(cache); result={}
    if not cache.exists(): return result
    for receipt in (cache/'receipts').glob('*.json'):
        try:
            row=json.loads(receipt.read_text()); sha=receipt.stem
            if verified(cache,sha,row['bytes']): result[sha]=row['bytes']
        except (OSError,ValueError,KeyError,TypeError): continue
    return result


def capacity_window(cache, remaining, interval):
    # Admit the exact forthcoming bytes against the live run ledger.
    from storage import current_scope
    rows = check_running(cache, remaining)
    reserve = max(0, rows[0]['remaining_bytes'])
    width = min(remaining, interval)
    scope = current_scope()
    if scope and 'noncheckpoint_limit_bytes' in scope and rows[0]['scope'] == scope['identity']:
        width = min(width, scope['noncheckpoint_limit_bytes'] - rows[0]['noncheckpoint_bytes'])
    require(width > 0, 'orderly stop: no available transfer capacity')
    return width, reserve


def check_open_filesystem(fd, remaining, write_bytes):
    # fstatvfs observes the actual open local file, including changes by other jobs.
    s = os.fstatvfs(fd)
    require(s.f_bavail*s.f_frsize >= max(20*1024**3,.1*s.f_blocks*s.f_frsize) + remaining + write_bytes
            and (not s.f_files or s.f_favail >= .05*s.f_files),
            'orderly stop: local filesystem capacity fell during transfer')


def install(cache, sha, size, stream, *, capacity_interval=None):
    cache=Path(cache)
    if verified(cache,sha,size):
        target=object_path(cache,sha);claim_object(target,sha,size);return target
    environment(cache,0)
    target=object_path(cache,sha);target.parent.mkdir(parents=True,exist_ok=True)
    pending=target.with_name(sha+'.pending-'+uuid.uuid4().hex)
    h=hashlib.sha256(); copied=0
    if capacity_interval is not None:
        require(type(capacity_interval) is int and capacity_interval > 0, 'invalid transfer capacity interval')
    credit = reserve = 0
    with pending.open('xb') as f:
        while True:
            if capacity_interval is not None and credit == 0 and copied < size:
                credit, reserve = capacity_window(cache, size-copied, capacity_interval)
            chunk=stream.read(min(1024**2, credit) if credit else 1024**2)
            if not chunk: break
            if capacity_interval is None:
                check_running(cache,size-copied)
            else:
                require(len(chunk) <= credit, 'object exceeds admitted capacity')
                check_open_filesystem(f.fileno(), reserve, len(chunk))
                credit -= len(chunk)
            copied+=len(chunk);require(copied<=size,'object exceeds declared length')
            h.update(chunk);f.write(chunk)
        f.flush();os.fsync(f.fileno())
    if capacity_interval is not None:
        check_running(cache, 0)
    require(copied==size and h.hexdigest()==sha,'incoming object content mismatch')
    # Preserve corrupt/half-committed evidence; no authoritative source unlink.
    if target.exists():
        require(not target.is_symlink(),'inventory object symlink')
        retained=target.with_name(sha+'.corrupt-'+uuid.uuid4().hex)
        os.rename(target,retained)
    os.replace(pending,target);syncdir(target.parent)
    marker=cache/'receipts'/(sha+'.json')
    if marker.exists():
        try: same=json.loads(marker.read_text())==dict(schema='closed-object-v1',sha256=sha,bytes=size)
        except (ValueError,OSError):same=False
        if not same: os.rename(marker,marker.with_name(marker.name+'.corrupt-'+uuid.uuid4().hex));syncdir(marker.parent)
    write(marker,dict(schema='closed-object-v1',sha256=sha,bytes=size))
    claim_object(target,sha,size)
    return target


def manifest_from_tar(data):
    archive=tarfile.open(fileobj=io.BytesIO(data)) if isinstance(data,bytes) else tarfile.open(data)
    return archive


def validate_manifest(manifest):
    require(manifest.get('schema')=='closed-cas-v1','typed closed CAS schema')
    files=manifest['files'];objects=manifest['objects']
    require(isinstance(files,dict) and isinstance(objects,dict),'typed manifest maps')
    expected={}
    for name,ref in files.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'closed graph path escape')
        require(Path(ref['path']).is_absolute() and type(ref['bytes']) is int and ref['bytes']>=0,'closed graph logical ref')
        object_path('.',ref['sha256'])
        require(ref['sha256'] not in expected or expected[ref['sha256']]==ref['bytes'],'hash size conflict')
        expected[ref['sha256']]=ref['bytes']
    require(set(expected)==set(objects),'object graph incomplete')
    for sha,row in objects.items():
        require(row['bytes']==expected[sha] and row['member'] in (None,'objects/'+sha),'object wire member mismatch')
    require('RETURN.json' in files,'closed graph missing consumer return')
    require(manifest['closure']['return_sha256']==files['RETURN.json']['sha256'],'return identity drift')
    closures=[v for v in files.values() if v['path']==manifest['closure']['closed_path']]
    require(len(closures)==1 and closures[0]['sha256']==manifest['closure']['closed_sha256'],'closed identity drift')
    return files


def unpack(data,destination,cache,*,expected_boot=None,capacity_interval=None):
    destination=Path(destination);cache=Path(cache)
    incoming=destination.with_name(destination.name+'.incoming')
    with manifest_from_tar(data) as tar:
        members=tar.getmembers()
        require(members and members[0].name=='MANIFEST.json' and members[0].size<=16*1024**2,'manifest first/size bound')
        manifest=json.loads(tar.extractfile(members[0]).read());files=validate_manifest(manifest)
        if expected_boot: require(manifest['closure']['source_boot']==expected_boot,'closed source boot drift')
        require(cache.resolve().stat().st_dev==destination.parent.resolve().stat().st_dev if cache.exists() else True,'inventory and graph must share a local mount')
        incoming.mkdir(parents=True,exist_ok=True)
        good={}
        wire={r['member']:s for s,r in manifest['objects'].items() if r['member'] is not None}
        seen=set()
        for item in members[1:]:
            require(item.isfile() and item.name in wire and item.name not in seen,'duplicate/unknown closed object member')
            sha=wire[item.name];row=manifest['objects'][sha]
            require(item.size==row['bytes'],'closed object byte length')
            # Read and hash even if another pull already installed this object.
            with tar.extractfile(item) as stream:
                if verified(cache,sha,row['bytes']):
                    h=hashlib.sha256()
                    for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
                    require(h.hexdigest()==sha,'redundant wire object drift')
                else: install(cache,sha,row['bytes'],stream,capacity_interval=capacity_interval)
                good[sha]=True
            seen.add(item.name)
        require(seen==set(wire),'missing transmitted object')
        for sha,row in manifest['objects'].items():
            require(sha in good or verified(cache,sha,row['bytes']),'missing/corrupt inventory object; repull same closed source')
        closure=manifest['closure'];closed=json.loads(object_path(cache,closure['closed_sha256']).read_text())
        require(closed['immutable'] and closed['source_boot']==closure['source_boot'],'source closure boot/immutability changed')
        require(closed['return_ref']['sha256']==closure['return_sha256'] and closed['return_ref']['path']==files['RETURN.json']['path'],'source closure return changed')
        if destination.exists():
            prior=read(destination/'WIRE-MANIFEST.json')
            require(prior['files']==manifest['files'] and prior['closure']==manifest['closure'],'published graph identity changed')
            restore_graph_links(destination,manifest,cache)
            from transport import restore_locations
            result=restore_locations(destination)
            if not isinstance(data,bytes):compact_wire(Path(data),manifest)
            return result
        for name,ref in files.items():
            target=incoming/name;target.parent.mkdir(parents=True,exist_ok=True)
            require(not target.is_symlink() and target.resolve().is_relative_to(incoming.resolve()),'incoming graph symlink/escape')
            if target.exists():require(target.stat().st_size==ref['bytes'] and digest(target)==ref['sha256'],'incoming graph collision')
            else:os.link(object_path(cache,ref['sha256']),target)
        paths={ref['path']:str(destination/name) for name,ref in files.items()}
        write(incoming/'LOCATION-MAP.json',paths)
        write(incoming/'CLOSED-MANIFEST.json',files)
        write(incoming/'WIRE-MANIFEST.json',manifest)
        write(incoming/'CLOSED-COMMIT.json',dict(manifest_sha256=digest(incoming/'CLOSED-MANIFEST.json'),
              wire_sha256=digest(incoming/'WIRE-MANIFEST.json'),files=len(files),immutable=True,
              closure=closure,cache_role='local_reusable_custody_not_permanent_backup'))
        for folder in sorted([incoming,*[p for p in incoming.rglob('*') if p.is_dir()]],key=lambda p:len(p.parts),reverse=True):syncdir(folder)
        os.rename(incoming,destination);syncdir(destination.parent)
    from transport import restore_locations
    result=restore_locations(destination)
    # The durable graph and CAS replace the large transient wire copy. Retain a
    # manifest-only replay envelope at the same original tar interface.
    if not isinstance(data,bytes):compact_wire(Path(data),manifest)
    return result


def restore_graph_links(destination,manifest,cache):
    """Repair missing published links from verified CAS; never overwrite drift.

    A cold-restored graph keeps its original immutable identity. All graph
    metadata is verified before any links are created. Corrupt existing bytes
    remain evidence and cause failure, rather than being silently repaired.
    """
    destination=Path(destination).resolve()
    validate_manifest(manifest)
    commit=read(destination/'CLOSED-COMMIT.json')
    prior=read(destination/'WIRE-MANIFEST.json')
    files=read(destination/'CLOSED-MANIFEST.json')
    paths=read(destination/'LOCATION-MAP.json')
    require(commit['immutable'] and prior['files']==manifest['files'] and
            prior['closure']==manifest['closure'] and files==manifest['files'],
            'cold restore published graph identity changed')
    require(digest(destination/'CLOSED-MANIFEST.json')==commit['manifest_sha256'] and
            digest(destination/'WIRE-MANIFEST.json')==commit['wire_sha256'] and
            commit['closure']==manifest['closure'], 'cold restore published commit drift')
    require(paths=={ref['path']:str(destination/name) for name,ref in files.items()},
            'cold restore location map drift')
    missing=[]
    for name,ref in files.items():
        target=destination/name
        require(not target.is_symlink() and target.resolve().is_relative_to(destination),
                'cold restore graph path escape')
        if target.exists():
            require(target.is_file() and target.stat().st_size==ref['bytes'] and
                    digest(target)==ref['sha256'],'cold restore existing graph object corrupt')
        else:
            require(verified(cache,ref['sha256'],ref['bytes']), 'cold restore object absent from verified CAS')
            missing.append((target,ref))
    for target,ref in missing:
        target.parent.mkdir(parents=True,exist_ok=True)
        require(not target.is_symlink() and target.resolve().is_relative_to(destination),
                'cold restore graph parent changed')
        try:os.link(object_path(cache,ref['sha256']),target)
        except FileExistsError:
            require(not target.is_symlink() and target.is_file() and
                    target.stat().st_size==ref['bytes'] and digest(target)==ref['sha256'],
                    'cold restore concurrent object collision')
        syncdir(target.parent)
    return len(missing)


def compact_wire(path,manifest):
    compact=json.loads(json.dumps(manifest))
    for row in compact['objects'].values():row['member']=None
    pending=path.with_name(path.name+'.compact')
    raw=json.dumps(compact,sort_keys=True).encode()
    with pending.open('wb') as stream:
        with tarfile.open(fileobj=stream,mode='w') as tar:
            info=tarfile.TarInfo('MANIFEST.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
        stream.flush();os.fsync(stream.fileno())
    os.replace(pending,path);syncdir(path.parent)


def cold_graph_identity(destination):
    """Authenticate a published graph's permanent location without hydration."""
    destination=Path(destination).resolve()
    ticket=read(destination/'COLD-CUSTODY.json')
    require(ticket['schema']=='closed-graph-cold-custody-v1' and
            ticket['destination']==str(destination),'cold custody destination drift')
    complete=ticket['complete_ref'];record=read(complete)
    require(record.get('schema')=='permanent-closed-cas-v1' and
            record['client_boot']!=record['intent']['source_boot'] and
            record.get('independent_file_and_parent_fsync') is True,
            'independent permanent custody required')
    graph=Path(complete['path']).parent/'CLOSED-MANIFEST.json'
    require(digest(graph)==record['graph_sha256'],'cold permanent graph changed')
    manifest=json.loads(graph.read_text());validate_manifest(manifest)
    wire=read(destination/'WIRE-MANIFEST.json');commit=read(destination/'CLOSED-COMMIT.json')
    require(commit['immutable'] and digest(destination/'WIRE-MANIFEST.json')==commit['wire_sha256'] and
            digest(destination/'CLOSED-MANIFEST.json')==commit['manifest_sha256'] and
            manifest['files']==wire['files']==read(destination/'CLOSED-MANIFEST.json') and
            manifest['closure']==wire['closure']==commit['closure'],'cold published identity drift')
    require(manifest['closure']['source_boot']==record['intent']['source_boot'] and
            manifest['closure']['return_sha256']==record['watermark'],'cold permanent watermark drift')
    require(read(destination/'LOCATION-MAP.json')==
            {r['path']:str(destination/n) for n,r in manifest['files'].items()},'cold location map changed')
    return ticket,record,manifest


def authorize_cold_graph(destination,complete,cache):
    """Record verified permanent custody; this function never deletes files."""
    destination=Path(destination).resolve();cache=Path(cache).resolve()
    from transport import restore_locations
    restore_locations(destination)
    record=read(complete);graph=Path(pin(complete)['path']).parent/'CLOSED-MANIFEST.json'
    require(digest(graph)==record['graph_sha256'],'permanent graph drift')
    manifest=json.loads(graph.read_text());validate_manifest(manifest)
    require(record.get('schema')=='permanent-closed-cas-v1' and
            record['client_boot']!=record['intent']['source_boot'] and
            record.get('independent_file_and_parent_fsync') is True,'permanent client not accepted')
    wire=read(destination/'WIRE-MANIFEST.json')
    require(manifest['files']==wire['files'] and manifest['closure']==wire['closure'] and
            manifest['closure']['source_boot']==record['intent']['source_boot'] and
            manifest['closure']['return_sha256']==record['watermark'],'permanent graph identity mismatch')
    for h,row in manifest['objects'].items():
        source=object_path(record['object_root'],h)
        require(not source.is_symlink() and source.is_file() and
                source.stat().st_size==row['bytes'] and digest(source)==h,
                'permanent object unavailable; preserve local graph')
    require(cache.exists() and cache.stat().st_dev==destination.stat().st_dev,'cold cache mount mismatch')
    return write(destination/'COLD-CUSTODY.json',dict(schema='closed-graph-cold-custody-v1',
                 destination=str(destination),complete_ref=pin(complete),cache_root=str(cache)))


def cold_graph_loaders(destination):
    """Register exact missing objects for lazy read; never hydrate all history."""
    destination=Path(destination).resolve()
    if not (destination/'COLD-CUSTODY.json').exists():return {}
    ticket,record,manifest=cold_graph_identity(destination)
    loaders={}
    for name,ref in manifest['files'].items():
        target=destination/name
        # Control JSON stays local for dependency traversal and source closure.
        if target.exists():continue
        require(target.suffix!='.json' and not target.is_symlink() and
                target.resolve().is_relative_to(destination),'cold control missing/path escape')
        def restore(name=name,ref=ref):
            current,receipt,graph=cold_graph_identity(destination)
            require(current==ticket and graph['files'][name]==ref,'cold read custody changed')
            target=destination/name
            require(not target.is_symlink() and target.resolve().is_relative_to(destination),'cold read path escape')
            if target.exists():
                require(target.is_file() and target.stat().st_size==ref['bytes'] and digest(target)==ref['sha256'],
                        'cold read existing corruption')
                return target
            source=object_path(receipt['object_root'],ref['sha256'])
            require(not source.is_symlink() and source.is_file() and source.stat().st_size==ref['bytes'] and
                    digest(source)==ref['sha256'],'cold read permanent object corrupt')
            with source.open('rb') as stream:install(current['cache_root'],ref['sha256'],ref['bytes'],stream)
            target.parent.mkdir(parents=True,exist_ok=True)
            try:os.link(object_path(current['cache_root'],ref['sha256']),target)
            except FileExistsError:
                require(not target.is_symlink() and digest(target)==ref['sha256'],'cold concurrent collision')
            syncdir(target.parent)
            return target
        loaders[str(target)]=restore
    return loaders


def restore_archive(complete,destination,cache):
    """Independent cold restore: all objects must exist in permanent custody."""
    complete=Path(complete);record=json.loads(complete.read_text());root=complete.parent
    require(record.get('schema')=='permanent-closed-cas-v1' and
            record['client_boot']!=record['intent']['source_boot'] and
            record.get('independent_file_and_parent_fsync') is True,
            'independent complete permanent custody required')
    graph=root/'CLOSED-MANIFEST.json';require(digest(graph)==record['graph_sha256'],'permanent graph digest drift')
    manifest=json.loads(graph.read_text());validate_manifest(manifest)
    require(manifest['closure']['source_boot']==record['intent']['source_boot'] and
            manifest['closure']['return_sha256']==record['watermark'],'permanent custody watermark drift')
    permanent=Path(record['object_root'])
    for sha,row in manifest['objects'].items():
        source=object_path(permanent,sha)
        require(source.is_file() and source.stat().st_size==row['bytes'] and digest(source)==sha,'permanent full object missing/corrupt')
        with source.open('rb') as f: install(cache,sha,row['bytes'],f)
    envelope=json.loads(json.dumps(manifest))
    for row in envelope['objects'].values():row['member']=None
    raw=json.dumps(envelope,sort_keys=True).encode();buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w') as tar:
        info=tarfile.TarInfo('MANIFEST.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    return unpack(buf.getvalue(),destination,cache,expected_boot=record['intent']['source_boot'])


def release_diagnostic_graph(spec, *, iteration_guard=None):
    """Reclaim only an explicitly released, archived local diagnostic graph.

    No remote removal or capacity credit is inferred. The existing iteration
    and transfer guards exclude consumers and publishers. Shared graph links
    can be retained while retiring this graph's own links. A durable exact
    intent makes partial unlink resumable.
    """
    import fcntl
    from contextlib import ExitStack
    require(spec['schema']=='local-diagnostic-graph-release-v1','diagnostic release schema')
    config=read(spec['config_ref']);root=Path(config['output_root']).resolve()
    graph=Path(spec['graph']).resolve();cache=Path(spec['cache_root']).resolve()
    require(graph.is_relative_to(root) and cache.is_relative_to(root) and graph!=root and cache!=root,
            'diagnostic release outside exact run')
    require(config['run_id']==spec['run_id'],'diagnostic run identity drift')
    with ExitStack() as stack:
        # The original controller owns iteration.guard for its entire run.
        # Borrow its live file description, never unlock it or create a second
        # scheduler. A standalone maintenance invocation still acquires it.
        guards=[cache/'transfer.guard']
        if iteration_guard is None:
            guards.insert(0,root/'iteration.guard')
        else:
            require(not iteration_guard.closed,'controller guard closed')
            fd=iteration_guard.fileno();actual=os.fstat(fd);expected=(root/'iteration.guard').stat()
            require((actual.st_dev,actual.st_ino)==(expected.st_dev,expected.st_ino),
                    'controller guard not the original run inode')
            # Reassert on the SAME open file description. This cannot replace
            # another owner's lock; EWOULDBLOCK propagates before any deletion.
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for guard_path in guards:
            guard=stack.enter_context(guard_path.open('ab'))
            fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cursor=read(spec['cursor_ref'])
        require(cursor['config_ref']==spec['config_ref'] and not cursor.get('pending_stage') and
                cursor['phase'] in ('capture','dev','freeze','done'), 'active or unknown diagnostic consumer')
        release=read(spec['consumer_release_ref'])
        require(release['schema']=='diagnostic-consumers-released-v1' and
                release['config_ref']==spec['config_ref'] and release['run_id']==spec['run_id'] and
                type(release['policy_step']) is int and release['policy_step']<cursor['step'] and
                release['unknown_requests']==0 and release['consumer_state']=='terminal',
                'diagnostic dependency release absent')
        acceptance=read(release['acceptance_ref'])
        require(acceptance['recomputed']['passed'] is True and acceptance['joint_output']['passed'] is True and
                acceptance['raw_ref']==release['raw_ref'],'failed diagnostic cannot be reclaimed')
        ticket,record,manifest=cold_graph_identity(graph)
        require(ticket['cache_root']==str(cache) and record['intent']['run_id']==spec['run_id'],
                'cold custody not for this run/cache')
        # Rollout closure is produced before the accepted HF raw. The release
        # binds that raw; each selected manifest must belong to both its exact
        # HTTP/worker chain and this independently archived graph below.
        raw=read(release['raw_ref'])
        from diagnostic_retention import owned_logits
        owned=owned_logits(raw)
        release_key=spec.get('release_key')
        require(release_key is None or release_key==release['raw_ref']['sha256'],
                'diagnostic release key must identify its raw owner')
        stem='DIAGNOSTIC-RELEASE'+('-'+release_key if release_key else '')
        marker=graph/(stem+'-INTENT.json' if release_key else 'DIAGNOSTIC-RELEASE-INTENT.json')
        # File selection comes only from original typed full-logits manifests.
        selected={}
        for ref in spec['logits_manifest_refs']:
            require(ref in owned,'diagnostic payload belongs to another raw policy')
            require(any(r==ref for r in manifest['files'].values()),'logits metadata absent from graph')
            meta=read(ref);tensor=meta['tensor_ref']
            require(meta['schema']=='omni-response-logits-v1' and meta['full_vocabulary'] is True and
                    meta['dtype']=='torch.bfloat16' and tensor['sha256']==meta['sha256'] and
                    tensor['bytes']==meta['bytes'] and Path(tensor['path']).suffix=='.bin',
                    'only typed full logits payloads eligible')
            for name,r in manifest['files'].items():
                if r==tensor:selected[name]=r
            require(any(r==tensor for r in selected.values()),'logits payload absent from graph')
        require(selected,'empty diagnostic release')
        expected=dict(schema='local-diagnostic-release-intent-v1',spec=spec,files=selected)
        shared_policy=spec.get('shared_links_policy','preserve_all')
        require(shared_policy in ('preserve_all','release_owned_keep_shared'),
                'unknown shared diagnostic retention policy')
        if marker.exists():
            require(read(marker)==expected,'diagnostic release restart identity drift')
        # Verify every permanent payload before any unlink, including on resume.
        by_hash={}
        for name,ref in selected.items():by_hash.setdefault(ref['sha256'],[]).append((name,ref))
        for h,items in by_hash.items():
            ref=items[0][1];permanent=object_path(record['object_root'],h)
            require(not permanent.is_symlink() and permanent.is_file() and permanent.stat().st_size==ref['bytes'] and
                    digest(permanent)==h,'permanent diagnostic missing; preserve source')
            paths=[graph/name for name,_ in items]+[object_path(cache,h)]
            live=[p for p in paths if p.exists()]
            for p in paths:
                require(not p.is_symlink() and p.resolve().is_relative_to(root),'diagnostic release path escape')
                require(p.exists() or marker.exists(),'missing local diagnostic before durable release')
            for p in live:
                require(p.is_file() and p.stat().st_size==ref['bytes'] and digest(p)==h,
                        'local diagnostic corrupt; preserve evidence')
            inodes={}
            for p in live:inodes.setdefault((p.stat().st_dev,p.stat().st_ino),[]).append(p)
            for same in inodes.values():
                require(same[0].stat().st_nlink>=len(same),'diagnostic hardlink accounting changed')
                if shared_policy=='preserve_all':
                    require(same[0].stat().st_nlink==len(same),'diagnostic has another live graph/reader link')
        if not marker.exists():write(marker,expected)
        reclaimed=0;unlinked=0;shared_objects=[]
        for h,items in by_hash.items():
            for p in [graph/name for name,_ in items]:
                if p.exists():
                    stat=p.stat()
                    if stat.st_nlink==1:reclaimed+=stat.st_blocks*512
                    p.unlink();syncdir(p.parent);unlinked+=1
            # The exact graph has relinquished its references. Other graphs,
            # current policies or readers still own all remaining hardlinks.
            # Keep the CAS entry until the last such owner retires; do not
            # misreport a removed logical reference as reclaimed disk bytes.
            cached=object_path(cache,h)
            if cached.exists():
                stat=cached.stat()
                if stat.st_nlink==1:
                    reclaimed+=stat.st_blocks*512
                    cached.unlink();syncdir(cached.parent);unlinked+=1
                else:shared_objects.append(h)
            receipt=cache/'receipts'/(h+'.json')
            if not cached.exists() and receipt.exists():receipt.unlink();syncdir(receipt.parent)
        # Historical callbacks are installed without reading any tensor bytes.
        from transport import restore_locations
        restore_locations(graph)
        from storage import commit
        result=dict(intent_ref=pin(marker),removed_links_this_attempt=unlinked,
                    observed_freed_blocks_bytes_this_attempt=reclaimed,remote_source_removed=False,
                    shared_objects_retained=shared_objects,capacity_admission_changed=False)
        commit(graph/(stem+'-DONE.json' if release_key else 'DIAGNOSTIC-RELEASED.json'),result,replace=True)
        return result


def restore_pull_inventory(graph, cache):
    """Before re-pulling a retired source, hydrate its exact missing CAS objects.

    Caller holds the original transfer guard. Never search unrelated archives
    or advertise a hash whose bytes have not been verified locally.
    """
    graph=Path(graph);cache=Path(cache)
    if not (graph/'COLD-CUSTODY.json').is_file():return
    ticket,record,manifest=cold_graph_identity(graph)
    require(Path(ticket['cache_root']).resolve()==cache.resolve(),'pull restore cache differs')
    for h,row in manifest['objects'].items():
        if verified(cache,h,row['bytes']):continue
        source=object_path(record['object_root'],h)
        require(not source.is_symlink() and source.is_file() and
                source.stat().st_size==row['bytes'] and digest(source)==h,
                'pull restore permanent object missing/corrupt')
        with source.open('rb') as stream:install(cache,h,row['bytes'],stream)


def retire_inventory(spec):
    """Terminal original run only; verify permanent custody before cache eviction.

    Published logical graphs retain their hardlinks and remain readable. This
    operation never claims their blocks reclaimed, never touches source, and
    retains all bad/partial evidence. A future run restores from permanent CAS.
    """
    import fcntl
    from dependencies import local_path
    config=read(spec['iteration_config_ref']);cursor=read(spec['cursor_ref'])
    require(cursor['config_ref']==pin(spec['iteration_config_ref']) and cursor['phase']=='done','active/unknown run cannot evict inventory')
    require(config['run_id']==spec['run_id'] and Path(spec['cursor_ref']['path']).parent==Path(config['output_root']),'terminal run identity drift')
    cache=Path(spec['cache_root']);require(cache.resolve().is_relative_to(Path(config['output_root']).resolve()),'cache eviction outside exact run')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    covered={}
    for complete in spec['permanent_complete_refs']:
        record=read(complete);require(record['schema']=='permanent-closed-cas-v1','permanent commit required')
        require(record['intent']['run_id']==spec['run_id'] and record['client_boot']!=record['intent']['source_boot'],'permanent run/client identity drift')
        require(record['intent']['source_boot']!=boot,'affected source cannot evict from permanent evidence')
        graph=Path(complete['path']).parent/'CLOSED-MANIFEST.json'
        require(digest(graph)==record['graph_sha256'],'permanent lifecycle graph changed')
        manifest=read(graph);validate_manifest(manifest)
        for h,row in manifest['objects'].items():
            path=object_path(record['object_root'],h)
            require(path.is_file() and path.stat().st_size==row['bytes'] and digest(path)==h,'permanent lifecycle object unavailable; preserve cache')
            covered[h]=row['bytes']
    cache.mkdir(parents=True,exist_ok=True)
    with (cache/'transfer.guard').open('ab') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        objects=inventory(cache)
        require(all(covered.get(h)==n for h,n in objects.items()),'not every local object has permanent custody; preserve all')
        intent=dict(schema='terminal-inventory-eviction-v1',spec=spec,objects=objects,source_removed=False)
        marker=cache/'EVICTION-INTENT.json'
        if marker.exists():
            prior=read(marker);require(prior['spec']==spec,'eviction recovery identity drift')
            objects=prior['objects']
        else:write(marker,intent)
        reclaimed=0
        for h,n in objects.items():
            target=object_path(cache,h)
            if target.exists():
                require(verified(cache,h,n),'cache drift during terminal eviction; preserve evidence')
                if target.stat().st_nlink==1:reclaimed+=target.stat().st_blocks*512
                target.unlink();syncdir(target.parent)
            receipt=cache/'receipts'/(h+'.json')
            if receipt.exists():receipt.unlink();syncdir(receipt.parent)
        return write(cache/'EVICTED.json',dict(spec=spec,evicted_objects=len(objects),
            published_graph_hardlinks_retained=True,source_removed=False,local_cache_is_backup=False))


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--retire-config',required=True);a=p.parse_args()
    print(json.dumps(retire_inventory(read(a.retire_config))))
if __name__=='__main__':main()
