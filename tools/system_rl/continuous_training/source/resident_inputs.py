"""Append verified window members to the stable, read-only actor bind mounts."""
import fcntl,hashlib,json,os
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def flush(directory):
    fd=os.open(directory,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def publish(roster, table, inbox, roles, identity):
    """All paths are owner-local. A command can consume ONLY a closed projection."""
    inbox=Path(inbox);table=Path(table);roles=set(roles)
    if (not inbox.is_absolute() or inbox!=inbox.resolve() or table!=table.resolve() or table==inbox
            or inbox.parent!=table.parent or not roles or
            any(len(Path(role).parts)!=1 or role in ('.','..') for role in roles)):
        raise ValueError('invalid resident input root')
    marker=json.loads((table/'INPUT-CLOSED.json').read_bytes())
    if marker!=identity:raise ValueError('staged input is not the requested closed transfer')
    if len(roster)!=marker['files'] or marker['identity']['host_root']!=str(table):
        raise ValueError('input roster differs')
    inbox.mkdir(exist_ok=True,mode=0o700)
    if inbox.stat().st_dev!=table.stat().st_dev:raise ValueError('resident inputs need same local device')
    touched={inbox,inbox.parent};seen=set();linked=0
    with (inbox/'projection.guard').open('ab') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        for role in roles:
            path=inbox/role
            if path.resolve()!=path:raise ValueError('mount root escaped')
            path.mkdir(exist_ok=True,mode=0o700)
        for row in roster:
            name=row['path'];rel=Path(name)
            if (not rel.parts or len(rel.parts)<2 or rel.parts[0] not in roles or '..' in rel.parts
                    or str(rel)!=name or name in seen):raise ValueError('invalid mounted member')
            seen.add(name);source=table/rel;dest=inbox/rel
            if (source.resolve()!=source or dest.resolve()!=dest or not source.is_file()
                    or digest(source)!=row['sha256']):raise ValueError('resident input source drift')
            if dest.exists():
                if not dest.is_file() or (not dest.samefile(source) and digest(dest)!=row['sha256']):
                    raise ValueError('conflicting immutable resident input: '+name)
                continue
            dest.parent.mkdir(parents=True,exist_ok=True)
            os.link(source,dest)
            linked+=1
            parent=dest.parent
            while parent!=inbox:touched.add(parent);parent=parent.parent
        for parent in sorted(touched,key=lambda p:-len(p.parts)):flush(parent)
    return dict(files=len(seen),archive_sha256=marker['archive_sha256'],projection=str(inbox))
