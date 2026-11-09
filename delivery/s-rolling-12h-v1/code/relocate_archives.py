"""Move explicit immutable source archives to another disk, preserving paths."""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    dest = Path(cfg['destination'])
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'relocate.guard').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows=[]
        for raw in cfg['files']:
            source=Path(raw);target=dest/source.name
            assert source.parent==Path(cfg['source_directory']) and source.name.startswith('omni-') and source.suffix=='.tar'
            if source.is_symlink():
                assert source.resolve()==target.resolve() and target.is_file()
                rows.append(dict(source=str(source),target=str(target),status='already_relocated',bytes=target.stat().st_size))
                continue
            before=source.stat()
            assert source.is_file() and time.time()-before.st_mtime>3600, 'archive is not stable'
            assert shutil.disk_usage(dest).free>before.st_size*2+5*1024**3, 'destination headroom'
            ids=subprocess.check_output(['docker','ps','-q'],text=True).split()
            active=json.loads(subprocess.check_output(['docker','inspect',*ids],text=True)) if ids else []
            assert not any(Path(m['Source'])==source for d in active for m in d.get('Mounts',[])), 'active file bind mount'
            for entry in Path('/proc').glob('[0-9]*/fd/*'):
                try: opened=entry.resolve()
                except OSError: continue
                assert opened!=source,'archive currently open'
            sha=digest(source)
            if args.apply:
                if not target.exists():
                    tmp=target.with_name(target.name+'.pending')
                    with source.open('rb') as src,tmp.open('xb') as out:
                        shutil.copyfileobj(src,out,4*1024*1024);out.flush();os.fsync(out.fileno())
                    shutil.copystat(source,tmp)
                    assert digest(tmp)==sha,'copy checksum mismatch'
                    os.replace(tmp,target)
                else: assert digest(target)==sha,'existing destination differs'
                now=source.stat()
                assert (now.st_ino,now.st_size,now.st_mtime_ns)==(before.st_ino,before.st_size,before.st_mtime_ns),'source changed'
                link=source.with_name(source.name+'.relocate-link')
                link.symlink_to(target)
                os.replace(link,source)
                for folder in (dest,source.parent):
                    fd=os.open(folder,os.O_RDONLY);os.fsync(fd);os.close(fd)
                assert digest(source)==sha,'old access path changed content'
            rows.append(dict(source=str(source),target=str(target),sha256=sha,bytes=before.st_size,status='relocated' if args.apply else 'verified_plan'))
            receipt=dict(at=time.time(),applied=args.apply,archives=rows,model_interruption=False,logs_or_trajectories_deleted=False)
            (dest/'RECEIPT.json').write_text(json.dumps(receipt,indent=2))
        print(json.dumps(dict(at=time.time(),applied=args.apply,archives=rows,total_bytes=sum(r['bytes'] for r in rows))))

if __name__=='__main__': main()
