"""Owner-local command submission. A retry can only observe identical bytes."""
import argparse,hashlib,json,os,sys,tempfile
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def submit(target,run_id,boot,command):
    hot=target/'hot';exchange=hot/'resident'
    if (target!=target.resolve() or not target.is_absolute() or
            boot!=Path('/proc/sys/kernel/random/boot_id').read_text().strip()):
        raise ValueError('resident actor host or target differs')
    cfg=json.loads((target/'TRAIN.local.json').read_bytes())
    ready=json.loads((exchange/'READY.json').read_bytes())
    intent=json.loads((hot/'INTENT.json').read_bytes())
    if (cfg['continuous_actor']['run_id']!=run_id or ready['run_id']!=run_id or
            command['run_id']!=run_id or boot!=ready['source_boot'] or boot!=cfg['boot_id'] or
            intent['config_sha256']!=sha(target/'TRAIN.local.json') or intent['boot_id']!=boot):
        raise ValueError('command not bound to the launched actor')
    seq=command['sequence']
    if type(seq)!=int or not 0<seq<cfg['continuous_actor']['max_commands']:
        raise ValueError('command sequence outside resident owner')
    parent=exchange/'results'/f'{seq-1:06d}'/'CLOSED.json'
    if sha(parent)!=command['previous_close_sha256']:raise ValueError('incorrect preceding window')
    folder=exchange/'commands';folder.mkdir(exist_ok=True)
    p=folder/f'{seq:06d}.json'
    if not p.exists():
        if (hot/'EXIT.json').exists() or (exchange/'END.json').exists():
            raise ValueError('cannot submit to terminated resident owner')
        fd,tmp=tempfile.mkstemp(dir=folder)
        try:
            with os.fdopen(fd,'w') as f:
                json.dump(command,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
            try:os.link(tmp,p)
            except FileExistsError:pass
        finally:os.unlink(tmp)
        for d in (folder,exchange):
            fd=os.open(d,os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
    if json.loads(p.read_bytes())!=command:raise ValueError('resident command sequence already reserved')
    return dict(run_id=run_id,source_boot=boot,sequence=seq,sha256=sha(p))

if __name__=='__main__':
    target,run_id,boot=sys.argv[1:]
    print(json.dumps(submit(Path(target),run_id,boot,json.load(sys.stdin))))
