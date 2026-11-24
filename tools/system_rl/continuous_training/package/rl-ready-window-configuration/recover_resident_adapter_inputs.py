"""Complete a rejected preflight's missing local adapter members, then launch the reserved actor once."""
import argparse,fcntl,hashlib,json,os,re,subprocess,sys,tempfile
from pathlib import Path


def recover(target, expected, boot):
    read=lambda f:json.loads(f.read_bytes())
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    if target!=target.resolve() or Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=boot:
        raise ValueError('changed recovery path/boot')
    with (target/'adapter-preflight.guard').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg=read(target/'TRAIN.local.json');hot=Path(cfg['hot']);cp=Path(cfg['restore_checkpoint']['directory'])
        if (cfg['boot_id']!=boot or sha(cp/'COMMITTED.json')!=expected or cfg['restore_checkpoint']['sha256']!=expected
                or target.name!=cfg['name'] or hot!=target/'hot' or cp.parents[4].parent!=target.parent
                or any((hot/n).exists() for n in ('INTENT.json','CONTAINER.json','EXIT.json'))
                or (target/'RECOVERED-LAUNCH.json').exists()):
            raise ValueError('not a staged, unlaunched resident checkpoint')
        check=subprocess.run(['docker','container','inspect',cfg['name']],capture_output=True,text=True)
        if check.returncode!=1 or not any(s in check.stderr for s in ('No such object','No such container')):raise ValueError('container identity is not free')
        m=cfg['capture_mounts'];logical=Path(m['logical_roots']['closed_actor_command'])
        projected=Path(m['resident_input_root'])/'closed_actor_command'
        if logical!=cp.parent or projected.parent.parent!=target.parent or projected!=projected.resolve():
            raise ValueError('different closed command')
        added={}
        for row in read(cp/'COMMITTED.json')['ranks']:
            for n,h in row['files'].items():
                if not n.startswith('lora_adapter/'):continue
                if len(Path(n).parts)!=2 or '..' in Path(n).parts:raise ValueError('unsafe checkpoint member')
                src=cp/n; dst=projected/cp.name/n
                if src.resolve()!=src or dst.resolve()!=dst or sha(src)!=h:raise ValueError('unsealed adapter member')
                if not dst.exists():
                    os.link(src,dst)
                    fd=os.open(dst.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
                    added[n]=h
                if sha(dst)!=h:raise ValueError('conflicting projected adapter')
        sys.path.insert(0,str(target))
        from importlib.util import spec_from_file_location,module_from_spec
        spec=spec_from_file_location('guarded_launcher',target/'launch_retained_hf_fsdp.py')
        launcher=module_from_spec(spec);spec.loader.exec_module(launcher)
        receipt=target/'ADAPTER-INPUT-SUPPLEMENT.json'
        if receipt.exists():
            if read(receipt)['checkpoint_sha256']!=expected:raise ValueError('different supplement')
        else:launcher.save(receipt,dict(checkpoint_sha256=expected,projected=added))
        cmd=[sys.executable,'-B',str(target/'launch_retained_hf_fsdp.py'),'--config',str(target/'TRAIN.local.json')]
        previous=target/'capture-preflight'
        if not (target/'CAPTURE-PREFLIGHT.json').exists() and previous.is_dir() and not previous.is_symlink():
            previous.rename(target/('capture-preflight.failed-'+str(previous.stat().st_ino)))
        if not (target/'CAPTURE-PREFLIGHT.json').exists():
            subprocess.run(cmd+['--check-capture'],cwd=target,check=True)
        launcher.save(target/'RECOVERED-LAUNCH.json',dict(checkpoint_sha256=expected,preflight_sha256=sha(target/'CAPTURE-PREFLIGHT.json')))
        with (target/'launch.log').open('ab') as log:
            p=subprocess.Popen(cmd,cwd=target,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=p.pid,added=added)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('target',type=Path);p.add_argument('expected');p.add_argument('boot')
    recover(**vars(p.parse_args()))
