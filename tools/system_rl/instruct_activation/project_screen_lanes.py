"""Project a measured host-local packet onto idle lanes, changing only deployment identity."""
import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
from decode_lineage import snapshot, digest
def read(p): return json.loads(Path(p).read_bytes())
def save(p,v): p.write_text(json.dumps(v,indent=2)+'\n')
def project(profile,lock):
    parent,target=map(Path,(profile['parent'],profile['target']))
    release=read(lock)
    qualified='review' in release
    if qualified:
        from io_utils import digest as runtime_digest
        expected=release['review']
        baseline=snapshot(parent)
        assert expected['within_target'] and runtime_digest(baseline)==expected['runtime']
        proof_path=Path(profile['training_proof'])
        assert digest(proof_path)==release['training']['sha256']
        proof=read(proof_path)
        assert proof['passed'] and proof['runtime']==expected['runtime'] and proof['natural_round']==expected['round']
    else:
        baseline=release['runtime']
    source=parent/('lane-'+profile['source_lane'])
    before=read(source/'config.json'); oldbundle=Path(before['policy_overlay']['bundle'])
    if not qualified: assert oldbundle==parent/'bundle'
    source_bundle=parent/'bundle'
    assert subprocess.check_output(['findmnt','-T',str(target.parent),'-no','FSTYPE'],text=True).strip() in ('xfs','ext4','btrfs')
    usage=os.statvfs(target.parent)
    assert usage.f_bavail*usage.f_frsize>=max(20*1024**3,usage.f_blocks*usage.f_frsize*.1)+int(profile.get('storage_budget_bytes',0)) and usage.f_favail/usage.f_files>=.05
    mode=profile.get('adapter_updates','fixed')
    if mode not in ('fixed','parent_inbox'):raise ValueError('invalid adapter update mode')
    inbox=before['policy_overlay'].get('update_mount')
    if mode=='parent_inbox':
        server=read(oldbundle/('SERVER-'+profile['source_lane']+'.json'))
        assert server['release_kind']=='formal' and server['update_inbox']==inbox and not server.get('successor_ref')
        update_path=Path(inbox);assert update_path.is_dir() and update_path.resolve()==update_path
        assert subprocess.check_output(['findmnt','-T',inbox,'-no','FSTYPE'],text=True).strip() in ('xfs','ext4','btrfs')
    target.mkdir(mode=0o700)
    bundle=target/'bundle'
    shutil.copytree(source_bundle,bundle,ignore=shutil.ignore_patterns('__pycache__'))
    loc=bundle/'LOCATIONS.json'
    def move(path):
        p=Path(path)
        return str(bundle/p.relative_to(oldbundle)) if p.is_relative_to(oldbundle) else path
    locations=read(loc)
    for ref in locations: ref['path']=move(ref['path'])
    save(loc,locations)
    serv=read(source_bundle/('SERVER-'+profile['source_lane']+'.json'))
    def relink(v):
        if isinstance(v,list): return [relink(x) for x in v]
        if not isinstance(v,dict): return v
        obj={k:relink(x) for k,x in v.items()}
        if 'path' in obj:
            obj['path']=move(obj['path'])
            if obj['path']==str(loc): obj.update(bytes=loc.stat().st_size,sha256=digest(loc))
        return obj
    for name,lane in profile['lanes'].items():
        dest=target/('lane-'+name); dest.mkdir()
        cfg={**before,**lane,'root':str(dest)}
        if 'host_template' in profile:
            local=read(Path(profile['host_template'])/('lane-'+name)/'config.json')
            assert qualified and all(local[k]==cfg[k] for k in ('cards','image','model','port','host','served_model_name'))
            assert local['host_boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            for key in ('host_boot_id','guard','readonly_root','socket_ifname'):
                cfg[key]=local[key]
        cfg['policy_overlay']={**before['policy_overlay'],'bundle':str(bundle),'server':str(bundle/('SERVER-'+name+'.json'))}
        if mode=='fixed':cfg['policy_overlay'].pop('update_mount',None)
        server=relink(serv)
        if 'host_template' in profile:
            activation=server['local_activation']
            assert (activation['source_boot'],activation['guard'],activation['model'])==(
                before['host_boot_id'],before['guard'],before['model'])
            activation.update(source_boot=cfg['host_boot_id'],guard=cfg['guard'],model=cfg['model'])
        if mode=='fixed':server.pop('update_inbox',None)
        server.update(endpoint=f"http://{cfg['host']}:{cfg['port']}/v1/chat/completions",service_identity=dict(container_name=cfg['name']+'-'+profile['attempt'],physical_devices=cfg['cards'],image=cfg['image']))
        save(Path(cfg['policy_overlay']['server']),server)
        for py in ('service.py','launch.py'): shutil.copyfile(source/py,dest/py)
        save(dest/'config.json',cfg); save(dest/'PINS.json',{p.name:digest(p) for p in dest.iterdir()})
    assert snapshot(target)==baseline,'inference source/engine/policy differs from locked v2'
    save(target/'DEPLOY.json',dict(profile=profile,lock_sha256=digest(lock)))
if __name__=='__main__':project(read(sys.argv[1]),Path(sys.argv[2]))
