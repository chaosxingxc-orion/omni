"""Stage the experiment projections on the rollout host; all cutovers use its original guard."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

from build import pin, save
from paired_experiment import digest


def provision(site, baseline, root, exp, host, sources):
    here=Path(__file__).resolve().parent
    registry=Path(site['service_registry'])
    record=json.loads(registry.read_bytes())
    if record['host']!=host: raise ValueError('service registry and runner host differ')
    prev=Path(record['remote_root'])
    dest=Path(site['remote_run_root'])/root.name
    def remote(*argv,raw=None):
        return subprocess.check_output(['ssh','-oBatchMode=yes',host,shlex.join(list(map(str,argv)))],input=raw)
    def doc(*parts): return json.loads(remote('cat',Path(*parts)))
    prep=root/'observation-deployment.json'
    def registered():
        new=dict(host=host,remote_root=str(dest),owner=root.name)
        with tempfile.NamedTemporaryFile(mode='w',dir=registry.parent,delete=False) as f:
            json.dump(new,f); f.flush(); os.fsync(f.fileno())
        os.replace(f.name,registry)
        fd=os.open(registry.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    actor=json.loads((root/'ACTOR-START.json').read_bytes())
    def pinned(ref):
        path=Path(ref['path'])
        if pin(path)!=ref:raise ValueError('initial policy file changed: '+str(path))
        return path
    branch=json.loads(pinned(actor['branch_ref']).read_bytes())
    configured=branch['schema']=='base-node-policy-branch-v1'
    if prep.exists():
        dep=json.loads(prep.read_bytes())
        if (dep['remote_root']!=str(dest) or doc(dest,'CONFIG.json')!=dep['config'] or dep['config']['experiment_sha256']!=digest(exp)
                or dep['config']['version']!=actor['version'] or (configured and dep['config'].get('initial_branch_sha256')!=actor['branch_ref']['sha256'])):
            raise ValueError('staged deployment differs')
        if doc(dest,'READY.json').get('experiment_sha256')!=digest(exp):
            raise ValueError('staged pair is not ready for this experiment')
        registered()
        return prep
    if not dest.is_absolute() or not prev.is_absolute() or dest==prev:
        raise ValueError('missing independent service namespace')
    seed=Path(baseline['service_seed_root']) if 'service_seed_root' in baseline else prev
    c=doc(seed,'CONFIG.json')
    # The seed supplies the engine and graph. For a configured start only the
    # actor publication supplies weights; do not choose them from a service seed.
    weight_files={}
    if configured:
        policy=json.loads(pinned(branch['policy_ref']).read_bytes())
        pub=json.loads(pinned(policy['publication_ref']).read_bytes())
        if (pub['schema']!='configured-node-initial-policy-v1' or actor['version']!=policy['version'] or
                actor['assets']!={Path(a['path']).name:a for a in pub['assets']}):
            raise ValueError('selected actor and service publication differ')
        weight_files={'source.json':pub['initialization']['source_ref'],'geometry.json':pub['geometry_ref'],
                      'base.json':pub['base_manifest_ref'],'profile.json':pub['profile_ref'],
                      'publication.json':policy['publication_ref'],**actor['assets']}
        c.update(initial_branch_sha256=actor['branch_ref']['sha256'],version=actor['version'],
                 configured_files={k:v['sha256'] for k,v in weight_files.items()})
        for key in ('parent_commit','origin_sha256'):c.pop(key,None)
    elif c['version']!=actor['version'] or 'configured_files' in c:
        raise ValueError('registered service seed and initial actor weights differ')
    prev_attempt=doc(prev,'CONFIG.json')['attempt'];attempt='exp'+digest(dict(run=root.name,experiment=digest(exp)))[:14]
    for k in ('released_session_ref','failed_startup_predecessor','failed_diagnostic_predecessor'):
        c.pop(k,None)
    observed={}
    for lane in exp['lanes']:
        cfg=doc(prev,'lane-'+lane,'config.json'); ready=doc(prev,lane+'-READY.json')
        name=cfg['name']+'-'+prev_attempt
        identity=json.loads(remote('docker','inspect',name))[0]
        if identity['Id']!=ready['container_id']:raise ValueError('different registered predecessor')
        observed[lane]=identity['State']['Status']
        c['old_ids'][lane]=ready['container_id'];c['old_services'][lane]=name
        c['predecessors'][lane]=dict(root=str(prev),attempt=prev_attempt)
    if set(observed.values())=={'exited'}:
        release=record.get('release_path')
        if not release and record.get('owner'):
            primary=next(iter(exp['lanes']))
            release=str(prev/('lane-'+primary)/'formal-data'/(record['owner']+'-session')/'RELEASE.json')
        if not release: raise ValueError('stopped predecessors lack a terminal release locator')
        raw=remote('cat',release)
        import hashlib
        c['released_session_ref']=dict(path=release,sha256=hashlib.sha256(raw).hexdigest())
    elif set(observed.values())!={'running'}:raise ValueError('unreconciled mixed predecessor states')
    c.update(attempt=attempt,check_attempt=attempt+'check',experiment_sha256=digest(exp))
    c['overlays'].update(exp['readers'])
    staged={'inputs/EXPERIMENT.json':root/'package/EXPERIMENT.json',
            **{'inputs/overlays/'+n:here.parent/member for n,member in sources.items()},
            **{n:here/n for n in ('deploy_resident_graph_lanes.py','paired_experiment.py','rank_profile.py')}}
    # Exclusive fresh directory. Partial cutovers must be reconciled, never blindly replayed.
    remote('mkdir','-m','700',dest)
    remote('cp','-a','--reflink=auto',seed/'inputs',dest/'inputs')
    if weight_files:
        area=dest/'inputs'/('configured-'+actor['branch_ref']['sha256'])
        remote('mkdir','-m','700',area)
        for name,ref in weight_files.items():
            with pinned(ref).open('rb') as stream:
                subprocess.run(['ssh','-oBatchMode=yes',host,shlex.join(['tee',str(area/name)])],
                               stdin=stream,stdout=subprocess.DEVNULL,check=True)
    for name,p in staged.items():
        remote('python3','-c','import sys;from pathlib import Path;p=Path(sys.argv[1]);p.write_bytes(sys.stdin.buffer.read())',dest/name,raw=p.read_bytes())
    conf=save(root/'service-config.json',c)
    remote('tee',dest/'CONFIG.json',raw=Path(conf['path']).read_bytes())
    save(prep,dict(remote_root=str(dest),predecessor=str(prev),config=c))
    remote('python3','-B',dest/'deploy_resident_graph_lanes.py',dest)
    r=doc(dest,'READY.json')
    if r.get('experiment_sha256')!=digest(exp) or set(r['lanes'])!=set(exp['lanes']):
        raise ValueError('wrong ready deployment')
    registered()
    return prep
