"""Compare immutable publication on the real transport, outside the service inbox."""
import argparse, hashlib, json, shutil, sys, time
from pathlib import Path
from prepare_recording_macros import read,seal
from prepare_transition_config import bind_inbox,pin


def prepare(parent,source,out,mode):
    spec=read(pin(parent));runner=read(spec['runner_ref'])
    if 'actor_reader_release_ref' in spec and 'idle_recovery_ref' not in spec:
        raise ValueError('switch needs explicit predecessor client')
    out.mkdir(parents=True,exist_ok=True)
    prior=spec['sources']['transition_controller'];dest=out/'transition_controller.py'
    if pin(prior['path'])!=prior or prior['sha256']!=read(pin(source/'DISPATCH-COMPATIBILITY.json'))['inbox_transition']:
        raise ValueError('unexpected predecessor')
    if dest.exists() and dest.read_bytes()!=(source/dest.name).read_bytes():raise ValueError('transition drift')
    if not dest.exists():shutil.copyfile(source/dest.name,dest)
    spec['sources']['transition_controller']=pin(dest)
    bind_inbox(spec,source,out,Path(runner['release'])/'tree/parents/jev-system-rl/implementation',mode)
    return seal(out/'TRANSITION.json',spec)


def replay(config,step):
    from contextlib import contextmanager
    from unittest.mock import patch
    import subprocess,tempfile
    from types import SimpleNamespace
    spec=read(pin(config));run=Path(read(spec['iteration_ref'])['output_root'])
    sys.path.insert(0,str(config.parent))
    from transition_controller import main
    old=sys.argv
    try:
        sys.argv=['check',str(config),'--check'];main()
    finally:sys.argv=old
    import formal_parity as pub,formal_rollout
    from dependencies import load,ROOT
    legacy=load(ROOT/'jev-system-rl/implementation/formal_parity.py','original_inbox_wire')
    guard=read(pin(run/'resident-lease/REMOTE-LEASE.json'));remote=guard['remote']
    folder=run/f'policy-{step:06d}';ck=folder/'adapter-input';targetref=read(pin(folder/'a/ACTIVATION.json'))['policy_ref']
    original_marker=read(pin(folder/'MARKER.json'));prior=read(pin(folder/'a/SWITCHED.json'))['previous_policy_ref']
    original_inbox=Path(original_marker['marker']).parent
    expected={str(f.relative_to(ck)):pin(f)['sha256'] for f in ck.rglob('*') if f.is_file()}
    if len(expected)!=original_marker['installed'] or expected['candidate/POLICY.json']!=targetref['sha256']:
        raise ValueError('not the same publication')
    root=original_inbox.parent
    with formal_rollout.provider_modules() as (lease,*_):
        def remote_py(code,*args):
            return lease.command(remote,['python3','-B','-c',code,*map(str,args)],capture_output=True,text=True).stdout.strip()
        scratch=Path(remote_py(
             "import sys,tempfile;from pathlib import Path;sys.path.insert(0,sys.argv[2]);from local_runtime_storage import check_paths;p=Path(sys.argv[1]);check_paths({'root':p},min_free_bytes=(20<<30)+int(sys.argv[3]));print(tempfile.mkdtemp(prefix='policy-wire-replay-',dir=p))",
             root,read(remote['config'])['hot_root'],3*sum(f.stat().st_size for f in ck.rglob('*') if f.is_file())))
        if scratch.parent!=root or not scratch.name.startswith('policy-wire-replay-'):raise ValueError('bad scratch')
        session=dict(remote=remote,root=run/'resident-lease',local=SimpleNamespace(check_local=lambda:None))
        # No marker in this private sibling is watched; pin the actual policy from the original install.
        before=json.loads(remote_py("import json,sys;from pathlib import Path;print(Path(sys.argv[1]).read_text())",original_marker['marker']))
        if before!={'previous':prior,'successor':targetref}:raise ValueError('original marker differs')
        arms=[]
        for name,method in [('legacy',legacy.publish_inbox),('binary',pub.publish_inbox)]:
            inbox=scratch/name
            remote_py("from pathlib import Path;import sys;Path(sys.argv[1]).mkdir()",inbox)
            target=inbox/'policy';size=[]
            def command(*args,**kw):
                wire=kw['input'];size.append(len(wire))  # legacy json.dumps uses ensure_ascii=True
                return lease.command(*args,**kw)
            @contextmanager
            def scoped():yield (SimpleNamespace(command=command),None,None,None)
            with patch.object(formal_rollout,'provider_modules',scoped):
                t=time.monotonic()
                marker=method(remote,dict(policy_overlay=dict(update_mount=str(inbox))),ck,target,prior,targetref,step-1,session=session)
                elapsed=time.monotonic()-t
            got=json.loads(remote_py("from pathlib import Path;import json,hashlib,sys;p=Path(sys.argv[1]);print(json.dumps([json.loads(Path(sys.argv[2]).read_bytes()),{str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file()}]))",target,marker['marker']))
            if got!=[before,expected] or marker['installed']!=len(expected):raise ValueError('returned bytes differ')
            arms.append(dict(mode=name,seconds=elapsed,wire_bytes=size[0]))
        # Corrupt an isolated small packet before remote receipt, exercising the actual reader.
        with tempfile.TemporaryDirectory(dir=config.parent) as d:
            (Path(d)/'data').write_bytes(b'test member')
            inbox=scratch/'negative';remote_py("from pathlib import Path;import sys;Path(sys.argv[1]).mkdir()",inbox)
            def damaged(*a,**kw):
                kw['input']=kw['input'][:-1]+b'!'
                return lease.command(*a,**kw)
            @contextmanager
            def invalid():yield (SimpleNamespace(command=damaged),None,None,None)
            with patch.object(formal_rollout,'provider_modules',invalid):
                try:pub.publish_inbox(remote,dict(policy_overlay=dict(update_mount=str(inbox))),Path(d),inbox/'policy',prior,targetref,step-1,session=session)
                except subprocess.CalledProcessError as exc:
                    if b'bad wire member' not in exc.stderr:raise
                else:raise ValueError('wrong bytes accepted')
            if remote_py("import sys;from pathlib import Path;print(list(Path(sys.argv[1]).iterdir()))",inbox)!='[]':raise ValueError('invalid wire was published')
        if arms[1]['seconds']>=arms[0]['seconds'] or arms[1]['wire_bytes']>=arms[0]['wire_bytes']:
            raise ValueError('no measured benefit: '+str(arms))
        remote_py("import sys,shutil;from pathlib import Path;p=Path(sys.argv[1]);assert p.resolve()==p and p.parent==Path(sys.argv[2]) and p.name.startswith('policy-wire-replay-') and set(x.name for x in p.iterdir())=={'binary','legacy','negative'};shutil.rmtree(p)",scratch,root)
    return seal(config.parent/'ACCEPTANCE.json',dict(passed=True,config_sha256=pin(config)['sha256'],step=step,
        files=len(expected),all_digests_equal=True,original_marker_equal=True,corrupt_rejected=True,arms=arms))


def live(config,cohort,out):
    import math, shlex, subprocess
    from check_lane_storage import live as lane_live
    spec=read(pin(config));run=Path(read(spec['iteration_ref'])['output_root'])
    probe=out.with_name(out.stem+'-dispatch.json')
    settled=read(lane_live(config,cohort,probe));step=settled['policy_step']
    d=run/f'policy-{step:06d}';local=d/'adapter-input';m=read(pin(d/'MARKER.json'))
    candidate=read(pin(local/'candidate/POLICY.json'));policy=read(pin(d/'POLICY.json'))
    closed=read(pin(d/'ACTIVATIONS-CLOSED.json'));lanes=read(closed['activation_ref'])
    ret=read(pin(run/f'step-{step:06d}/update/returned/RETURN.json'))
    if (m['wire_format']!=spec['inbox_wire_format'] or spec['inbox_wire_format']!='raw_members_v1'
        or not closed['terminal_verified'] or closed['unknown_requests']!=0 or lanes['checkpoint_ref']!=ret['checkpoint_ref']
        or candidate['step']!=step or policy['step']!=step or candidate['artifact_version']!=policy['artifact_version']
        or any(not math.isfinite(m[k]) or m[k]<0 for k in ('prepare_seconds','transfer_and_commit_seconds'))):
        raise ValueError('production transfer or predecessor mismatch')
    files={str(f.relative_to(local)):pin(f)['sha256'] for f in local.rglob('*') if f.is_file()}
    size=sum((local/k).stat().st_size for k in files)
    if not size<m['wire_bytes'] or len(files)!=m['installed']:raise ValueError('invalid member count or size')
    receipts={};change=None
    for lane,ref in lanes['lanes'].items():
        r=read(ref);a=read(r['activation_ref']);c=read(r['cache_ref']);s=read(r['switch_ref']);old=read(r['previous_activation_ref'])
        pair=dict(previous=old['policy_ref'],successor=a['policy_ref'])
        if (set(lanes['lanes'])!=set(settled['lanes']) or r['lane']!=lane or (change and pair!=change) or
            a['policy_ref']['sha256']!=files['candidate/POLICY.json'] or a['policy_version']!=policy['artifact_version']
            or a['weights_ref']['sha256']!=r['adapter_sha256'] or a['weights_ref'] not in candidate['assets']
            or s['previous_policy_ref']!=old['policy_ref'] or s['new_id']!=a['adapter_id'] or s['old_id']!=old['adapter_id']
            or s['old_id']==s['new_id'] or not s['old_removed_from_engine'] or not s['cache_reset']
            or c['policy_ref']!=a['policy_ref'] or not all(c[k] for k in ('kv_cache','prefix_cache','reset_succeeded'))
            or a['fixture'] or len(a['workers'])!=4 or {w['rank'] for w in a['workers']}!=set(range(4))
            or any(w['sha256']!=r['adapter_sha256'] or w['adapter_id']!=s['new_id'] or
                   w['compared_original_keys']!=384 for w in a['workers'])):raise ValueError('invalid service switch')
        change=pair;receipts[lane]=dict(adapter_id=s['new_id'],readback_keys=[w['compared_original_keys'] for w in a['workers']])
    if not change:raise ValueError('missing service receipts')
    # Read-only remote verification, keyed by the physical service policy; do not touch the inbox.
    remote_program="""import sys,json,hashlib
from pathlib import Path
marker=Path(sys.argv[1])
v=json.loads(marker.read_bytes());d=Path(v['successor']['path']).parents[1]
if d.parent!=marker.parent or d.resolve()!=d:raise ValueError('unexpected publication root')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
print(json.dumps([v,{str(p.relative_to(d)):digest(p) for p in d.rglob('*') if p.is_file()}]))
"""
    target=read(spec['runner_ref'])['rollout_target']
    got=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','--',target,
                   shlex.join(['python3','-B','-',m['marker']])],input=remote_program.encode()))
    if got!=[change,files]:raise ValueError('actual service inbox differs from checkout')
    return seal(out,dict(passed=True,step=step,dispatch=pin(probe),services=receipts,marker=m,
                         all_digests_equal=True,member_bytes=size,policy_version=policy['artifact_version']))


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    for k,fields in [('prepare',('parent','source','out')),('replay',('config',)),('live',('config','cohort','out'))]:
        a=sub.add_parser(k)
        for f in fields:a.add_argument('--'+f,type=Path,required=True)
        if k=='prepare':a.add_argument('--mode',required=True)
        elif k=='replay':a.add_argument('--step',type=int,required=True)
    args=vars(p.parse_args());fn={'prepare':prepare,'replay':replay,'live':live}[args.pop('action')]
    print(json.dumps(fn(**args)))
