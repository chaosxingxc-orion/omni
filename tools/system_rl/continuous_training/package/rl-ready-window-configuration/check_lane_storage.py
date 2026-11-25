"""Replay immutable packet publication in the control image; no HTTP, lease or registry writes."""
import argparse, hashlib, json, os, shutil, sys, tempfile, time
from pathlib import Path
from prepare_transition_config import pin, validate_lane_storage
from prepare_recording_macros import read, seal


def prepare(parent,source,out,profile):
    spec=read(pin(parent));runner=read(spec['runner_ref'])
    validate_lane_storage(profile,Path(runner['release'])/'tree/parents')
    if 'actor_reader_release_ref' in spec and 'idle_recovery_ref' not in spec:
        raise ValueError('actor client is relative to transition entry: explicit predecessor required')
    prior=read(pin(source/'DISPATCH-COMPATIBILITY.json'))['predecessors']
    out.mkdir(parents=True,exist_ok=True)
    for key in ('capture_stage','formal_rollout','transition_controller'):
        old=spec['sources'][key];path=source/('source' if key!='transition_controller' else '')/(key+'.py')
        if pin(old['path'])!=old or old['sha256'] not in prior[key]:
            raise ValueError('unqualified previous module '+key)
        dest=out/path.name
        if dest.exists() and pin(dest)['sha256']!=pin(path)['sha256']:raise ValueError('different staged module')
        if not dest.exists():shutil.copyfile(path,dest)
        spec['sources'][key]=pin(dest)
    return seal(out/'TRANSITION.json',dict(spec,lane_storage=profile))


def replay(config,inventory):
    spec=read(pin(config));sys.path.insert(0,str(config.parent))
    from transition_controller import main
    argv=sys.argv
    try:
        sys.argv=['check',str(config),'--check'];main()
    finally:sys.argv=argv
    import capture_stage
    from fresh_adapter import scope
    import subprocess
    from unittest.mock import patch
    inv=read(pin(inventory));closed=read(inv['generation_closed_ref'])
    if not inv['closed'] or not closed['groups']:raise ValueError('no closed generation')
    cohorts=Path(inv['cohort_root'])
    refs={k:v for k,v in inv['files'].items() if k.endswith('/SUPPLY.json')}
    requests=[];budget=read(inv['files']['INPUT.json'])['capture']['runtime_storage_budget_bytes']
    for key in sorted(refs):
        ep=Path(key).parent; files=list((cohorts/ep).glob('*.packet.json'))
        p,=files;j=inv['files'][str(ep/'native-journal'/(next((cohorts/ep/'native-journal').glob('*.dispatch.json')).name))]
        packet=read(inv['files'][str(p.relative_to(cohorts))])
        if read(j)['request_id']!=packet['request_id']:raise ValueError('different dispatched packet')
        requests.append((ep,p.name,read(inv['files'][str(ep/'LANE.json')]),packet,read(refs[key])))
    if not requests:raise ValueError('empty replay')
    with scope(),tempfile.TemporaryDirectory(dir=config.parent) as tmp:
        from cohort_io import ACL,imports,save,storage
        with imports([ACL]):import sync_runtime as rt
        original=rt.capacity
        def arm(label,mode):
            root=Path(tmp)/label;root.mkdir();t=time.monotonic();probes=0
            call=subprocess.check_output
            def count(cmd,*a,**kw):
                nonlocal probes
                if cmd[0]=='findmnt':probes+=1
                return call(cmd,*a,**kw)
            pins={}
            with patch.object(subprocess,'check_output',count),capture_stage.lane_capacity(root,rt,mode) as stats:
                for ep,name,lane,packet,supply in requests:
                    d=root/ep;d.mkdir(parents=True)
                    for k,val in [('LANE.json',lane),(name,packet),('SUPPLY.json',supply)]:
                        if k==name:
                            storage(root,budget);storage(root,budget)
                        ref=save(d/k,val);pins[str(ep/k)]=(ref['sha256'],ref['bytes'])
            sec=time.monotonic()-t
            if rt.capacity is not original:raise ValueError('validator not restored')
            return dict(seconds=sec,findmnt=probes,live=stats),pins
        before,one=arm('before',None);after,two=arm('after',spec['lane_storage'])
        if one!=two or after['findmnt']!=1 or after['live']['live_checks']!=before['findmnt'] or after['seconds']>=before['seconds']:
            raise ValueError('incomplete or slower capacity replay: '+str((before,after)))
        from types import SimpleNamespace
        with capture_stage.lane_capacity(tmp,rt,spec['lane_storage']) as stats:
            last=Path(tmp)/'after'/requests[-1][0]/'SUPPLY.json'
            for run in (lambda:storage(tmp,10**24),lambda:storage('/proc'),lambda:save(last,{'wrong':True})):
                try:run()
                except ValueError:pass
                else:raise ValueError('invalid write/mount/budget accepted')
            v=os.statvfs(tmp)
            full=SimpleNamespace(**{n:getattr(v,n) for n in ('f_blocks','f_frsize','f_bavail','f_files')},f_favail=0)
            with patch.object(os,'fstatvfs',lambda fd:full):
                try:storage(tmp)
                except ValueError:pass
                else:raise ValueError('inode threshold ignored')
            if stats['other_mount_checks']!=1:raise ValueError('different mount did not call original')
    return seal(config.parent/'ACCEPTANCE.json',dict(passed=True,config_sha256=pin(config)['sha256'],
        generation=pin(inventory),packets=len(requests),files=len(one),all_bytes_equal=True,
        negative=['oversize','other_mount','immutable_collision','inode_exhaustion'],before=before,after=after))


def live(config,cohort,out):
    from report_batch_stages import dispatch_preparation
    from report_policy_dispatch_gap import measure
    spec=read(pin(config));root=Path(read(spec['iteration_ref'])['output_root'])
    cohort=cohort.resolve(strict=True)
    if cohort.parent.parent!=root/'rollout':raise ValueError('different run')
    gen=read(pin(cohort.parent/'GENERATION-RESULT.json'));inv=read(gen['generation_inventory_ref'])
    closed=read(gen['generation_closed_ref']);capture=read(gen['input_ref'])
    drained=read(pin(cohort.parent/'DRAINED.json'))
    if (not inv['closed'] or inv['cohort_root']!=str(cohort) or gen['input_ref']!=inv['files']['INPUT.json']
            or not drained['terminal_verified'] or drained['unknown_requests']!=0 or drained['usage']['unknown']!=0
            or not (drained['generation_closed_ref']==inv['generation_closed_ref']==gen['generation_closed_ref'])
            or capture['capture'].get('lane_storage')!=spec['lane_storage']):raise ValueError('not a settled candidate window')
    policy=read(capture['policy']['current_ref']);step=policy['step']
    gap=measure(root,step,cohort)
    lanes=dispatch_preparation(cohort,(root/f'policy-{step:06d}/QUALIFIED-INPUT.json').stat().st_mtime_ns,
                               gap['marks_ns']['first_dispatch'])
    stats={}
    for lane in capture['capture']['lanes']:
        f,=(cohort.parent/'lease'/lane).glob('CAPACITY-*.json');s=read(pin(f))
        if (s['mount_probes']!=1 or s['live_checks']<lanes['lanes'][lane]['requests'] or s['lane']!=lane or
            s['config']!=pin(cohort.parent/'CAPTURE.json') or s['source']!=spec['sources']['capture_stage']):
            raise ValueError('wrong lane receipts')
        stats[lane]={k:s[k] for k in ('mount_probes','live_checks','other_mount_checks')}
    sent={};received={}
    for ext,table in [('dispatch',sent),('response',received)]:
        for name,ref in inv['files'].items():
            if not name.endswith('.'+ext+'.json'):continue
            row=read(ref);k=row['request_id']
            if k in table:raise ValueError('duplicate per-lane request')
            table[k]=(row['request_sha256'],row['intent_sha256'],row['identity'])
    if len(sent)!=sum(v['requests'] for v in lanes['lanes'].values()) or sent!=received:raise ValueError('unsettled request')
    return seal(out,dict(policy_step=step,recordings=len(closed['groups']),requests=len(sent),unknown=0,
        gap_seconds=gap['elapsed_s'],gap_stages=gap['seconds'],lanes=lanes['lanes'],checks=stats,
        config=pin(config),generation=gen['generation_inventory_ref']))


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='action',required=True)
    for action,fields in [('prepare',('parent','source','out','profile')),('replay',('config','inventory')),
                          ('live',('config','cohort','out'))]:
        cmd=s.add_parser(action)
        for f in fields:cmd.add_argument('--'+f,type=Path,required=True)
    a=vars(p.parse_args());act=a.pop('action')
    if act=='prepare':a['profile']=read(pin(a['profile']))
    print(json.dumps({'prepare':prepare,'replay':replay,'live':live}[act](**a)))
