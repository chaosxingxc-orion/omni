"""Recorded transition between owned profiles, retaining exact originals for recovery.

No inference or scientific session mutation. A failed apply is never auto-replayed.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import switch as common


def commands(profile, smi=None, release_status=False):
    nic=next(x['ifname'] for x in json.loads(common.call(['ip','-j','-4','addr','show','scope','global'])) if x.get('addr_info'))
    env={**os.environ,'TREE':str(common.D/'publish/omni'),'CPUSET':'32-47','PORT_CONTAINER_USER':'0:0','PORT_IMAGE':common.IMAGE}
    args=['bash',str(common.D/'work/scripts/port_py.sh'),str(common.P/'prepare_commands.py'),'--nic',nic,'--profile',profile]
    if smi:args+=['--smi',str(smi)]
    if release_status:args+=['--release-status']
    return json.loads(subprocess.check_output(args,env=env,text=True))


def wait_card_release(profile, root, timeout=600, interval=10):
    """Bounded host-side wait only for delayed HBM release after owned stop.

    A process, bad health, or missing card refuses immediately. No model polling,
    no retry of container creation or requests. Preserve every observed table.
    """
    end=time.monotonic()+timeout
    index=0
    while True:
        smi=root/f'cards-release-{index:03d}.txt'
        if smi.exists():raise ValueError('release observation already exists')
        smi.write_text(common.call(['npu-smi','info']))
        observed=commands(profile,smi,release_status=True)
        if observed['ready']:return
        if not observed['memory_release_only']:raise ValueError('cards not free: '+json.dumps(observed))
        if time.monotonic()>=end:raise TimeoutError('owned stop memory release timed out')
        time.sleep(interval);index+=1


def validate(doc, candidate, expected_id=None):
    if not doc or doc['Name']!='/'+candidate['name']:raise ValueError('container name differs')
    if doc['Image']!=common.IMAGE:raise ValueError('image differs')
    if doc['Config'].get('Labels',{}).get(common.LABEL)!=candidate['profile_id']:raise ValueError('foreign profile')
    if not doc['State']['Running'] or doc['RestartCount']:raise ValueError('server stopped or restarted')
    if doc['HostConfig']['RestartPolicy']['Name'] not in ('no',''):raise ValueError('automatic restart')
    if expected_id is not None and doc['Id']!=expected_id:raise ValueError('container identity differs')


def validate_stopped_archive(doc, candidate, expected_id, archive_name):
    if not doc or doc['Name'] != '/' + archive_name or doc['Id'] != expected_id:
        raise ValueError('archived original identity differs')
    if doc['Image'] != common.IMAGE or doc['Config'].get('Labels', {}).get(common.LABEL) != candidate['profile_id']:
        raise ValueError('archived original ownership differs')
    if doc['State']['Running'] or doc['RestartCount']:
        raise ValueError('archived original running or restarted')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['plan','apply','resume-after-stop']);p.add_argument('--record',type=Path,required=True)
    p.add_argument('--source');p.add_argument('--target');p.add_argument('--recreate',action='store_true');a=p.parse_args()
    root=a.record.resolve()
    if not root.is_relative_to(common.RECORDS.resolve()) or not common.re.fullmatch('[A-Za-z0-9_-]+',root.name):raise ValueError('owned record path required')
    if a.mode=='plan':
        if not a.source or not a.target or (a.source==a.target and not a.recreate) or root.exists():raise ValueError('fresh distinct profile plan required')
        before,after=commands(a.source),commands(a.target)
        if any(c['host']!=socket.gethostname() for c in before+after):raise ValueError('wrong host')
        if sorted(c for i in before for c in i['cards'])!=list(range(8)) or sorted(c for i in after for c in i['cards'])!=list(range(8)):raise ValueError('all-eight-card topology required')
        originals=[]
        for candidate in before:
            doc=common.inspect(candidate['name']);validate(doc,candidate);originals.append(doc)
        old_names={c['name']:c['name']+'-before-'+root.name for c in before}
        for name in old_names.values():
            if common.inspect(name):raise ValueError('archive name occupied')
        source_names={c['name'] for c in before}
        for candidate in after:
            if candidate['name'] not in source_names and common.inspect(candidate['name']):raise ValueError('target occupied')
        root.mkdir(parents=True)
        common.write(root/'original-docker-inspect.json',originals)
        common.write(root/'plan.json',{'host':socket.gethostname(),'source':a.source,'target':a.target,'before':before,'after':after,'original_ids':{c['name']:d['Id'] for c,d in zip(before,originals)},'old_names':old_names,'model_calls':0})
        print(json.dumps({'planned':str(root),'model_calls':0}));return
    plan=common.read(root/'plan.json')
    if plan['host']!=socket.gethostname():raise ValueError('wrong host')
    with (common.D/'locks/generation.guard').open('r+') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (common.D/'locks/flight.lock').exists():raise ValueError('scientific lease present')
        if a.mode == 'resume-after-stop':
            # Explicit recovery only before any target creation; never replay a model job.
            if not (root/'apply.intent.json').exists():raise ValueError('no stopped transition')
            for name in ('apply.complete.json','candidate-container-ids.json','resume-after-stop.intent.json'):
                if (root/name).exists():raise ValueError('creation already attempted or complete; inspect exact state')
            if commands(plan['target']) != plan['after']:raise ValueError('target profile changed')
            for c in plan['before']:
                validate_stopped_archive(common.inspect(plan['old_names'][c['name']]), c,
                                         plan['original_ids'][c['name']], plan['old_names'][c['name']])
            for c in plan['after']:
                if common.inspect(c['name']):raise ValueError('target occupied')
            smi=root/'cards-before-resume.txt'
            if smi.exists():raise ValueError('recovery already inspected; preserve evidence')
            smi.write_text(common.call(['npu-smi','info']))
            commands(plan['target'],smi)
            common.write(root/'resume-after-stop.intent.json',{'time':time.time(),'pid':os.getpid(),'model_calls':0})
            ids=common.parallel(lambda c:common.call(c['argv']),plan['after'])
            common.write(root/'candidate-container-ids.json',dict(zip([c['name'] for c in plan['after']],ids)))
            common.wait_ready(plan['after'])
            common.write(root/'apply.complete.json',{'time':time.time(),'containers':[common.inspect(c['name']) for c in plan['after']],'model_calls':0,'scientific_records_unchanged':True,'recovered_after_stop':True})
            print(json.dumps({'ready':True,'profile':plan['target'],'model_calls':0}));return
        if (root/'apply.intent.json').exists():raise ValueError('existing attempt: inspect; no blind replay')
        if commands(plan['source'])!=plan['before'] or commands(plan['target'])!=plan['after']:raise ValueError('profile changed')
        for c in plan['before']:
            validate(common.inspect(c['name']),c,plan['original_ids'][c['name']])
            if common.inspect(plan['old_names'][c['name']]):raise ValueError('archive occupied')
        source_names={c['name'] for c in plan['before']}
        for c in plan['after']:
            if c['name'] not in source_names and common.inspect(c['name']):raise ValueError('target occupied')
        common.quiescent(plan['before'])
        common.write(root/'apply.intent.json',{'time':time.time(),'pid':os.getpid()})
        common.parallel(lambda c:common.call(['docker','stop','--time','60',c['name']]),plan['before'])
        for c in plan['before']:common.call(['docker','rename',c['name'],plan['old_names'][c['name']]])
        wait_card_release(plan['target'],root)
        ids=common.parallel(lambda c:common.call(c['argv']),plan['after'])
        common.write(root/'candidate-container-ids.json',dict(zip([c['name'] for c in plan['after']],ids)))
        common.wait_ready(plan['after'])
        common.write(root/'apply.complete.json',{'time':time.time(),'containers':[common.inspect(c['name']) for c in plan['after']],'model_calls':0,'scientific_records_unchanged':True})
    print(json.dumps({'ready':True,'profile':plan['target'],'model_calls':0}))


if __name__=='__main__':main()
