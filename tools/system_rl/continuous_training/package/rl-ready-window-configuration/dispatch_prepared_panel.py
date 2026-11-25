"""Dispatch one frozen recording panel through the existing collector's native journal.

A lane owns a deterministic slice, and an exclusive host guard outlives every HTTP
call. Intent without an observed response is never sent again. The final index
references the original prepared packets instead of copying their audio again.
"""
import concurrent.futures as cf
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
import urllib.request

from build import pin, save
from prepare_s1_recordings import checked, needed, read

def main(config):
    cfg=read(config);root=Path(cfg['root']);root.mkdir(mode=0o700,parents=True,exist_ok=True)
    sys.path.insert(0,cfg['collector'])
    from common import canonical, digest, Storage
    from direct_transport import DirectHTTP
    from content import jsonl
    from source.panel_receipts import collection_calls
    for k,v in cfg['storage_environment'].items():os.environ[k]=v
    space=Storage(root,cfg['storage'])
    panel=read(checked(cfg['panel']));draws=panel['draws'];n=len(draws)
    needed(n==cfg['count']==len({d['reference_id'] for d in draws}) and
           panel['purpose']=='paired_baseline_screen' and not panel['formal_training_set'],'different panel')
    lanes=read(checked(cfg['lanes']));base=read(checked(cfg['base_manifest']))
    checked(cfg['system_ref'])
    pages=read(checked(panel['sampling']['pages']))
    needed(cfg['system_ref']['sha256']==cfg['system_version'] and
           cfg['frozen_upstream_ref']['sha256']==pages['configuration']['knowledge_release']['sha256'] and
           base['status']=='CLOSED_IMMUTABLE' and not base['fixture'] and base['model']==cfg['model'],
           'screen system, knowledge or base differs')
    needed(len({(r['ssh'],r['port']) for r in lanes})==len(lanes) and
           len({r['id'] for r in lanes})==len(lanes),'overlapping lanes')
    src=Path(cfg['inputs']);signature=pin(config)
    with contextlib.ExitStack() as stack:
        lock=stack.enter_context((root/'dispatch.guard').open('a'))
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(root/'SUBMITTED.json',dict(configuration=signature,panel=cfg['panel'],count=n,
              scope='frozen_system_baseline_screen',lanes=[r['id'] for r in lanes]))
        if (root/'COMPLETE.json').exists():return
        witnesses=[];transports=[];guards={};children=[]
        def close(p):
            if p.stdin:p.stdin.close()
            if p.poll() is None:
                try:p.wait(timeout=3)
                except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=10)
        for r in lanes:
            conf=read(Path(r['local'])/'config.json');guard=conf['guard']
            needed(all(Path(a['path']).is_relative_to(conf['model']) for a in base['assets']) and
                conf['served_model_name']==cfg['model'] and 'policy_overlay' not in conf and
                '--enable-lora' not in conf['serve_args'] and '--enable-prefix-caching' in conf['serve_args']
                and conf['client_chat_template_kwargs']=={'enable_thinking':False},'wrong screen engine')
            key=(r['ssh'],guard)
            if key not in guards:
                cmd="import sys,fcntl; f=open(sys.argv[1],'rb'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); print('HELD',flush=True); sys.stdin.read()"
                proc=subprocess.Popen(['ssh','-T',r['ssh'],shlex.join(['python3','-B','-c',cmd,guard])],
                      stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
                stack.callback(close,proc);needed(proc.stdout.readline().strip()=='HELD','host lease unavailable')
                guards[key]=proc;children.append(proc)
            proc=subprocess.Popen(['ssh','-S','none','-NT','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=30','-L',
                f"127.0.0.1:{r['local_port']}:127.0.0.1:{r['port']}",r['ssh']])
            children.append(proc);stack.callback(close,proc)
            url=f"http://127.0.0.1:{r['local_port']}"
            expected=pin(Path(r['local'])/'config.json')
            cmd=shlex.join(['docker','inspect',r['container']])
            info=json.loads(subprocess.check_output(['ssh',r['ssh'],cmd]))[0]
            needed(info['State']['Running'] and info['Image']==conf['image'] and
                info['Config']['Labels']['omni.activation']==conf['name'] and
                info['HostConfig']['ReadonlyRootfs'] and sorted(x['PathOnHost'] for x in info['HostConfig']['Devices']
                if x['PathOnHost'].removeprefix('/dev/davinci').isdigit())==sorted('/dev/davinci'+str(c) for c in conf['cards']),
                'unexpected service identity')
            launch=json.loads(subprocess.check_output(['ssh',r['ssh'],shlex.join(['docker','exec',info['Id'],'cat','/runtime/state/LAUNCH.json'])]))
            needed(launch['boot_id']==conf['host_boot_id'] and launch['config_sha256']==expected['sha256'] and
                conf['model'] in launch['argv'] and all(m['RW'] is False for m in info['Mounts'] if m['Destination']==conf['readonly_root']),
                'different live config or model mount')
            end=time.monotonic()+cfg['startup_timeout_seconds']
            while True:
                needed(all(p.poll() is None for p in children),'lost tunnel or lease')
                try:
                    with urllib.request.urlopen(url+'/v1/models',timeout=5) as resp: models=json.load(resp)['data']
                    break
                except OSError:
                    if time.monotonic()>end:raise
                    time.sleep(3)
            needed([m['id'] for m in models]==[cfg['model']] and models[0]['root'] in (conf['model'],cfg['model']),'wrong live model')
            witnesses.append(dict(id=r['id'],container=info['Id'],boot=launch['boot_id'],model=cfg['model'],
                service_config=expected,weight_path=conf['model'],adapter=None))
            lane_dir=root/r['id'];lane_dir.mkdir(exist_ok=True)
            ident=dict(run_id=cfg['run_id'],attempt_id=r['attempt'],arm=r['id'],protocol_hash=signature['sha256'],
                       split_identity_hash=cfg['panel']['sha256'])
            transports.append(DirectHTTP(lane_dir,dict(protocol='vllm-chat-completions',endpoint=url+'/v1/chat/completions',
                 package_src=cfg['package_src'],journal_identity=ident,timeout_seconds=cfg['request_timeout_seconds'])))
        service=save(root/'SERVICES.json',dict(lanes=witnesses,base_manifest=cfg['base_manifest'],policy='frozen_instruct_no_adapter'))
        run=save(root/'DATA-REQUEST.json',dict(stage='screen',node='S1',configuration=signature,panel=cfg['panel'],
            base_manifest=cfg['base_manifest'],endpoint_witness_ref=service,scope='dataset_preparation',count=n))
        counts=[0]*len(lanes);mtx=threading.Lock()
        def one(i):
            lane=i%len(lanes);d=draws[i];path=src/'packets'/(str(i)+'.json');p=read(path)
            needed(p['stage']=='S1' and p['source_binding']['split']=='TRAIN' and p['source_binding']['reference_id']==d['reference_id']
                  and p['branch']==d['fold'] and digest(canonical(p['wire']))==p['wire_sha256'] and
                  p['wire']['model']==cfg['model'],'packet not in frozen screen')
            needed(all(c.poll() is None for c in children),'lost service lease')
            tr=transports[lane];rid=p['request_id'];answer=tr.submit(rid,p)
            env=answer['native_envelope']
            import base64
            resp=json.loads(base64.b64decode(read(env['response_ref']['path'])['response_b64'],validate=True))
            needed(resp['model']==p['wire']['model'],'response from different model')
            sha=answer['native_response_sha256']
            receipt=dict(answer,provider_request_id=rid,request_id=rid,wire_sha256=p['wire_sha256'],response_sha256=sha,
                         receipt=dict(request_sha256=p['wire_sha256'],response_sha256=sha))
            row=dict(sample_id=p['sample_id'],stage='S1',request_id=rid,status='known',submits=1,
                     packet_ref=pin(path),receipt=receipt,lane=lanes[lane]['id'])
            save(root/'rows'/(str(i)+'.json'),row)
            with mtx:
                counts[lane]+=1
                if counts[lane]==1 or sum(counts)%cfg['progress_every']==0:
                    from common import save as update
                    update(root/'PROGRESS.json',dict(completed=sum(counts),population=n,by_lane=dict(zip((r['id'] for r in lanes),counts)),time=time.time()))
        def ready(i):
            while not (src/'packets'/(str(i)+'.json')).is_file():
                needed(all(c.poll() is None for c in children),'lost lease waiting for producer')
                if subprocess.run(['systemctl','--user','is-active','--quiet',cfg['producer_unit']]).returncode:
                    needed((src/'packets'/(str(i)+'.json')).is_file(),'producer ended without expected packet')
                time.sleep(2)
        width=cfg['concurrency_per_lane']*len(lanes)
        with cf.ThreadPoolExecutor(max_workers=width) as pool:
            pending=set()
            for i in range(n):
                if len(pending)>=width:
                    done,pending=cf.wait(pending,return_when=cf.FIRST_COMPLETED)
                    for task in done:task.result()
                if i%cfg['progress_every']==0:space.check()
                ready(i);pending.add(pool.submit(one,i))
            for task in cf.as_completed(pending):task.result()
        while not (src/'INPUTS.json').exists():
            needed(subprocess.run(['systemctl','--user','is-active','--quiet',cfg['producer_unit']]).returncode==0 or
                   (src/'INPUTS.json').is_file(),'producer did not seal inputs')
            time.sleep(1)
        inputs=read(src/'INPUTS.json')
        needed(inputs['panel_ref']==cfg['panel'] and len(inputs['rows'])==n,'rendered panel differs')
        def rows():
            for i,d in enumerate(draws):
                row=read(root/'rows'/(str(i)+'.json'));e=inputs['rows'][i]
                needed(e['packet']==row['packet_ref'] and e['reference_id']==d['reference_id'] and e['fold']==d['fold'],'sealed input differs')
                yield row
        stream=jsonl(root/'RECEIPTS.jsonl',rows(),space.check)
        version=save(root/'VERSION.json',dict(mode='live',synthetic=False,status='SEALED',unresolved_requests=0,
            system_version=cfg['system_version'],execution_version=signature['sha256'],dataset_request_ref=run,endpoint_witness_ref=service,
            base_manifest=cfg['base_manifest'],inputs_ref=pin(src/'INPUTS.json'),receipts=stream,content_format='prepared-panel-refs-v1'))
        save(root/'COMPLETE.json',dict(immutable=True,marker_scope='collection_analysis',version=version))

if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
