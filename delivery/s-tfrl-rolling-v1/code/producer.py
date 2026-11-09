"""Detached CPU producer: actual prepare -> fixed image budget -> owner ready.

No service/guard/lane ownership. All mutations stay in this delivery and an
independent CPU workspace. Failed CPU attempts are archived without model retry.
"""
import concurrent.futures,fcntl,json,os,shlex,subprocess,sys,tarfile,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'delivery/s-tfrl-rolling-v1'
sys.path.insert(0,str(ROOT/'delivery/s-single-intent-fewshot-v1/code'))
from common import read,save,sha
IMAGE='sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026'
M='/workspace/omni'
HOST='npu2'
STAGE='/srv/omni-local-state/s-tfrl-rolling-cpu-v2'
REMOTE=STAGE+'/package'
WORK='/srv/omni-local-state/s-rolling-12h-v1/base-workspace'

def ssh(args,**kwargs):return subprocess.run(['ssh','-o','BatchMode=yes',HOST,shlex.join(args)],check=True,**kwargs)

def verify(task):
    with (OUT/'cpu-verify.guard').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return verify_locked(task)

def verify_locked(task):
    folder=ROOT/task['prepared'];name=task['task_id'];receipt=OUT/'evidence'/name
    if (receipt/'CPU-VERIFIED.json').exists():return
    if (receipt/'LAUNCH.json').exists() and not (receipt/'WAIT.json').exists():
        cid=read(receipt/'LAUNCH.json')['stdout'].strip()
        r=ssh(['sudo','-n','docker','wait',cid],capture_output=True,text=True)
        save(receipt/'WAIT.json',dict(at=time.time(),stdout=r.stdout,stderr=r.stderr,returncode=r.returncode,adopted_CPU_container=True))
        with (receipt/'fixed-image.log').open('wb') as f:ssh(['sudo','-n','docker','logs',cid],stdout=f,stderr=subprocess.STDOUT)
        if r.stdout.strip()=='0':
            with (receipt/'evidence.tar').open('wb') as f:ssh(['sudo','-n','tar','cf','-','-C',STAGE,'evidence/'+name],stdout=f)
            with tarfile.open(receipt/'evidence.tar') as tf:
                assert all(not m.name.startswith('/') and '..' not in Path(m.name).parts and (m.isfile() or m.isdir()) for m in tf.getmembers())
                tf.extractall(OUT,filter='data')
            assert read(receipt/'CPU-VERIFIED.json')['status']=='PASS_FIXED_IMAGE_CPU'
            return
    if receipt.exists():
        receipt.rename(receipt.with_name(name+'-attempt-'+str(time.time_ns())))
    files=[p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files.extend([p for p in (OUT/'code').glob('*.py')])
    archive=OUT/'staging'/(name+'.tar.gz');archive.parent.mkdir(exist_ok=True)
    with tarfile.open(archive,'w:gz',compresslevel=1) as tf:
        for p in files:tf.add(p,arcname=str(p.relative_to(OUT)),recursive=False)
    remote_package=STAGE+'/packages/'+name+'-'+str(time.time_ns())
    ssh(['mkdir','-p',remote_package])
    with archive.open('rb') as f:ssh(['tar','xzf','-','-C',remote_package],stdin=f)
    namec='omni-sroll-cpu-'+name+'-'+str(time.time_ns())[-10:]
    argv=['sudo','-n','docker','run','-d','--pull','never','--name',namec,'--network','none','--cpus','3','--memory','10g',
        '-e','TORCH_DEVICE_BACKEND_AUTOLOAD=0','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','OMP_NUM_THREADS=1',
        '--mount',f'type=bind,src={WORK},dst={M},readonly',
        '--mount',f'type=bind,src={remote_package},dst={M}/delivery/s-tfrl-rolling-v1,readonly',
        '--mount',f'type=bind,src={STAGE}/evidence,dst=/evidence',
        '--mount','type=bind,src=/mnt/cephfs/omni/models/qwen3-omni-30b-a3b-thinking,dst=/model,readonly',
        '--entrypoint','/usr/local/python3.12.13/bin/python3',IMAGE,'-B',M+'/delivery/s-tfrl-rolling-v1/code/verify_one.py',
        '--config',M+'/'+task['config'],'--output','/evidence/'+name]
    receipt.mkdir(parents=True,exist_ok=True)
    r=ssh(argv,capture_output=True,text=True)
    save(receipt/'LAUNCH.json',dict(at=time.time(),argv=argv,stdout=r.stdout,stderr=r.stderr,returncode=r.returncode,
        immutable_per_attempt_package=remote_package,
        archive_sha256=sha(archive.read_bytes()),archive_bytes=archive.stat().st_size))
    cid=r.stdout.strip();r=ssh(['sudo','-n','docker','wait',cid],capture_output=True,text=True)
    save(receipt/'WAIT.json',dict(at=time.time(),stdout=r.stdout,stderr=r.stderr,returncode=r.returncode))
    with (receipt/'fixed-image.log').open('wb') as f:ssh(['sudo','-n','docker','logs',cid],stdout=f,stderr=subprocess.STDOUT)
    assert r.stdout.strip()=='0','fixed image failed, preserve CPU attempt and logs'
    with (receipt/'evidence.tar').open('wb') as f:ssh(['sudo','-n','tar','cf','-','-C',STAGE,'evidence/'+name],stdout=f)
    with tarfile.open(receipt/'evidence.tar') as tf:
        assert all(not m.name.startswith('/') and '..' not in Path(m.name).parts and (m.isfile() or m.isdir()) for m in tf.getmembers())
        tf.extractall(OUT,filter='data')
    assert r.stdout.strip()=='0','fixed image failed, preserve CPU attempt'
    assert read(receipt/'CPU-VERIFIED.json')['status']=='PASS_FIXED_IMAGE_CPU'

def publish(task):
    name=task['task_id'];receipt=OUT/'evidence'/name
    cfg=read(ROOT/task['config']);budget=read(receipt/'budget.json')
    assert budget['manifest_sha256']==cfg['manifest_sha256']
    rel=lambda p:str(p.relative_to(ROOT))
    files=[p for p in (ROOT/task['prepared']).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    adapter_paths=[]
    for field in ('runner','bind_script'):
        if task.get(field):adapter_paths.append(ROOT/task[field])
    files.extend(adapter_paths)
    stage=OUT/'staging'/(name+'-FILES.json')
    save(stage,dict(files=[dict(path=rel(p),sha256=sha(p.read_bytes()),bytes=p.stat().st_size) for p in files]))
    result=task|dict(status='ready',manifest=cfg['manifest'],manifest_sha256=cfg['manifest_sha256'],
        budget=rel(receipt/'budget.json'),CPU_verification=rel(receipt/'CPU-VERIFIED.json'),workspace_files=rel(stage),
        dependencies=[],target='S frozen TRAIN/P3 parameters',samples=cfg.get('full_denominator',545),branches=cfg['expected_rows'],estimated_requests=cfg['expected_rows'],
        resources=dict(TP=4,cards=4,instances=1,concurrency=64,fixed_image=IMAGE),runtime_binding='execution owner only',
        estimated_seconds=cfg['expected_rows']*4,ETA_basis='rough current D 4s/branch; future throughput unknown',
        score_protocol=dict(primary='unchanged strict type/filler multiset + official conditional SLU',
            paired='same545; explicit local-branch and legacy all-row failures',
            normalized='accepted limited typed sidecar identical config both sides; not model gain',E2E=False),
        pins={rel(p):sha(p.read_bytes()) for p in [ROOT/task['config'],ROOT/cfg['manifest'],receipt/'budget.json',receipt/'CPU-VERIFIED.json']+adapter_paths},
        monetary_cost=None)
    with (OUT/'ready-writer.guard').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        doc=read(OUT/'READY-TASKS.json')
        if name not in {t['task_id'] for t in doc['tasks']}:doc['tasks'].append(result)
        save(OUT/'READY-TASKS.json',doc|dict(updated_at=time.time()))
    save(receipt/'PUBLISHED.json',dict(at=time.time(),task_id=name,status='READY_FOR_OWNER_AUTO_ADMISSION'))

def prepare(task):
    folder=ROOT/task['prepared']
    if (folder/'CPU-PREPARED.json').exists():return task
    args=[sys.executable,'-B',str(OUT/'code/local_prepare.py'),task['task_id']]
    if 'seed' in task:args+=[str(task['seed'])]
    with (OUT/(task['task_id']+'.prepare.log')).open('ab') as f:
        subprocess.run(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True)
    return task

def main():
    lock=(OUT/'producer.guard').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ssh(['mkdir','-p',REMOTE,STAGE+'/evidence'])
    window=read(OUT/'REQUESTS/QUEUE-DEFICIT.json')['window']
    tasks=[]
    for i,name in enumerate(['compact-json-v1','absent-slot-v1']):
        tasks.append(dict(task_id=name,prepared='delivery/s-tfrl-rolling-v1/candidates/'+name,
            config='delivery/s-tfrl-rolling-v1/candidates/'+name+'/single_intent_no_examples/native-UNBOUND.json',
            preferred_lane=['C','D'][i],parent_version='prepared-v5-dynamic/single_intent_no_examples',
            method=name,purpose='single mechanism informed by real failure/counterevidence'))
    for i,seed in enumerate(read(OUT/'config/grpo-v1.json')['seeds']):
        name='grpo-fewshot-seed'+str(seed)
        tasks.append(dict(task_id=name,seed=seed,prepared='delivery/s-tfrl-rolling-v1/grpo-reserve/'+name,
            config='delivery/s-tfrl-rolling-v1/grpo-reserve/'+name+'/single_intent_fewshot/native-UNBOUND.json',
            preferred_lane=['A','B','C','D'][i%4],parent_version='prepared-v5-dynamic/single_intent_fewshot',
            method='training_free_same_policy_group_sampling',purpose='actual eight-seed same-policy variance and group advantage; no weights update'))
    save(OUT/'PRODUCTION-PLAN.json',dict(at=time.time(),window=window,tasks=tasks,extra_requests=16350,
        estimated_lane_hours=10*11000/3600,coverage_12h='NOT_YET_SUFFICIENT; add actual teacher and legal TRAIN successor as feedback arrives'))
    save(OUT/'PRODUCER-START.json',dict(pid=os.getpid(),ppid=os.getppid(),at=time.time(),window=window,agent=False,
        runtime_monitor_seconds=600,mode='CPU preparation and ready handoff only'))
    # The first two prompt jobs may already be materializing in the invoking turn.
    for task in tasks[:2]:
        while not (ROOT/task['prepared']/'CPU-PREPARED.json').exists():time.sleep(5)
        try:verify(task);publish(task)
        except Exception as e:save(OUT/'evidence'/task['task_id']/'PRODUCER-FAILED.json',dict(at=time.time(),error=str(e),traceback=traceback.format_exc(),model_calls=0))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(prepare,t):t for t in tasks[2:]}
        for future in concurrent.futures.as_completed(futures):
            task=futures[future]
            if time.time()>=window['end_epoch']:break
            try:
                future.result();verify(task);publish(task)
            except Exception as e:
                save(OUT/'evidence'/task['task_id']/'PRODUCER-FAILED.json',dict(at=time.time(),error=str(e),traceback=traceback.format_exc(),model_calls=0))
    save(OUT/'PRODUCER-BATCH-COMPLETE.json',dict(at=time.time(),window_end=window['end_epoch'],status='BATCH_DONE_ROLLING_GOAL_ACTIVE'))

if __name__=='__main__':main()
