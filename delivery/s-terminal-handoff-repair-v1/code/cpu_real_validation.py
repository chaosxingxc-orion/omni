"""Real frozen TRAIN bind/load and native pre-admission checks; never calls a model.

OWNER_RECONCILED, service/session and container probes here are explicitly CPU
configuration fixtures. A trace stops the unmodified native coroutine immediately
before its first local guard / journal / lease. No model function is replaced.
"""
import argparse
import ast
import asyncio
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OWNER = ROOT / 'delivery/s-terminal-handoff-repair-v1'
PACKAGE = OWNER / 'handoff-execution-code'
SOURCE = ROOT / 'delivery/s-tfrl-rolling-v1/train-paired-fewshot/train-remainder-part1-fewshot-v1/single_intent_fewshot/native-UNBOUND.json'
BUDGET = ROOT / 'delivery/s-tfrl-rolling-v1/evidence/train-remainder-part1-fewshot-v1/budget.json'
sys.path.insert(0, str(OWNER/'code'))
sys.path.insert(0, str(ROOT/'delivery/s-single-intent-fewshot-v1/code'))
from common import read, save, sha, require, load
import terminal_handoff as handoff

RELS = ('train_runtime.py','bind_train.py','train-runtime-v1/run_s_native.py','train-runtime-v1/native_resources.py')


def no_io_audit(event, args):
    if event in ('socket.connect','socket.connect_ex','socket.getaddrinfo','socket.bind','os.system'):
        raise RuntimeError('CPU suite forbids external I/O: '+event)
    if event == 'subprocess.Popen':
        argv = args[1]
        require(isinstance(argv, (list,tuple)) and len(argv)>2 and
                argv[0] == sys.executable and argv[1] == '-B' and
                Path(argv[2]).resolve() == PACKAGE/'bind_train.py', 'only real CPU binder subprocess allowed')


def inventory(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob('*')
                  if p.name in ('contact-journals','flight.lock','generation.guard') or
                  p.name.endswith(('.intent.json','.response.json')))


def native_check(module, cfg, config_path, execution_path, out, expected, receipt_block_only=False):
    # Stop at the real pre-admission boundary after capture/session/PID checks.
    # settrace observes native code; it does not replace preflight, HTTP, journal,
    # receipt validation, lease acquisition, or any model call.
    line = next(i for i,s in enumerate(Path(module.__file__).read_text().splitlines(),1)
                if "localguard=(out/'generation.guard')" in s)
    class CpuBoundaryReached(Exception):
        pass
    reached = False
    def trace(frame, event, arg):
        nonlocal reached
        if frame.f_code is module.run.__code__ and event == 'line' and frame.f_lineno == line:
            reached = True
            raise CpuBoundaryReached('CPU only; stopped before local guard/journal/lease')
        return trace if frame.f_code is module.run.__code__ else None
    if receipt_block_only:
        # Local fallback executes the EXACT AST from native run's predecessor
        # receipt block. No imports/model functions are mocked. Fixed suite
        # never uses this mode and exercises the full native coroutine.
        tree=ast.parse(Path(module.__file__).read_text())
        run=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='run')
        first=next(i for i,n in enumerate(run.body) if isinstance(n,ast.Assign) and
                   any(isinstance(t,ast.Name) and t.id=='pred' for t in n.targets))
        last=next(i for i,n in enumerate(run.body) if isinstance(n,ast.Assign) and
                  any(isinstance(t,ast.Name) and t.id=='capture_path' for t in n.targets))
        code=compile(ast.Module(body=run.body[first:last],type_ignores=[]),module.__file__,'exec')
        ns=dict(module.__dict__,runtime=cfg['runtime'],c=cfg)
        try:exec(code,ns)
        except (RuntimeError,ValueError) as exc:
            require(expected!='pass' and expected in str(exc),'unexpected native receipt rejection')
        else:require(expected=='pass','tampered receipt accepted')
        return dict(native_source=str(Path(module.__file__).relative_to(ROOT)),expected=expected,
                    mode='exact_native_receipt_AST_local_only',model_calls=0,journal_created=False)
    prior = sys.gettrace()
    try:
        sys.settrace(trace)
        try:
            asyncio.run(module.run(str(config_path),str(ROOT),str(out),str(execution_path)))
        except CpuBoundaryReached:
            require(expected == 'pass', 'invalid predecessor unexpectedly accepted')
        except (ValueError,RuntimeError) as exc:
            require(expected != 'pass' and expected in str(exc), 'unexpected native rejection: '+repr(exc))
        else:
            raise AssertionError('native coroutine returned without CPU boundary')
    finally:
        sys.settrace(prior)
    require(reached == (expected == 'pass'), 'native pre-admission result')
    require(not inventory(out), 'native validation created journal/lease/guard')
    return dict(native_source=str(Path(module.__file__).relative_to(ROOT)), expected=expected,
                reached_before_journal=reached, journal_created=False, model_calls=0)


def build_runtime(out, cfg, mode):
    root=out/'runtime';root.mkdir()
    lane=root/'lane';locks=lane/'locks';locks.mkdir(parents=True)
    guard=locks/'generation.guard';guard.touch()  # CPU fixture only, no live guard.
    seed=read(OWNER/'config/AB-runtime-seed.json')
    rt=copy.deepcopy(seed['A'])
    rt.update(status='OWNER_RECONCILED',TP=4,cards=4,instances=1,context_length=65536,
              boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              service_pid=os.getpid(),shared_guard=str(guard),shared_guard_inode=guard.stat().st_ino,
              container_lane=str(lane),serving_root=str(root/'serving'),
              native_fit_receipt=handoff.pin(BUDGET))
    # Real seed values are CPU-only config; no endpoint/health/metrics request.
    session=Path(rt['serving_root'])/rt['session_id']/'session.json'
    save(session,dict(session_id=rt['session_id'],pid=rt['service_pid'],
                      identity=dict(port=int(rt['endpoint'].rsplit(':',1)[1]))))
    if mode=='complete':
        pred=dict(status='COMPLETE',unknown=0,container='CPU-complete',container_state=dict(Status='exited',Running=False,Pid=0))
    else:
        evidence=copy.deepcopy(read(OWNER/'fixtures/D-FAILED-PRESERVED.json'))
        launch=evidence['launch'];failed=root/'jobs'/launch['task_id']
        launch.update(state=str(failed),pid=2147483647,runtime=copy.deepcopy(rt))
        save(failed/'LAUNCH.json',launch)
        save(failed/'run/TERMINAL.json',evidence['terminal'])
        save(failed/'runtime-config.json',dict(run_id=launch['run_id'],attempt_id=launch['attempt_id'],runtime=rt))
        if mode!='failed-no-flight':
            save(locks/'flight.lock',dict(schema='saea-generation-lease-v1',run_id=launch['run_id'],
                 attempt_id=launch['attempt_id'],session_id=rt['session_id'],study_head=rt['study_head'],
                 owner='s-correct-p-data-v1',run_ids=[launch['run_id']],pid=launch['pid'],token='c'*64))
        container=dict(Id=launch['container_id'],State=dict(Status='exited',Running=False,Pid=0))
        task=dict(task_id='CPU-independent-successor',cards=[0,1,2,3],dependencies=[])
        policy=dict(schema=handoff.SCHEMA,enabled=True,allowlist={task['task_id']:[launch['task_id']]})
        def verify(runtime,cards):
            require(runtime['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'CPU boot changed')
            require(runtime['service_pid']==os.getpid() and cards==[0,1,2,3], 'CPU identity')
        with guard.open('r+b') as held:
            fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
            rt['terminal_handoff_receipt']=handoff.reconcile(root,task,cfg,rt,evidence,policy,held,
                verify,lambda runtime:dict(running=0,waiting=0),lambda cid:container)
        pred=dict(**evidence['terminal'],container=launch['container_id'],container_state=container['State'])
    save(root/'predecessor.json',pred)
    rt['predecessor_receipt']=handoff.pin(root/'predecessor.json')
    capture=dict(status='PASS_RUNTIME_CAPTURE_NOT_STARTED',released=True,unknown=0,
                 **{k:rt[k] for k in ('boot_id','session_id','service_pid','service_id')},
                 lane_root=str(lane),guard_inode=guard.stat().st_ino,
                 predecessor=dict(receipt=rt['predecessor_receipt']),fit_sha256=rt['native_fit_receipt']['sha256'])
    save(root/'capture.json',capture);rt['capture_receipt']=handoff.pin(root/'capture.json')
    save(root/'CPU-ONLY.json',dict(runtime_admitted=False,owner_reconciled_is_fixture=True,
         service_and_container_probes='CPU config only',seed_source='config/AB-runtime-seed.json',
         original_fixed_window=read(OWNER/'config/WINDOW.json'),flight_bytes='CPU schema fixture, not historical flight'))
    return rt


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--receipt-block-only',action='store_true');a=p.parse_args()
    out=Path(a.output).resolve();require(not out.exists(),'fresh CPU output required');out.mkdir(parents=True)
    sys.addaudithook(no_io_audit)
    start=time.monotonic();original_pin=sha(SOURCE.read_bytes())
    sys.path.insert(0,str(PACKAGE))
    import train_runtime as train
    cfg=read(SOURCE);require((cfg['full_denominator'],cfg['expected_rows'])==(933,2799),'real 933/2799 input')
    # Runtime modules really import from this new package, never old execution-code.
    require(Path(train.__file__).resolve()==PACKAGE/'train_runtime.py','new TRAIN module')
    require(Path(train.original.runner.__file__).resolve()==PACKAGE/'train-runtime-v1/run_s_native.py','new TRAIN native')
    results=[]
    for workers in (64,96,128):
        attempt=out/('train'+str(workers));attempt.mkdir()
        actual=copy.deepcopy(cfg);actual.update(concurrency=workers,attempt_id=cfg['attempt_id']+'-cpu-c'+str(workers))
        actual['runtime']=dict(actual['runtime'],status='OWNER_RECONCILED',cpu_only_configuration=True)
        config=attempt/'config.json';execution=attempt/'execution.json';save(config,actual)
        t=time.monotonic()
        r=subprocess.run([sys.executable,'-B',str(PACKAGE/'bind_train.py'),'--config',str(config),
                           '--state',str(attempt/'unused-state'),'--output',str(execution)],capture_output=True,text=True)
        (attempt/'bind.stdout.log').write_text(r.stdout);(attempt/'bind.stderr.log').write_text(r.stderr)
        require(r.returncode==0,'real TRAIN binder failed: '+r.stderr)
        loaded=train.original.runner.load_execution(execution,actual,config)
        require(loaded['executor_max_workers']==workers,'native executor count')
        require(not (attempt/'unused-state').exists(),'binder wrote contact journal/state')
        results.append(dict(workers=workers,samples=933,branches=2799,wall_seconds=time.monotonic()-t,
                            bind_and_native_load=True,execution_sha256=sha(execution.read_bytes()),journal_created=False))
    validations=[]
    if a.receipt_block_only:
        real_cfg,real_packets=train.preflight(out/'train64/config.json',ROOT)
        train.original.validate_fit(read(BUDGET),real_cfg,real_packets,ROOT)
    # Actual coroutine with real TRAIN preflight/budget, both candidate native files.
    # Base-native test uses the established branch adapter seam with real TRAIN
    # preflight/receipt, solely to exercise its predecessor/capture logic at 64.
    base=load('cpu_handoff_base_native',OWNER/'code/native-runtime/run_s_native.py')
    base.preflight=train.preflight;base.validate_receipt=train.original.validate_fit
    resources=load('cpu_new_package_resources',PACKAGE/'train-runtime-v1/native_resources.py').prepare_native_resources()
    require(resources['network_calls']==0,'offline native resource')
    for label,module in [('train',train.original.runner),('base',base)]:
        for mode in ('complete','failed-flight','failed-no-flight','failed-tampered'):
            case=out/(label+'-'+mode);case.mkdir()
            actual=copy.deepcopy(cfg);actual['attempt_id']+='-cpu-'+label+'-'+mode
            actual['runtime']=build_runtime(case,actual,mode)
            config=case/'config.json';execution=case/'execution.json';save(config,actual)
            spec=read(out/'train64/execution.json')
            spec.update(run_id=actual['run_id'],attempt_id=actual['attempt_id'],
                        original_runtime_config_sha256=sha(config.read_bytes()),
                        runner_sha256=sha(Path(module.__file__).read_bytes()),
                        resources_module_sha256=sha(Path(module.__file__).with_name('native_resources.py').read_bytes()))
            save(execution,spec)
            expected='pass'
            if mode=='failed-tampered':
                receipt=actual['runtime']['terminal_handoff_receipt']
                Path(receipt['path']).write_bytes(Path(receipt['path']).read_bytes()+b' ')
                expected='handoff receipt pin'
            validations.append(dict(case=label+'-'+mode,**native_check(module,actual,config,execution,case/'native-out',expected,a.receipt_block_only)))
    require(sha(SOURCE.read_bytes())==original_pin,'source config changed')
    # Helpers only create fixture metadata under case/runtime; no native journals.
    require(not list(out.rglob('contact-journals')),'no journal allowed')
    save(out/'RESULT.json',dict(status='PASS_REAL_CPU_BIND_LOAD_NATIVE_VALIDATION',
         cpu_only=True,full_native_coroutine=not a.receipt_block_only,runtime_admitted=False,model_calls=0,NPU_calls=0,journal_created=False,
         model_functions_mocked=False,source_config_sha256=original_pin,bind_load=results,native=validations,
         pins={rel:sha((PACKAGE/rel).read_bytes()) for rel in RELS},resources=resources,
         wall_seconds=time.monotonic()-start))
    print(json.dumps(read(out/'RESULT.json')))

if __name__=='__main__':main()
