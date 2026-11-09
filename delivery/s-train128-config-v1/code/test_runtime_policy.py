"""Offline fixtures for bounded TRAIN concurrency/deadline and mocked launch wiring."""
import ast
import copy
import contextlib
import io
import json
import tempfile
from types import SimpleNamespace
from pathlib import Path

def main():
 source=Path(__file__).with_name('host_driver.py')
 tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='apply_runtime_policy')
 ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns);apply=ns['apply_runtime_policy']
 policy={'tasks':{'new':{'timeout_seconds':900}}}
 before=dict(dataset_partition='eligible_remainder',timeout_seconds=600,concurrency=96,manifest_sha256='unchanged')
 cfg=copy.deepcopy(before);change=apply(cfg,{'task_id':'new'},policy)
 assert cfg==dict(before,timeout_seconds=900) and change['original_timeout_seconds']==600
 for task,partition in [('old','eligible_remainder'),('new','S_valid'),('new','fit')]:
  cfg=dict(before,dataset_partition=partition);old=copy.deepcopy(cfg)
  assert apply(cfg,{'task_id':task},policy) is None and cfg==old
 for value in [0,599,1201,True,'900',900.0]:
  try:apply(copy.deepcopy(before),{'task_id':'new'},{'tasks':{'new':{'timeout_seconds':value}}})
  except AssertionError:pass
  else:raise AssertionError('unbounded deadline accepted')
 for concurrency in (64,96,128):
  cfg=copy.deepcopy(before);change=apply(cfg,{'task_id':'new'},{'tasks':{'new':{'concurrency':concurrency}}})
  assert cfg==dict(before,concurrency=concurrency) and change['effective_concurrency']==concurrency
 for timeout in (600,1200):
  cfg=copy.deepcopy(before);apply(cfg,{'task_id':'new'},{'TRAIN':{'timeout_seconds':timeout,'concurrency':128}})
  assert (cfg['timeout_seconds'],cfg['concurrency'])==(timeout,128)
 for value in [0,32,65,97,129,True,'128',128.0,None]:
  cfg=copy.deepcopy(before)
  try:apply(cfg,{'task_id':'new'},{'tasks':{'new':{'timeout_seconds':900,'concurrency':value}}})
  except AssertionError:assert cfg==before
  else:raise AssertionError('unsupported concurrency accepted')
 for key in ('model','manifest','packet','score','runner','max_tokens'):
  try:apply(copy.deepcopy(before),{'task_id':'new'},{'tasks':{'new':{key:'forbidden'}}})
  except AssertionError:pass
  else:raise AssertionError('scientific/resource override accepted')
 cfg=copy.deepcopy(before);apply(cfg,{'task_id':'new'},{'TRAIN':{'concurrency':128},'tasks':{'new':{'timeout_seconds':900}}})
 assert (cfg['concurrency'],cfg['timeout_seconds'])==(96,900),'task setting replaces TRAIN default as before'
 checks=['only_targeted_TRAIN','evaluation_and_existing_settings_unchanged','bounded_integer_deadline',
         'concurrency64_96_128','invalid_values_atomic','scientific_changes_rejected','existing_task_precedence']
 with contextlib.redirect_stdout(io.StringIO()):checks+=launch_fixtures(source)
 print(json.dumps(dict(status='PASS',tests=checks,fixture_groups=len(checks),NPU_calls=0,model_calls=0,formal_fixed_image=False)))

def launch_fixtures(source):
 # Import stdlib only; launch is invoked with all external effects replaced below.
 ns={'__name__':'offline_fixture'};exec(compile(source.read_text(),str(source),'exec'),ns)
 save=ns['save'];read=ns['read'];checks=[]
 for concurrency in (64,96,128):
  with tempfile.TemporaryDirectory() as tmp:
   H=Path(tmp);W=H/'workspace';W.mkdir();(H/'execution-code').mkdir()
   required=['train_runtime.py','bind_train.py','train-runtime-v1/run_s_native.py','train-runtime-v1/native_resources.py']
   pins={}
   for rel in required:
    p=H/'execution-code'/rel;p.parent.mkdir(exist_ok=True);p.write_text('# fixture only\n');pins[rel]=ns['sha'](p)
   save(H/'TRAIN96-VERIFIED.json',{'pins':pins})
   cfg=dict(dataset_partition='eligible_remainder',timeout_seconds=600,concurrency=64,
            attempt_id='v1',run_id='fixture',manifest_sha256='unchanged',expected_rows=3)
   save(W/'config.json',cfg);save(W/'budget.json',dict(status='PASS_FIXED_IMAGE_CPU',rows=3,manifest_sha256='unchanged'))
   guard=H/'shared.guard';guard.touch()
   rt=dict(shared_guard=str(guard),serving_root=str(H/'serving'),session_id='session',service_pid=1,
           boot_id='boot',service_id='service',shared_guard_inode=guard.stat().st_ino)
   task=dict(task_id='new',workspace=str(W),config='config.json',budget='budget.json',runtime=rt,
             predecessor=dict(container='old',terminal=str(H/'old-terminal.json')),cards=[0,1,2,3],
             window_end_epoch=ns['time'].time()+3600,runner='old64.py',bind_script='old64-bind.py',execution_attempt_suffix='-trial')
   save(H/'spec.json',task)
   # 96 deliberately omits concurrency: this must keep the current default.
   setting={'timeout_seconds':900}
   if concurrency!=96:setting['concurrency']=concurrency
   save(H/'runtime-policy.json',dict(tasks={'new':setting}))
   calls=[]
   def run(argv,**kw):
    calls.append(argv);return SimpleNamespace(returncode=0,stdout='container\n',stderr='')
   ns['subprocess']=SimpleNamespace(run=run)
   ns['inspect']=lambda name:dict(Id=name,State=dict(Running=False,Pid=0))
   ns['historic']=lambda p: (dict(status='COMPLETE',unknown=0) if str(p).endswith('old-terminal.json') else dict(session_id='session',pid=1))
   ns['historical_pin']=lambda p:dict(path=str(p),sha256='fixture')
   ns['verify']=lambda *a:None;ns['metrics']=lambda *a:dict(running=0,waiting=0)
   a=SimpleNamespace(root=str(H),spec=str(H/'spec.json'),lane='A')
   # Preserve unknown launch guard: zero mocked docker calls, zero policy artifacts.
   state=H/'jobs/new';save(state/'launch-intent.json',{'unknown':True})
   try:ns['launch'](a)
   except AssertionError as e:assert str(e)=='UNKNOWN_LAUNCH_NO_RESEND'
   else:raise AssertionError('unknown launch accepted')
   assert not calls and not (state/'EXECUTION-OVERRIDE.json').exists()
   (state/'launch-intent.json').unlink()  # fixture cleanup only, never live recovery
   (state/'run').mkdir()
   try:ns['launch'](a)
   except RuntimeError as e:assert str(e)=='EXISTING_RUN_STATE_RECONCILE_NO_RESEND'
   else:raise AssertionError('existing attempt can be retagged')
   assert not calls
   (state/'run').rmdir()  # fixture cleanup only
   # Missing/incomplete/corrupt package must fail before binding or admission.
   fixed=read(H/'TRAIN96-VERIFIED.json')
   for broken in (None,{'pins':{}},{'pins':dict(pins,**{'bind_train.py':'bad'})}):
    if broken is None:(H/'TRAIN96-VERIFIED.json').unlink()
    else:save(H/'TRAIN96-VERIFIED.json',broken)
    try:ns['launch'](a)
    except (AssertionError,FileNotFoundError):pass
    else:raise AssertionError('unverified package accepted')
    assert not calls
   save(H/'TRAIN96-VERIFIED.json',fixed)
   ns['launch'](a)
   effective=read(state/'runtime-config.json');override=read(state/'EXECUTION-OVERRIDE.json')
   expected=dict(cfg,concurrency=concurrency,timeout_seconds=900,attempt_id=f'v1-c{concurrency}-trial')
   assert {k:v for k,v in effective.items() if k!='runtime'}==expected
   assert override['effective_concurrency']==concurrency and override['effective_timeout_seconds']==900
   assert override['attempt_id']==effective['attempt_id']==read(state/'LAUNCH.json')['attempt_id']
   assert len(calls)==2
   for argv,name in zip(calls,['bind_train.py','train_runtime.py']):
    assert any(x.endswith('/rolling-code/'+name) for x in argv)
    assert f'type=bind,src={H}/execution-code,dst={ns["M"]}/delivery/s-single-intent-execution-v1/rolling-code,readonly' in argv
   # A second call returns the known receipt; never docker run again.
   ns['launch'](a);assert len(calls)==2
   # No policy still defaults to96; nonTRAIN retains original settings and runner.
   clone=copy.deepcopy(cfg);t=copy.deepcopy(task)
   override,change=ns['configure_train_execution'](H,clone,t,{}, {})
   assert clone['concurrency']==96 and clone['attempt_id']=='v1-c96' and change is None
   clone=dict(cfg,dataset_partition='S_valid');old=copy.deepcopy(clone);t=copy.deepcopy(task)
   assert ns['configure_train_execution'](H,clone,t,{}, {'TRAIN':{'concurrency':128}})==(None,None)
   assert clone==old and t==task
   for timeout in (599,1201,True):
    clone=dict(cfg,timeout_seconds=timeout)
    try:ns['configure_train_execution'](H,clone,copy.deepcopy(task),{}, {})
    except AssertionError:pass
    else:raise AssertionError('unbounded source deadline accepted')
   checks.append(f'mocked_launch_c{concurrency}_binding_mount_identity_pins_no_resend')
 return checks

if __name__=='__main__':main()
