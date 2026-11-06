import importlib.util, json, sys
from pathlib import Path
import pytest

SCRIPT=Path(__file__).resolve().parents[2]/'scripts/environment'
sys.path.insert(0,str(SCRIPT))
spec=importlib.util.spec_from_file_location('overlay_handoff',SCRIPT/'configured_overlay_score_handoff.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)

class FakeLauncher:
 def __init__(self,state='not-launched'):self.state=state;self.calls=[]
 def execute(self,path,mode):
  self.calls.append(mode)
  if mode=='launch':self.state='running'
  return {'state':self.state}

def base(tmp_path,monkeypatch,attempt='OPENED',producer_state='exited'):
 state=tmp_path/'state';run=tmp_path/'run';run.mkdir();a=tmp_path/'attempt';a.write_text(json.dumps({'state':attempt})); config={'state_dir':str(state),'producer_launch':{'path':'ignored'},'run_id':'RUN','attempt_id':'generation-v1'}
 monkeypatch.setattr(h,'pin',lambda p:Path('/ignored'));monkeypatch.setattr(h.launcher,'load_config',lambda p:{'expected_commit':'x','argv':['--candidate','c','--out','/work/generation-v1']});monkeypatch.setattr(h,'producer_status',lambda *a:{'state':producer_state,'exit_code':1});monkeypatch.setattr(h,'files',lambda *a:(Path('/logical'),Path('/logical/runs/RUN/generation-v1'),run));monkeypatch.setattr(h,'map_path',lambda *args:a)
 return config,run

def test_running_never_writes_or_launches(tmp_path,monkeypatch):
 d,run=base(tmp_path,monkeypatch,producer_state='running');f=FakeLauncher();assert h.step(d,f)['status']=='waiting-producer';assert not f.calls and not list(run.iterdir())

def test_finalized_does_not_launch(tmp_path,monkeypatch):
 d,_=base(tmp_path,monkeypatch,attempt='FINALIZED');monkeypatch.setattr(h,'producer_status',lambda *a:{'state':'exited','exit_code':0,'container_id':'a'*64});monkeypatch.setattr(h,'finalized',lambda *a:{'verified':True});f=FakeLauncher();assert h.step(d,f)['status']=='score-ready';assert not f.calls

def test_known_failure_launches_once_restart_observes(tmp_path,monkeypatch):
 d,_=base(tmp_path,monkeypatch);monkeypatch.setattr(h,'sealed_inputs',lambda *a:{'manifest':'m'});monkeypatch.setattr(h,'make_launch',lambda *a:{'schema':'test'});f=FakeLauncher();assert h.step(d,f)['status']=='recovery-running';assert f.calls==['layout','launch'];assert h.step(d,f)['status']=='recovery-running';assert f.calls==['layout','launch','status']

def test_unknown_or_mismatched_state_does_not_launch(tmp_path,monkeypatch):
 d,_=base(tmp_path,monkeypatch,attempt='UNKNOWN');f=FakeLauncher()
 with pytest.raises(ValueError,match='unknown attempt'):h.step(d,f)
 assert not f.calls

def test_partial_config_without_receipt_is_held(tmp_path,monkeypatch):
 d,_=base(tmp_path,monkeypatch);state=Path(d['state_dir']);state.mkdir();(state/'recovery-launch.json').write_text('{}');f=FakeLauncher()
 with pytest.raises(ValueError,match='unresolved without FINALIZED'):h.step(d,f)
 assert f.calls==['status']

def test_duplicate_lock_refused(tmp_path,monkeypatch):
 import fcntl
 state=tmp_path/'state';state.mkdir();config={'state_dir':str(state)};monkeypatch.setattr(h,'load',lambda path:config)
 with (state/'handoff.guard').open('a+b') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  with pytest.raises(BlockingIOError):h.execute('/ignored','once')

def test_bad_producer_identity_refused(tmp_path,monkeypatch):
 import subprocess
 doc={'receipt_dir':str(tmp_path),'expected_commit':'a'};p=tmp_path/'launcher';p.write_text('tool');monkeypatch.setattr(h,'pin',lambda pth:p);monkeypatch.setattr(h.subprocess,'run',lambda *a,**k:subprocess.CompletedProcess([],0,json.dumps({'state':'exited','container_id':'a'*64}),''))
 with pytest.raises(ValueError,match='container differs'):h.producer_status({'producer_launcher':{'sha256':'x'},'producer_launch':{'path':'x'},'container_id':'b'*64},doc)

def test_real_failure_gate_rejects_unknown_and_existing_recovery(tmp_path,monkeypatch):
 import hashlib
 host=tmp_path/'run';host.mkdir();state=tmp_path/'worktree';state.mkdir();(host/'score-batch').mkdir();run='RUN';d={'run_id':run,'attempt_id':'generation-v1','worktree':str(state),'worktree_commit':'a'*40};producer={'expected_commit':'b'*40,'argv':['--candidate','candidate']};rel=tmp_path/'release.json';base={'config':{'fragments':{'experiment':'candidate'}},'run_id':run,'attempt':{'attempt_id':'generation-v1'},'status':'COMPLETED','scores':None,'study_commit':'b'*40,'split':{'count':1080},'row_checkpoint':{'complete_rows':1080}}
 for suffix in ('run-manifest.json','generation-manifest.json'):(host/(run+'.'+suffix)).write_text(json.dumps(base))
 g=hashlib.sha256((host/(run+'.generation-manifest.json')).read_bytes()).hexdigest();rel.write_text(json.dumps({'run_id':run,'attempt_id':'generation-v1','study_commit':'b'*40,'generation_sealed':True,'generation_lease_released':True,'rows':1080,'artifacts':{'generation_manifest':{'sha256':g}}}));monkeypatch.setattr(h,'map_path',lambda *a:rel)
 for name in ('score-jobs.jsonl','score-ledger-cells.json'):(host/name).write_text('{}')
 err=host/'score-batch/generation-v1.stderr.log';err.write_text('delivery config differs from recorded run');failure={'failed':1,'succeeded':0,'job_count':1,'operation':'finalize','jobs':[{'status':'failed','returncode':1,'result_file':None,'complete_receipt_file':None,'stderr_sha256':hashlib.sha256(err.read_bytes()).hexdigest()}]};(host/'score-batch/batch-summary.json').write_text(json.dumps(failure));(host/'flight-receipts.json').write_text(json.dumps({'stages':{'score':{'result':failure}}}))
 class R:
  returncode=0
  def __init__(self,stdout):self.stdout=stdout
 monkeypatch.setattr(h.launcher,'run',lambda argv:R(('a'*40+'\n') if 'rev-parse' in argv else ''))
 assert h.sealed_inputs(d,producer,host)['manifest']==g
 (host/'score-recovery-partial').mkdir()
 with pytest.raises(ValueError,match='recovery already exists'):h.sealed_inputs(d,producer,host)
 (host/'score-recovery-partial').rmdir();err.write_text('unrelated failure')
 with pytest.raises(ValueError,match='not known overlay'):h.sealed_inputs(d,producer,host)
