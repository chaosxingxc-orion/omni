import hashlib,json,os,sys,traceback
from pathlib import Path
from unittest.mock import patch
root=Path('/candidate/studies/speech-aware-evidence-acquisition'); data=Path('/mnt/cephfs/xingchao/omni'); out=Path('/fixture'); package=Path('/out')/(os.environ.get('FIXTURE_ARM', 'A')+'-final')
sys.path.insert(0,str(root/'scripts/analysis'))
import admit_real_engine_candidate as admission
from speech_aware_evidence_acquisition.core.session import serving,receipt,vllm_identity

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
config=json.loads(next(package.glob('docs/readiness/*/external-successor-admission.json')).read_bytes())
mode = os.environ.get('FIXTURE_MODE', 'success')
config['workspace']=str(out/('workspace' if mode == 'success' else 'workspace-'+mode))
if mode != 'success':
 config['decision_relpath'] = 'coordination/'+config['candidate']+'/offline-'+mode+'.json'
if mode == 'missing-pin':
 del config['plans']['fit']
old=data/config['predecessor_decision']['relpath']; config['predecessor_decision']['sha256']=sha(old)
manifest_doc=json.loads((package/'assets.json').read_bytes())
original=json.loads((package/('docs/readiness/'+config['candidate']+'/predecessor-registration.json')).read_bytes())
old_receipt=json.loads(old.read_bytes()); ledger=out/('exposure-ledger-'+mode+'.md')
ledger.write_bytes((root/'docs/exposure-ledger.md').read_bytes())
keys=['date','run/experiment id','execution profile','scope','model/tool','speech carrier + split','split role','split identity hash','consumed','calls','speech audio seconds','gpu hours','inherited exposure','artifacts','notes']
rows=[]
for kind in ('generation','fit'):
 b=old_receipt['budget'][kind]; values=['2026-11-05',b['run_id'],b['execution_profile'],'offline receipt fixture','model','slurp train',b['split_role'],b['split_identity_hash'],'no',str(b['planned_model_calls']),str(b['planned_speech_audio_seconds']),str(b['planned_gpu_hours']),'predecessor costs retained','docs/readiness/'+config['candidate']+'/generation-plan-registered-v1.json','offline fixture from archived receipt']
 rows.append('| '+' | '.join(values)+' |\n')
with ledger.open('a') as f:f.write('\n## Offline fixture predecessor rows\n\n| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'+''.join(rows))
manifest_doc['files']['docs/exposure-ledger.md']={'path':str(ledger),'sha256':sha(ledger)}
manifest=out/('assets-'+mode+'.json');manifest.write_text(json.dumps(manifest_doc));os.environ['SAEA_EXPERIMENT_MANIFEST']=str(manifest);os.environ['SAEA_EXPERIMENT_MANIFEST_SHA256']=sha(manifest)
spec=json.loads((package/config['candidate_pin']['relpath']).read_bytes())
os.environ['SAEA_ADMITTED_LEVEL']=str(spec['execution']['admitted_level']);os.environ['SAEA_DEPLOYMENT_PROFILE']=spec['profile']
resident=out/('resident-'+os.environ.get('FIXTURE_ARM','A')+'-'+mode);resident.mkdir(exist_ok=True);(resident/'session.json').write_text('{}')
runtime=json.loads((package/('docs/'+spec['runtime_receipt'])).read_bytes())
for item in runtime['deployment']['instances']:
 p=data/'serving'/item['instance_id']/'session-receipt.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
from contextlib import ExitStack
before_old=old.read_bytes(); old_plan=(package/'docs/readiness'/config['candidate']/'generation-plan-registered-v1.json').read_bytes()
try:
 with ExitStack() as stack:
  if mode == 'prepared-failure':
   from speech_aware_evidence_acquisition.core.flight import cli
   original_check=cli.assert_prepared_artifacts_current
   count=[0]
   def fail_final(flights):
    count[0]+=1
    original_check(flights)
    if count[0] == 2: raise cli_preparation_error('injected final prepared refusal')
   from speech_aware_evidence_acquisition.core.preparation import PreparationError as cli_preparation_error
   stack.enter_context(patch.object(cli,'assert_prepared_artifacts_current',fail_final))
  if mode == 'publication-failure':
   from speech_aware_evidence_acquisition.core import atomic_file
   real_publish=atomic_file.publish_new
   def fail_marker(path, raw):
    if Path(path).name == 'completion.json': raise OSError('injected completion publication failure')
    return real_publish(path, raw)
   stack.enter_context(patch.object(atomic_file,'publish_new',fail_marker))
  stack.enter_context(patch.object(serving,'find_resident_session',return_value=(resident,{'session_id':spec['service_session']})))
  stack.enter_context(patch.object(serving,'assert_identity_matches'))
  stack.enter_context(patch.object(receipt.SessionReceipt,'from_json',return_value=object()))
  stack.enter_context(patch.object(vllm_identity,'verify_vllm_session'))
  result=admission.admit_external(root,data,config)
 if mode != 'success': raise AssertionError('negative case accepted')
 print(json.dumps({'result':result},indent=2),flush=True)
except Exception as exc:
 if mode == 'success': traceback.print_exc();raise
 expected = {'missing-pin': (ValueError, 'three explicit successor document pins'),
             'prepared-failure': (cli_preparation_error, 'injected final prepared refusal') if mode == 'prepared-failure' else None,
             'publication-failure': (OSError, 'injected completion publication failure')}
 kind, message = expected[mode]
 assert isinstance(exc, kind) and message in str(exc), repr(exc)
 assert not (Path(config['workspace'])/'completion.json').exists()
 assert old.read_bytes()==before_old
 assert (package/'docs/readiness'/config['candidate']/'generation-plan-registered-v1.json').read_bytes()==old_plan
 new=data/config['decision_relpath']
 if mode == 'publication-failure':
  assert new.is_file() and (Path(config['workspace'])/'publication-intent.json').is_file()
  receipt_bytes = new.read_bytes()
  assert admission.resume_external_completion(config['workspace'], data)['receipt']['sha256'] == sha(new)
  assert admission.resume_external_completion(config['workspace'], data)['receipt']['sha256'] == sha(new)
  assert new.read_bytes() == receipt_bytes
 else: assert not new.exists()
 assert all(not os.environ.get(k) for k in ('SAEA_EXPERIMENT_WORKSPACE','SAEA_EXPERIMENT_WORKSPACE_MANIFEST','SAEA_EXPERIMENT_WORKSPACE_SHA256'))
 print(json.dumps({'mode':mode,'rejected':type(exc).__name__,'error':str(exc),'marker_absent_before_resume':True,'old_bytes_unchanged':True,'receipt_published':new.exists(),'resumed_twice':mode=='publication-failure'}),flush=True)

