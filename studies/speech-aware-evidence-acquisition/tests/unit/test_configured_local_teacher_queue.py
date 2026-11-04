import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest

SCRIPT = Path(__file__).resolve().parents[2]/'scripts/environment/configured_local_teacher_queue.py'
spec = importlib.util.spec_from_file_location('local_teacher_queue', SCRIPT)
q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)


def fixture(tmp_path, behaviors=('ok',)):
    study = tmp_path/'study'; (study/'scripts/analysis').mkdir(parents=True)
    (study/'scripts/environment').mkdir(); (study/'src').mkdir()
    (study/'src/adapter.py').write_text('VALUE = 1\n')
    cli = study/'scripts/analysis/external_exploration.py'
    cli.write_text('''import json,sys,time,os
from pathlib import Path
root=Path(sys.argv[2]); d=json.loads((root/'prepared.json').read_text())
print(os.environ.get('CUSTOM_API_KEY',''),flush=True)
if sys.argv[1]=='run':
 with (root/'calls').open('a') as f:f.write('dispatch\\n')
 if d['behavior']=='fail':raise SystemExit(5)
 if d['behavior']=='slow':time.sleep(0.4)
 if d['behavior']=='crash-window':time.sleep(0.8)
 (root/'collected.json').write_text(json.dumps({'safe':True}))
else:
 if not (root/'collected.json').exists():raise SystemExit(6)
''')
    implementation=tmp_path/'implementation.json'
    implementation.write_text(json.dumps(q.implementation_inventory(cli)))
    jobs=[]
    for i, behavior in enumerate(behaviors):
        d=tmp_path/str(i);d.mkdir(); p=d/'prepared.json';p.write_text(json.dumps({'schema':q.KINDS['external-exploration'][1],'behavior':behavior}))
        jobs.append({'id':'j'+str(i),'kind':'external-exploration','cli':{'path':str(cli),'sha256':q.sha(cli)},'prepared':{'path':str(p),'sha256':q.sha(p)},'implementation':{'path':str(implementation),'sha256':q.sha(implementation)},'depends_on':[]})
    config=tmp_path/'queue.json';config.write_text(json.dumps({'schema':q.SCHEMA,'queue_id':'fixture','state_dir':str(tmp_path/'state'),'max_parallel':2,'jobs':jobs}))
    return config,jobs


def await_terminal(config):
    doc,root=q.load(config)
    end=time.monotonic()+8
    while time.monotonic()<end:
        if (root/'state.json').exists():
            states=q.read(root/'state.json')['jobs']
            if all(v not in ('pending','running') for v in states.values()):return states
        time.sleep(.02)
    raise AssertionError('queue did not finish')


def test_detached_start_survives_launcher_exit_no_secret_no_redispatch(tmp_path,monkeypatch):
    config,jobs=fixture(tmp_path,('slow','ok'))
    monkeypatch.setenv('CUSTOM_API_KEY','fixture-secret-never-log')
    output=subprocess.check_output([sys.executable,str(SCRIPT),'start','--config',str(config)],text=True)
    supervisor=json.loads(output)
    assert supervisor['pid']!=os.getpid()
    assert await_terminal(config)=={'j0':'complete','j1':'complete'}
    q.run(config)
    assert (tmp_path/'0/calls').read_text()=='dispatch\n'
    assert (tmp_path/'1/calls').read_text()=='dispatch\n'
    for p in (tmp_path/'state').rglob('*'):
        if p.is_file():assert 'fixture-secret-never-log' not in p.read_text()


def test_failure_does_not_block_independent_job_or_retry(tmp_path):
    config,jobs=fixture(tmp_path,('fail','ok'))
    assert q.run(config)=={'j0':'failed-held','j1':'complete'}
    q.run(config)
    assert (tmp_path/'0/calls').read_text()=='dispatch\n'


def test_unknown_launch_is_held_and_collect_only_reconciles(tmp_path):
    config,jobs=fixture(tmp_path)
    _,root=q.load(config);d=q.private_dir(root/'j0')
    q.persist(d/'launch.json',{'job_id':'j0'})
    assert q.run(config)=={'j0':'unknown-held'}
    assert not (tmp_path/'0/calls').exists()
    (tmp_path/'0/collected.json').write_text('{"safe": true}')
    assert q.reconcile(config,'j0')['status']=='complete'
    assert q.run(config)=={'j0':'complete'}
    assert not (tmp_path/'0/calls').exists()


def test_live_child_prevents_reconcile(tmp_path):
    config,jobs=fixture(tmp_path)
    _,root=q.load(config);d=q.private_dir(root/'j0');q.persist(d/'child.json',q.process_identity(os.getpid()))
    with pytest.raises(ValueError,match='live'):q.reconcile(config,'j0')


def test_dependency_failure_keeps_descendant_pending(tmp_path):
    config,jobs=fixture(tmp_path,('fail','ok','ok'))
    doc=q.read(config);doc['jobs'][1]['depends_on']=['j0'];config.write_text(json.dumps(doc))
    assert q.run(config)=={'j0':'failed-held','j1':'pending','j2':'complete'}
    assert not (tmp_path/'1/calls').exists()


def test_changed_collected_or_prepared_is_rejected(tmp_path):
    config,jobs=fixture(tmp_path);q.run(config)
    (tmp_path/'0/collected.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):q.run(config)


@pytest.mark.parametrize('mutation',['cycle','duplicate','changed-pin'])
def test_invalid_configuration_never_creates_state(tmp_path,mutation):
    config,jobs=fixture(tmp_path);doc=q.read(config)
    if mutation=='cycle':doc['jobs'][0]['depends_on']=['j0']
    if mutation=='duplicate':doc['jobs'].append(doc['jobs'][0])
    if mutation=='changed-pin':doc['jobs'][0]['prepared']['sha256']='0'*64
    config.write_text(json.dumps(doc))
    with pytest.raises(ValueError):q.load(config)
    assert not (tmp_path/'state').exists()


def test_export_is_immutable_private_collected_handoff(tmp_path):
    config,jobs=fixture(tmp_path);q.run(config)
    target=tmp_path/'handoff'
    result=q.export(config,'j0',target)
    assert result['candidate_accepted'] is False and result['deployment_ready'] is False
    entry=next(i for i in result['files'] if i['source_path']==str(tmp_path/'0/collected.json'))
    assert q.sha(target/entry['path'])==q.sha(tmp_path/'0/collected.json')
    assert q.verify_handoff(target)==result
    assert target.stat().st_mode & 0o077 == 0
    with pytest.raises(ValueError,match='fresh'):q.export(config,'j0',target)


def test_killed_worker_keeps_live_child_held_then_collects_without_dispatch(tmp_path):
    config,jobs=fixture(tmp_path,('crash-window',));_,root=q.load(config)
    proc=q.launch_worker(config,root,jobs[0]);child_path=root/'j0/child.json'
    end=time.monotonic()+5
    while not child_path.exists() and time.monotonic()<end:time.sleep(.01)
    assert child_path.exists()
    proc.kill();proc.wait()
    assert q.job_status(root,jobs[0])=='running'
    with pytest.raises(ValueError,match='live'):q.reconcile(config,'j0')
    while q.alive(q.read(child_path)) and time.monotonic()<end:time.sleep(.01)
    assert q.job_status(root,jobs[0])=='unknown-held'
    assert q.reconcile(config,'j0')['status']=='complete'
    assert (tmp_path/'0/calls').read_text()=='dispatch\n'


def test_imported_implementation_change_blocks_dispatch_and_collect(tmp_path):
    config,jobs=fixture(tmp_path)
    (tmp_path/'study/src/adapter.py').write_text('VALUE = 2\n')
    with pytest.raises(ValueError,match='implementation'):q.run(config)
    with pytest.raises(ValueError,match='implementation'):q.reconcile(config,'j0')
    assert not (tmp_path/'0/calls').exists()


def test_old_failed_result_cannot_hide_live_child(tmp_path):
    config,jobs=fixture(tmp_path,('fail',));q.run(config)
    _,root=q.load(config);q.persist(root/'j0/child.json',q.process_identity(os.getpid()),replace=True)
    assert q.job_status(root,jobs[0])=='running'
    with pytest.raises(ValueError,match='live'):q.reconcile(config,'j0')


def test_failed_collect_can_be_explicitly_reconciled_later(tmp_path):
    config,jobs=fixture(tmp_path)
    first=q.reconcile(config,'j0');assert first['status']=='failed-held'
    (tmp_path/'0/collected.json').write_text('{"safe":true}')
    second=q.reconcile(config,'j0');assert second['status']=='complete'
    assert first['attempt_id']!=second['attempt_id']
    results=list((tmp_path/'state/j0/reconciliations').glob('*.result.json'))
    assert len(results)==2 and {q.read(p)['status'] for p in results}=={'complete','failed-held'}
    assert not (tmp_path/'0/calls').exists()


def test_original_provider_guard_blocks_reconciliation(tmp_path):
    import fcntl
    config,jobs=fixture(tmp_path);guard=tmp_path/'0/request.guard'
    with guard.open('a+b') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):q.reconcile(config,'j0')
    assert not (tmp_path/'0/calls').exists()


def test_recursive_export_source_closure_and_receiver_tamper(tmp_path):
    config,jobs=fixture(tmp_path);q.run(config)
    source=tmp_path/'source.json';source.write_text('{"evidence":"real source"}')
    extra=tmp_path/'0/material.json';extra.write_text(json.dumps({'source':{'path':str(source),'sha256':q.sha(source)}}))
    result=q.export(config,'j0',tmp_path/'handoff')
    entry=next(i for i in result['files'] if i['source_path']==str(source))
    assert (tmp_path/'handoff'/entry['path']).read_bytes()==source.read_bytes()
    (tmp_path/'handoff'/entry['path']).write_text('{}')
    with pytest.raises(ValueError,match='bytes changed'):q.verify_handoff(tmp_path/'handoff')


def test_export_rejects_credential_source_before_creating_target(tmp_path):
    config,jobs=fixture(tmp_path);q.run(config)
    (tmp_path/'0/leaked.json').write_text('{"headers":{"Authorization":"Bearer forbidden"}}')
    with pytest.raises(ValueError,match='credential|header'):q.export(config,'j0',tmp_path/'handoff')
    assert not (tmp_path/'handoff').exists()


def copied_profile(tmp_path, jobs):
    root = tmp_path/'0'
    template = root/'template.md'; template.write_text('algorithm-specific instructions')
    profile = root/'profile.json'
    profile.write_text(json.dumps({'schema':'saea-external-exploration-profile-v1',
        'template':{'path':'../../prompts/learner/tfrl/ace.md','sha256':q.sha(template)}}))
    prepared = root/'prepared.json'; document=q.read(prepared)
    document['files']={name:{'path':str(root/name),'sha256':q.sha(root/name)}
                       for name in ('profile.json','template.md')}
    prepared.write_text(json.dumps(document))
    jobs[0]['prepared']['sha256']=q.sha(prepared)
    return template,profile


def test_copied_profile_uses_prepared_template_binding_and_offline_closure(tmp_path):
    config,jobs=fixture(tmp_path);template,profile=copied_profile(tmp_path,jobs)
    doc=q.read(config);doc['jobs']=jobs;config.write_text(json.dumps(doc));q.run(config)
    result=q.export(config,'j0',tmp_path/'handoff')
    assert str(template) in {i['source_path'] for i in result['files']}
    template.unlink();profile.unlink()  # receiver must never consult sender paths
    assert q.verify_handoff(tmp_path/'handoff') == result


def test_copied_profile_rejects_different_template_digest(tmp_path):
    config,jobs=fixture(tmp_path);template,profile=copied_profile(tmp_path,jobs)
    document=q.read(profile);document['template']['sha256']='0'*64;profile.write_text(json.dumps(document))
    prepared=tmp_path/'0/prepared.json'; manifest=q.read(prepared)
    manifest['files']['profile.json']['sha256']=q.sha(profile);prepared.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='template binding differs'):
        q.source_inventory(jobs[0],tmp_path/'0/prepared.json')


def test_relative_reference_confined_and_hash_verified(tmp_path):
    config,jobs=fixture(tmp_path);q.run(config)
    source=tmp_path/'0/evidence.txt';source.write_text('evidence')
    extra=tmp_path/'0/material.json'
    extra.write_text(json.dumps({'source':{'path':'evidence.txt','sha256':q.sha(source)}}))
    q.export(config,'j0',tmp_path/'handoff')
    source.write_text('changed')
    with pytest.raises(ValueError,match='pinned file changed'):q.export(config,'j0',tmp_path/'bad-hash')
    extra.write_text(json.dumps({'source':{'path':'../outside','sha256':'0'*64}}))
    with pytest.raises(ValueError,match='escapes'):q.export(config,'j0',tmp_path/'escape')


def test_export_overlay_keeps_original_runner_identity(tmp_path):
    config,jobs=fixture(tmp_path);q.run(config)
    original=tmp_path/'original-runner.py';original.write_text('original historical runner')
    identity=tmp_path/'state/identity.json';document=q.read(identity)
    document['runner_sha256']=q.sha(original);identity.write_text(json.dumps(document));before=identity.read_bytes()
    with pytest.raises(ValueError,match='implementation changed'):q.export(config,'j0',tmp_path/'no-overlay')
    manifest=q.export(config,'j0',tmp_path/'overlay',source_runner={'path':str(original),'sha256':q.sha(original)})
    assert manifest['source_runner']['sha256']==q.sha(original)
    assert identity.read_bytes()==before
    assert (tmp_path/'0/calls').read_text()=='dispatch\n'
    with pytest.raises(ValueError,match='implementation changed'):q.run(config)
