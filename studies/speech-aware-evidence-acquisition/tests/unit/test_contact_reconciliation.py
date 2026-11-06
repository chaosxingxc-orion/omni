"""Positive retrospective proof, corruption, dispatch ambiguity and pinning."""
import copy
import json
from dataclasses import replace
from pathlib import Path
import sys
import pytest

from speech_aware_evidence_acquisition.core import contact_reconciliation as m
from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal, NativeContactReplay, canonical, digest
from test_contact_journal import payload
from test_frozen_core_adapter import _plan


def fixture(tmp_path):
    plan=_plan();j=ContactJournal(tmp_path/'data',plan,'bare-core');p=payload();j.intent(p)
    identity=j.identity;start='2026-11-06T00:00:00Z';end='2026-11-06T00:00:20Z'
    env={'SAEA_RELEASE_MANIFEST':'/opt/saea-release/source-manifest.json',
         'SAEA_OBSERVABILITY_RUN_ID':plan.run_id,'SAEA_OBSERVABILITY_ATTEMPT_ID':plan.attempt_id,
         'SAEA_OBSERVABILITY_MODE':'debug','SAEA_STAGE_TIMING_PATH':'/telemetry/stages.jsonl'}
    manifest={'schema':'saea-release-source-v1','commit':'c'*40,'files':{m.PREFIX+k:v for k,v in m.REVIEWED.items()}}
    env['SAEA_RELEASE_MANIFEST_SHA256']=digest(canonical(manifest))
    launch={'image':'sha256:'+'a'*64,'expected_commit':'c'*40,'environment':env,
            'source_identity':{'manifest_sha256':env['SAEA_RELEASE_MANIFEST_SHA256']},
            'mounts':[{'target':'/opt/saea-release','source':'/var/tmp/source','readonly':True},
                      {'target':'/telemetry','source':'/var/tmp/timing','readonly':False}]}
    container=[{'Id':'d'*64,'State':{'Status':'exited','Running':False,'Pid':0,'StartedAt':start,'FinishedAt':end},
                'Config':{'Labels':{'saea.launch.config':digest(canonical(launch))},'Image':launch['image'],
                          'Env':[k+'='+v for k,v in env.items()]},'HostConfig':{'PidMode':'host'},
                'Mounts':[{'Destination':'/opt/saea-release','Source':'/var/tmp/source','RW':False}]}]
    lease={'run_id':plan.run_id,'attempt_id':plan.attempt_id,'pid':999999999,'study_head':'c'*40,'started_at':start}
    attempt={'state':'ABORTED','run_id':plan.run_id,'attempt_id':plan.attempt_id,'carrier_lock_key':plan.carrier_lock_key,
             'actual':{'calls':1,'audio_seconds':100.0}}
    ap=tmp_path/'data/attempts'/plan.run_id/(plan.attempt_id+'.json');ap.parent.mkdir(parents=True);ap.write_bytes(canonical(attempt))
    effective={'mode':'debug','client_detail':True,'run_id':plan.run_id,'attempt_id':plan.attempt_id}
    timing_config={'effective':effective,'configuration_sha256':digest(json.dumps(effective,sort_keys=True).encode())}
    stages=['request_total','transport_total','request_render','session_total','session_namespace','session_namespace_probe']
    rows=[]
    wall=int(m._stamp(start)*1e9)
    for i,stage in enumerate(stages):
        rows.append({'schema':'saea-stage-timing-v1','stage':stage,'status':'error',
                     'started_monotonic':float(i),'elapsed_seconds':12.-2*i,
                     'started_unix_ns':wall+i*10**9,'pid':lease['pid'],'thread':1,
                     'run_id':plan.run_id,'attempt_id':plan.attempt_id,'sample_id':p['sample_id'],
                     'request_id':'req-1','observability_mode':'debug','span_id':str(i),'parent_span_id':str(i-1)})
    evidence={}
    docs={'launch':launch,'container':container,'source_manifest':manifest,'attempt':attempt,'lease':lease,'timing_config':timing_config}
    for name,value in docs.items():
        path=tmp_path/(name+'.json');path.write_bytes(canonical(value));evidence[name]={'path':str(path),'sha256':digest(path.read_bytes())}
    path=tmp_path/'timing.jsonl';path.write_bytes(b''.join(canonical(r)+b'\n' for r in rows));evidence['timing']={'path':str(path),'sha256':digest(path.read_bytes())}
    blob=j._path('req-1','intent').read_bytes();doc={'schema':m.SCHEMA,'policy':m.POLICY,'identity':identity,
        'request_id':'req-1','request_sha256':json.loads(blob)['request_sha256'],'intent_sha256':digest(blob),
        'verdict':'not_sent','evidence':evidence}
    return j,plan,doc,rows,docs


def test_positive_nested_failure_is_not_sent_without_fabricated_response(tmp_path):
    j,plan,doc,rows,docs=fixture(tmp_path)
    blob=j._path('req-1','intent').read_bytes()
    assert m.validate(doc,intent_blob=blob,identity=j.identity)['state']=='not_sent'
    assert j.inspect('req-1')['state']=='unknown'
    j._path('req-1','reconciliation').write_bytes(canonical(doc))
    assert j.inspect('req-1')['state']=='not_sent'
    assert not j._path('req-1','not_sent').exists() and not j._path('req-1','response').exists()
    replay=NativeContactReplay(tmp_path/'data',plan,replace(plan,attempt_id='recovery'), 'bare-core')
    replay.assert_all_contacts_resolved()
    assert replay.source_actual_usage['calls']==1
    assert 'delivery_custody' in replay.source_binding()
    Path(doc['evidence']['timing']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError):replay.assert_all_contacts_resolved()
    with pytest.raises(ValueError):j.inspect('req-1')


@pytest.mark.parametrize('fault',['no_failure','http','duplicate','thread','sample','pid','nonfinite','parent','outside','torn','source','active','launch','config','request'])
def test_ambiguous_or_unbound_evidence_is_refused(tmp_path,fault):
    j,plan,doc,rows,docs=fixture(tmp_path)
    if fault=='no_failure':rows[-1]['status']='ok'
    elif fault=='http':rows.append({**rows[-1],'stage':'http_request'})
    elif fault=='duplicate':rows.append(rows[2])
    elif fault=='thread':rows[-1]['thread']=2
    elif fault=='sample':rows[-1]['sample_id']='other'
    elif fault=='pid':rows[-1]['pid']=2
    elif fault=='nonfinite':rows[-1]['elapsed_seconds']=-1
    elif fault=='parent':rows[-1]['parent_span_id']='wrong'
    elif fault=='outside':rows[-1]['started_monotonic']=50
    elif fault=='source':docs['source_manifest']['files'][m.PREFIX+'core/model/transport.py']='0'*64
    elif fault=='active':docs['container'][0]['State']['Running']=True
    elif fault=='launch':docs['container'][0]['Config']['Labels']['saea.launch.config']='0'*64
    elif fault=='config':docs['timing_config']['effective']['client_detail']=False
    elif fault=='request':doc['request_id']='other'
    raw=b''.join(canonical(r)+b'\n' for r in rows)
    if fault=='torn':raw=raw[:-1]
    for name,value in docs.items():
        p=Path(doc['evidence'][name]['path']);p.write_bytes(canonical(value));doc['evidence'][name]['sha256']=digest(p.read_bytes())
    p=Path(doc['evidence']['timing']['path']);p.write_bytes(raw);doc['evidence']['timing']['sha256']=digest(raw)
    with pytest.raises(ValueError):m.validate(doc,intent_blob=j._path('req-1','intent').read_bytes(),identity=j.identity)


def test_reconciliation_cannot_replace_dispatch_or_response(tmp_path):
    j,plan,doc,*_=fixture(tmp_path)
    j.delivery_event('req-1',doc['request_sha256'],'dispatch')
    j._path('req-1','reconciliation').write_bytes(canonical(doc))
    with pytest.raises(ValueError):j.inspect('req-1')


def publishing_config(tmp_path):
    j,plan,doc,rows,docs=fixture(tmp_path)
    locks=tmp_path/'locks';locks.mkdir();(locks/'generation.guard').touch()
    (locks/'flight.lock').write_bytes(canonical(docs['lease']))
    from dataclasses import asdict
    pp=tmp_path/'plan.json';pp.write_bytes(canonical({'execution_plan':asdict(plan)}))
    cfg={'schema':'saea-pre-dispatch-reconciliation-config-v1','data':str(tmp_path/'data'),
         'plan':{'path':str(pp),'sha256':digest(pp.read_bytes())},'arm':'bare-core',
         'request_ids':['req-1'],'evidence':doc['evidence'],'lock_directory':str(locks)}
    return j,cfg,docs


@pytest.mark.parametrize('fault',[None,'running','changed_lease','changed_attempt'])
def test_publish_only_after_exact_terminal_custody(tmp_path,monkeypatch,fault):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/environment'))
    import reconcile_pre_dispatch_failure as cli
    import subprocess
    j,cfg,docs=publishing_config(tmp_path)
    current=copy.deepcopy(docs['container'])
    if fault=='running':current[0]['State']['Running']=True
    elif fault=='changed_lease':Path(cfg['lock_directory'],'flight.lock').write_text('{}')
    elif fault=='changed_attempt':
        next((tmp_path/'data/attempts').glob('*/*.json')).write_text('{}')
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:canonical(current))
    before=Path(cfg['lock_directory'],'flight.lock').read_bytes()
    if fault:
        with pytest.raises(ValueError):cli.execute(cfg,publish=True)
        assert not j._path('req-1','reconciliation').exists()
    else:
        assert cli.execute(cfg,publish=False)['published'] is False
        assert cli.execute(cfg,publish=True)['published'] is True
        assert j.inspect('req-1')['state']=='not_sent'
        with pytest.raises(ValueError):cli.execute(cfg,publish=True)
    assert Path(cfg['lock_directory'],'flight.lock').read_bytes()==before
