"""Offline CPU fixtures. Real receipts are input data only; all state is temporary."""
import ast
import contextlib
import copy
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import terminal_handoff as h
import host_driver as driver

HERE = Path(__file__).resolve().parent
FIX = HERE.parent / 'fixtures'


def dump(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


class Case:
    def __init__(self, root, locks):
        self.root = Path(root)
        self.locks = Path(locks)
        self.guard = self.locks / 'generation.guard'
        self.guard.touch()
        real = h.read(FIX / 'D-FAILED-PRESERVED.json')
        self.ev = copy.deepcopy(real)
        launch = self.ev['launch']
        # Fixture-only path/process rebinding; original real failure/cost stays exact.
        self.failed = self.root / 'jobs' / launch['task_id']
        launch['state'] = str(self.failed)
        launch['pid'] = 2147483647  # kernel pid_max is far lower; dead() still probes.
        self.rt = launch['runtime']
        self.rt.update(shared_guard=str(self.guard), shared_guard_inode=self.guard.stat().st_ino,
                       boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        self.cfg = dict(run_id='independent', attempt_id='v1')
        self.task = dict(task_id='new', cards=[4,5,6,7], dependencies=[])
        dump(self.failed / 'LAUNCH.json', launch)
        dump(self.failed / 'run/TERMINAL.json', self.ev['terminal'])
        dump(self.failed / 'runtime-config.json', dict(run_id=launch['run_id'], attempt_id=launch['attempt_id'], runtime=self.rt))
        # Real flight SHA is known but source bytes are absent locally. This is
        # explicitly a schema fixture, never represented as the original flight.
        self.flight = self.locks / 'flight.lock'
        dump(self.flight, dict(schema='saea-generation-lease-v1', run_id=launch['run_id'],
             attempt_id=launch['attempt_id'], session_id=self.rt['session_id'], study_head=self.rt['study_head'],
             owner='s-correct-p-data-v1', run_ids=[launch['run_id']], pid=launch['pid'], token='f'*64))
        journal = self.failed / 'run/contact-journals' / launch['run_id'] / launch['attempt_id']
        dump(journal / 'unknown.intent.json', dict(fixture=True, request_id='unknown-do-not-send'))
        self.policy = dict(schema=h.SCHEMA, enabled=True, allowlist={'new':[launch['task_id']]})
        self.container = dict(Id=launch['container_id'], State=dict(Status='exited', Running=False, Pid=0))
        self.metric = dict(running=0, waiting=0)
        self.service = dict(id=self.rt['service_id'])
        self.before = self.inventory()

    def inventory(self):
        return {str(p.relative_to(self.failed)):p.read_bytes() for p in self.failed.rglob('*') if p.is_file()}

    def verify(self, rt, cards):
        h.need(rt['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'boot changed')
        h.need(rt['service_id'] == self.service['id'], 'service changed')
        h.need(rt['shared_guard_inode'] == self.guard.stat().st_ino, 'guard changed')

    def replace_guard(self):
        # Allocate while the old path still exists: unlink+touch may reuse its
        # inode before execute() opens it, so it does not guarantee a mutation.
        before = self.guard.stat()
        replacement = self.guard.with_name('replacement.guard')
        with replacement.open('xb'):
            pass
        new = replacement.stat()
        assert new.st_dev == before.st_dev and new.st_ino != before.st_ino
        os.replace(replacement, self.guard)
        assert (self.guard.stat().st_dev, self.guard.stat().st_ino) == (new.st_dev, new.st_ino)

    def execute(self, held=None):
        if held is not None:
            return h.reconcile(self.root,self.task,self.cfg,self.rt,self.ev,self.policy,held,
                               self.verify,lambda rt:self.metric,lambda cid:self.container)
        with self.guard.open('r+b') as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self.execute(held)


class HandoffTests(unittest.TestCase):
    def case(self, root, locks=None):
        return Case(root, locks or root)

    def test_real_receipts_are_data_only(self):
        real = h.read(FIX/'D-FAILED-PRESERVED.json')
        self.assertEqual((real['terminal']['status'],real['terminal']['unknown'],real['terminal']['completed']),('FAILED',1,1103))
        self.assertEqual(real['terminal']['usage']['total_tokens'],6800582)
        result = json.loads(h.read(FIX/'D-SEED08-TIMEOUT-RECONCILIATION.json')['stdout'])
        self.assertEqual(result['unknown_preserved'],1)
        self.assertFalse(result['unknown_replayed'])
        self.assertIn('Invalid cross-device link',h.read(FIX/'D-SEED08-RECONCILIATION-ATTEMPT1.json')['stderr'])

    def test_success_preserves_unknown_and_cost_validates_native(self):
        with tempfile.TemporaryDirectory() as root:
            c=self.case(root);inode=c.guard.stat().st_ino
            receipt=c.execute()
            self.assertFalse(c.flight.exists())
            self.assertEqual(c.guard.stat().st_ino,inode)
            self.assertEqual(c.inventory(),c.before)
            pred=dict(**c.ev['terminal'], container=c.ev['launch']['container_id'],container_state=c.container['State'],terminal={'fixture':True})
            runtime=dict(c.rt,terminal_handoff_receipt=receipt)
            h.validate_predecessor(pred,runtime,c.cfg)
            self.assertEqual(h.read(receipt['path'])['terminal']['unknown'],1)
            for cfg in [dict(c.cfg,run_id=c.ev['launch']['run_id']),dict(c.cfg,attempt_id='wrong')]:
                with self.assertRaises(RuntimeError):h.validate_predecessor(pred,runtime,cfg)
            with self.assertRaises(RuntimeError):h.validate_predecessor(pred,dict(runtime,service_id='changed'),c.cfg)
            archive=Path(receipt['path']).parent/'flight.lock';archive.write_text('corrupt')
            with self.assertRaises(RuntimeError):h.validate_predecessor(pred,runtime,c.cfg)

    def test_rejections_preserve_flight_and_source(self):
        mutations={
            'disabled':lambda c:c.policy.update(enabled=False),
            'no_allowlist':lambda c:c.policy.update(allowlist={}),
            'wildcard':lambda c:c.policy.update(allowlist={'*':['*']}),
            'same_task':lambda c:c.task.update(task_id=c.ev['launch']['task_id']),
            'dependent':lambda c:c.task.update(dependencies=[c.ev['launch']['task_id']]),
            'same_run':lambda c:c.cfg.update(run_id=c.ev['launch']['run_id']),
            'unknown_terminal':lambda c:c.ev['terminal'].update(status='RUNNING'),
            'changed_terminal':lambda c:c.ev['terminal'].update(unknown=0),
            'changed_boot':lambda c:c.rt.update(boot_id='other'),
            'changed_service':lambda c:c.service.update(id='other'),
            'live_container':lambda c:c.container['State'].update(Status='running',Running=True,Pid=12),
            'busy':lambda c:c.metric.update(running=1),
            'waiting':lambda c:c.metric.update(waiting=1),
            'transition':lambda c:(c.locks/'engine-transition.pending.json').touch(),
            'wrong_flight_run':lambda c:dump(c.flight,dict(h.read(c.flight),run_id='other')),
            'live_flight_owner':lambda c:dump(c.flight,dict(h.read(c.flight),pid=os.getpid())),
            'live_runner':lambda c:c.ev['launch'].update(pid=os.getpid()),
            'changed_guard':lambda c:c.replace_guard(),
        }
        for label,mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as root:
                c=self.case(root);mutate(c);raw=c.flight.read_bytes()
                with self.assertRaises((RuntimeError,PermissionError)):c.execute()
                self.assertEqual(c.flight.read_bytes(),raw)
                self.assertEqual(c.inventory(),c.before)
                if label=='changed_guard':self.assertFalse((c.root/'reconciliation').exists())

        # Exercise the real caller lifetime too: the locked original descriptor
        # remains open even if its pathname is replaced before the first check.
        with self.subTest(label='changed_guard_while_held'), tempfile.TemporaryDirectory() as root:
            c=self.case(root);raw=c.flight.read_bytes()
            with c.guard.open('r+b') as held:
                fcntl.flock(held,fcntl.LOCK_EX | fcntl.LOCK_NB)
                c.replace_guard()
                self.assertEqual(os.fstat(held.fileno()).st_ino,c.rt['shared_guard_inode'])
                self.assertNotEqual(os.fstat(held.fileno()).st_ino,c.guard.stat().st_ino)
                # Bypass the fixture's identity probe to exercise reconcile's
                # own held-descriptor/path identity check directly.
                c.verify=lambda rt,cards:None
                with self.assertRaisesRegex(RuntimeError,'guard (?:inode )?changed'):c.execute(held)
            self.assertEqual(c.flight.read_bytes(),raw)
            self.assertEqual(c.inventory(),c.before)
            self.assertFalse((c.root/'reconciliation').exists())

    def test_live_runner_matching_launch_and_config_refused(self):
        with tempfile.TemporaryDirectory() as root:
            c=self.case(root);c.ev['launch']['pid']=os.getpid();dump(c.failed/'LAUNCH.json',c.ev['launch'])
            dump(c.flight,dict(h.read(c.flight),pid=os.getpid()))
            with self.assertRaisesRegex(RuntimeError,'live runner'):c.execute()

    def test_not_held_guard(self):
        with tempfile.TemporaryDirectory() as root:
            c=self.case(root)
            with c.guard.open('r+b') as held, self.assertRaisesRegex(RuntimeError,'guard not held'):
                c.execute(held)

    def test_busy_guard_refused(self):
        with tempfile.TemporaryDirectory() as root:
            c=self.case(root)
            with c.guard.open('r+b') as held:
                fcntl.flock(held,fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):c.execute()
            self.assertTrue(c.flight.exists())

    def test_no_flight_known_failure_and_paused_still_audited(self):
        for status,unknown in [('PAUSED',0),('FAILED',1)]:
            with self.subTest(status=status),tempfile.TemporaryDirectory() as root:
                c=self.case(root);c.flight.unlink()
                c.ev['terminal'].update(status=status,unknown=unknown)
                dump(c.failed/'run/TERMINAL.json',c.ev['terminal']);c.before=c.inventory()
                receipt=c.execute();doc=h.read(receipt['path'])
                self.assertFalse(doc['flight_present']);self.assertIsNone(doc['flight_owner_pid'])
                self.assertNotIn('flight.lock',doc['archive']);self.assertIn('NO-FLIGHT.json',doc['archive'])
                pred=dict(**c.ev['terminal'],container=c.ev['launch']['container_id'],container_state=c.container['State'],terminal={})
                h.validate_predecessor(pred,dict(c.rt,terminal_handoff_receipt=receipt),c.cfg)
                self.assertEqual(c.inventory(),c.before)
                with self.assertRaisesRegex(RuntimeError,'partial/existing archive'):c.execute()

    def test_no_flight_busy_live_boot_and_appearing_metadata_refused(self):
        for kind in ['busy','live','boot','appeared']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as root:
                c=self.case(root);c.flight.unlink();real=h.json_new
                if kind=='busy':c.metric['running']=1
                if kind=='boot':c.rt['boot_id']='other'
                if kind=='live':
                    c.ev['launch']['pid']=os.getpid();dump(c.failed/'LAUNCH.json',c.ev['launch'])
                def change(path,doc):
                    real(path,doc)
                    if kind=='appeared' and Path(path).name=='PREPARED.json':dump(c.flight,{'unexpected':True})
                with patch.object(h,'json_new',change),self.assertRaises(RuntimeError):c.execute()

    def test_second_check_changes_refused(self):
        for kind in ['boot','service','busy','flight','terminal','guard']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as root:
                c=self.case(root);real=h.json_new;flight_before=c.flight.read_bytes()
                def change(path,doc):
                    real(path,doc)
                    if Path(path).name=='PREPARED.json':
                        if kind=='boot':c.rt['boot_id']='changed'
                        if kind=='service':c.service['id']='changed'
                        if kind=='busy':c.metric['waiting']=1
                        if kind=='flight':c.flight.write_bytes(c.flight.read_bytes()+b' ')
                        if kind=='terminal':(c.failed/'run/TERMINAL.json').write_text('{}')
                        if kind=='guard':
                            c.replace_guard()
                            c.verify=lambda rt,cards:None
                with patch.object(h,'json_new',change),self.assertRaises(RuntimeError):c.execute()
                self.assertTrue(c.flight.exists())
                if kind=='guard':
                    self.assertEqual(c.flight.read_bytes(),flight_before)
                    self.assertEqual(c.inventory(),c.before)
                    self.assertFalse(list((c.root/'reconciliation').rglob('RECEIPT.json')))

    def test_partial_archive_write_hash_fsync_and_no_retry(self):
        for kind in ['write','hash','fsync','after_unlink']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as root:
                c=self.case(root);real=h.write_new
                def broken(path,raw):
                    name=Path(path).name
                    if name==('RECEIPT.json' if kind=='after_unlink' else 'flight.lock'):
                        if kind=='hash':return real(path,b'corrupted')
                        if kind=='fsync':
                            with patch.object(h.os,'fsync',side_effect=OSError('fixture fsync failure')):return real(path,raw)
                        Path(path).write_bytes(raw[:3]);raise OSError('fixture interrupted archive')
                    return real(path,raw)
                with patch.object(h,'write_new',broken),self.assertRaises((OSError,RuntimeError)):c.execute()
                self.assertEqual(c.flight.exists(),kind!='after_unlink')
                self.assertEqual(c.inventory(),c.before)
                with self.assertRaises((RuntimeError,FileNotFoundError)):c.execute()

    def test_actual_cross_device_copy(self):
        self.assertTrue(Path('/dev/shm').is_dir(),'CPU test requires /dev/shm for actual cross-device case')
        with tempfile.TemporaryDirectory(dir='/dev/shm') as root,tempfile.TemporaryDirectory() as locks:
            c=self.case(root,locks)
            self.assertNotEqual(Path(root).stat().st_dev,Path(locks).stat().st_dev)
            with patch.object(h.os,'rename',side_effect=AssertionError('no rename allowed')):
                receipt=c.execute()
            self.assertTrue(Path(receipt['path']).exists())
            self.assertEqual(c.inventory(),c.before)

    def test_complete_native_path_needs_no_handoff(self):
        h.validate_predecessor(dict(status='COMPLETE',unknown=0),{}, {})
        with self.assertRaises((RuntimeError,KeyError)):
            h.validate_predecessor(dict(status='COMPLETE',unknown=1),{}, {})

    def test_metrics_missing_or_nan_not_idle(self):
        for data in [b'',b'vllm:num_requests_running{x="y"} 0\n',b'vllm:num_requests_running{x="y"} NaN\nvllm:num_requests_waiting{x="y"} 0\n']:
            with patch.object(driver.urllib.request,'urlopen',return_value=SimpleNamespace(read=lambda:data)):
                with self.assertRaises(AssertionError):driver.metrics(dict(endpoint='fixture'))

    def test_existing_runtime_policy_launch_fixtures(self):
        import test_runtime_policy
        with contextlib.redirect_stdout(io.StringIO()):test_runtime_policy.main()

    def test_host_driver_failed_ace_subset_to_native_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            c=self.case(root);H=Path(root);W=H/'workspace';W.mkdir()
            cfg=dict(c.cfg,concurrency=64,timeout_seconds=600,expected_rows=9,manifest_sha256='fixed',execution_subset={'fixture':True})
            dump(W/'config.json',cfg)
            dump(W/'budget.json',dict(status='PASS_FIXED_IMAGE_CPU',rows=9,manifest_sha256='fixed'))
            runner=HERE/'native-runtime/run_s_native.py'
            (H/'handoff-native.py').write_bytes(runner.read_bytes())
            dump(H/'terminal-handoff-policy.json',dict(c.policy,native_runner_sha256=driver.sha(runner)))
            task=dict(c.task,workspace=str(W),config='config.json',budget='budget.json',runtime=c.rt,
                      predecessor=dict(container=c.ev['launch']['container_id'],terminal=str(c.failed/'run/TERMINAL.json')),
                      failed_predecessor_evidence=c.ev,window_end_epoch=time.time()+100,
                      runner='ACE/run_subset.py',bind_script='ACE/bind_subset.py')
            dump(H/'spec.json',task);calls=[]
            def hist(path):
                if str(path).endswith('/session.json'):return dict(session_id=c.rt['session_id'],pid=c.rt['service_pid'])
                return h.read(path)
            def inspect(cid):
                return c.container if cid==c.container['Id'] else dict(Id='new-container',State=dict(Pid=22))
            def run(argv,**kw):
                calls.append(argv);return SimpleNamespace(returncode=0,stdout='new-container',stderr='')
            args=SimpleNamespace(root=str(H),spec=str(H/'spec.json'),lane='D')
            with patch.object(driver,'historic',hist),patch.object(driver,'historical_pin',h.pin), \
                 patch.object(driver,'verify',c.verify),patch.object(driver,'metrics',lambda rt:c.metric), \
                 patch.object(driver,'inspect',inspect),patch.object(driver.subprocess,'run',run), \
                 contextlib.redirect_stdout(io.StringIO()):
                state=H/'jobs/new'
                dump(state/'launch-intent.json',{'unknown':True})
                with self.assertRaisesRegex(AssertionError,'UNKNOWN_LAUNCH_NO_RESEND'):driver.launch(args)
                self.assertTrue(c.flight.exists());self.assertFalse(calls)
                (state/'launch-intent.json').unlink() # temporary fixture only
                driver.launch(args)
                driver.launch(args) # known receipt never relaunches
            self.assertEqual(len(calls),2)
            self.assertTrue(any(a.endswith('ACE/bind_subset.py') for a in calls[0]))
            self.assertTrue(any(a.endswith('ACE/run_subset.py') for a in calls[1]))
            self.assertIn('PYTHONPATH='+str(H),calls[1])
            self.assertIn(f'type=bind,src={H}/handoff-native.py,dst={driver.M}/delivery/s-native-throughput-v1/code/run_s_native.py,readonly',calls[1])
            effective=h.read(state/'runtime-config.json')
            self.assertEqual(effective['execution_subset'],cfg['execution_subset'])
            self.assertEqual(effective['timeout_seconds'],600)
            h.validate_predecessor(h.read(state/'predecessor.json'),effective['runtime'],effective)
            self.assertEqual(c.inventory(),c.before)

    def test_native_edits_only_replace_predecessor_check(self):
        root=HERE.parents[2]
        pairs=[(root/'delivery/s-native-throughput-v1/code/run_s_native.py',HERE/'native-runtime/run_s_native.py'),
               (root/'delivery/s-single-intent-execution-v1/rolling-code/train-runtime-v1/run_s_native.py',HERE/'train-runtime-v1/run_s_native.py')]
        old="    require(pred['unknown']==0 and pred['status']=='COMPLETE','predecessor not COMPLETE')"
        new="    if pred.get('status') != 'COMPLETE' or pred.get('unknown') != 0:\n        from terminal_handoff import validate_predecessor\n        validate_predecessor(pred,runtime,c)"
        for original,candidate in pairs:
            self.assertEqual(candidate.read_text(),original.read_text().replace(old,new))
            compile(candidate.read_text(),str(candidate),'exec')

    def test_failed_train_uses_separate_pinned_package(self):
        with tempfile.TemporaryDirectory() as root:
            H=Path(root);rels=['train_runtime.py','bind_train.py','train-runtime-v1/run_s_native.py','train-runtime-v1/native_resources.py']
            for dirname,receipt in [('execution-code','TRAIN96-VERIFIED.json'),('handoff-execution-code','HANDOFF-TRAIN-VERIFIED.json')]:
                pins={}
                for rel in rels:
                    p=H/dirname/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(dirname);pins[rel]=driver.sha(p)
                dump(H/receipt,dict(pins=pins))
            for concurrency in [64,96,128]:
                cfg=dict(dataset_partition='eligible_remainder',concurrency=64,timeout_seconds=600,attempt_id='v1')
                override,_=driver.configure_train_execution(H,cfg,dict(task_id='new'),{},dict(tasks={'new':dict(concurrency=concurrency)}),failed_handoff=True)
                self.assertEqual(override['execution_code'],str(H/'handoff-execution-code'))
                self.assertEqual(cfg['attempt_id'],'v1-c'+str(concurrency))
            (H/'handoff-execution-code/train_runtime.py').write_text('bad')
            with self.assertRaisesRegex(AssertionError,'fixed execution package pin'):
                driver.configure_train_execution(H,dict(dataset_partition='eligible_remainder',concurrency=64,timeout_seconds=600,attempt_id='v1'),dict(task_id='new'),{}, {},failed_handoff=True)


def scheduler_functions(names):
    tree=ast.parse((HERE/'scheduler.py').read_text())
    return compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(HERE/'scheduler.py'),'exec')


class SchedulerTests(unittest.TestCase):
    def exercise(self, resume=None, terminal='FAILED'):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);events=[];window={'end_epoch':time.time()+100}
            initial=dict(predecessor=dict(container='historical',terminal='/historical/TERMINAL.json'),runtime={'same':True})
            def launch(tid):return dict(task_id=tid,container_id=tid,state='/jobs/'+tid,runtime=initial['runtime'],run_id=tid,attempt_id='v1',pid=999)
            def read(path):
                if str(path).endswith('initial-lanes.json'):return {'D':initial}
                if str(path).endswith('-CURRENT.json'):return dict(status=resume,launch=launch('failed'),terminal=dict(status=terminal,unknown=1))
                return dict(task=dict(task_id='failed'))
            if resume:(root/'lanes').mkdir();(root/'lanes/D-CURRENT.json').touch()
            chosen=iter([dict(task_id='failed'),dict(task_id='independent',runner='ACE/run_subset.py',bind_script='ACE/bind_subset.py')] if resume is None else [dict(task_id='independent',runner='ACE/run_subset.py',bind_script='ACE/bind_subset.py')])
            def wait(lane,la):
                if la['task_id']=='failed':return dict(status=terminal,unknown=1),dict(exit_code=1)
                window['end_epoch']=0;return dict(status='COMPLETE',unknown=0),dict(exit_code=0)
            class Thread:
                def __init__(self,*a,**kw):pass
                def start(self):pass
            ns=dict(HOSTS={'D':'host'},ROOTS={'host':'/hot'},O=root,WINDOW=window,time=time,read=read,
                save=lambda p,d:events.append(('save',str(p),copy.deepcopy(d))),choose=lambda lane:next(chosen),
                stage=lambda *a:'/workspace',launch_command=lambda *a:[],run=lambda *a,**kw:SimpleNamespace(stdout=''),
                remote_json=lambda host,args:launch('independent' if 'independent' in args[-1] else 'failed'),
                wait_container=wait,update=lambda t,**kw:events.append(('update',t['task_id'],kw)),score=lambda *a:None,
                subprocess=SimpleNamespace(run=lambda *a,**kw:None),threading=SimpleNamespace(Thread=Thread),
                hashlib=hashlib,traceback=__import__('traceback'),wake=threading.Event())
            exec(scheduler_functions({'lane_worker','failed_handoff'}),ns);ns['lane_worker']('D')
            return events

    def test_failure_reaches_independent_successor_not_historical(self):
        for resume in (None,'running','failed'):
            with self.subTest(resume=resume):
                events=self.exercise(resume)
                self.assertFalse(any(e[0]=='save' and e[1].endswith('-BLOCKED.json') for e in events),events)
                specs=[e[2] for e in events if e[0]=='save' and '/specs/' in e[1]]
                spec=specs[-1]
                self.assertEqual(spec['predecessor']['container'],'failed')
                self.assertEqual(spec['failed_predecessor_evidence']['terminal'],dict(status='FAILED',unknown=1))
                self.assertEqual(spec['runner'],'ACE/run_subset.py')
                self.assertEqual(spec['bind_script'],'ACE/bind_subset.py')

    def test_unknown_terminal_blocks_no_fallback(self):
        events=self.exercise(terminal='RUNNING')
        self.assertTrue(any(e[0]=='save' and e[1].endswith('-BLOCKED.json') for e in events))
        self.assertFalse(any(e[0]=='save' and '/specs/independent' in e[1] for e in events))

    def test_wait_complete_and_known_failed_unknown_return_unchanged(self):
        for terminal in [dict(status='COMPLETE',unknown=0),dict(status='FAILED',unknown=1)]:
            ns=dict(HOSTS={'D':'fake'},WINDOW={'end_epoch':time.time()+100},threading=threading,time=time,
                wake=threading.Event(),run=lambda *a,**kw:SimpleNamespace(stdout='0'),
                snapshot=lambda *a:dict(terminal=terminal))
            exec(scheduler_functions({'wait_container','observation_due'}),ns)
            self.assertEqual(ns['wait_container']('D',dict(container_id='old',state='/unused'))[0],terminal)

    def test_ace_pilot_excluded_from_capacity_estimate(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);O=root/'state';P=root/'producer';(O/'tasks').mkdir(parents=True)
            for tid,branches,eligible in [('full',1635,True),('pilot',9,False)]:
                dump(O/'tasks'/(tid+'.json'),dict(status='ready',task=dict(task_id=tid,branches=branches,config=tid+'.json',capacity_estimate_eligible=eligible)))
                dump(root/(tid+'.json'),dict(expected_rows=branches))
            ns=dict(O=O,P=P,R=root,WINDOW={'end_epoch':time.time()+100},time=time,read=h.read,save=lambda *a:None,tasks=lambda:[])
            exec(scheduler_functions({'reserve'}),ns);result=ns['reserve']()
            self.assertEqual(result['ready_branches'],1644)
            self.assertEqual(result['capacity_estimate_branches'],1635)
            self.assertEqual(result['unknown_eta_task_ids'],['pilot'])
            self.assertEqual(result['estimated_ready_lane_hours_range'],[1635*3/3600,1635*5.5/3600])


if __name__=='__main__':
    # Accidental external activity must fail loudly in CPU fixtures.
    with patch('socket.socket',side_effect=AssertionError('network forbidden in CPU tests')), \
         patch('subprocess.Popen',side_effect=AssertionError('process launch forbidden in CPU tests')):
        unittest.main(verbosity=2)
