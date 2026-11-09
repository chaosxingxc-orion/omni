"""Isolated offline fixtures. Only writes inside this delivery; no live imports."""
import copy,hashlib,importlib.util,json,os,sys,tempfile,time,types,unittest
from pathlib import Path
from unittest.mock import Mock,patch
HERE=Path(__file__).resolve().parents[1]
REPO=HERE.parents[1]
PARENT='prepared-v5-dynamic/single_intent_fewshot'
def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(canonical(value)+b'\n')
def require(ok,reason):
    if not ok:raise ValueError(reason)
def load():
    common=types.ModuleType('common')
    for k in ('canonical','sha','read','save','require'):setattr(common,k,globals()[k])
    teacher=types.ModuleType('teacher_round');teacher.ROOT=REPO;teacher.OUT=REPO/'delivery/s-tfrl-rolling-v1'
    producer=types.ModuleType('producer')
    for name in ('prepare','verify','publish'):setattr(producer,name,Mock(side_effect=AssertionError('external producer forbidden')))
    with patch.dict(sys.modules,dict(common=common,teacher_round=teacher,producer=producer)):
        spec=importlib.util.spec_from_file_location('candidate_feedback_teacher',HERE/'code/feedback_teacher.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
f=load()
class Fixtures(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=HERE/'tests',prefix='.fixture-')
        self.root=Path(self.temp.name);self.out=self.root/'delivery/s-tfrl-rolling-v1'
        self.patches=[patch.object(f,'ROOT',self.root),patch.object(f,'OUT',self.out),patch.object(f.subprocess,'run',side_effect=AssertionError('no subprocess'))]
        for p in self.patches:p.start()
        for name in ('prepare','verify','publish'):getattr(f.producer,name).reset_mock()
        self.ops=read(HERE/'config/tfgrpo-operations-v1.json');save(self.out/'config/tfgrpo-operations-v1.json',self.ops)
        self.rev='revision-v5-fixture';self.folder=self.out/'teacher'/self.rev
        self.groups=[dict(sample_id='fixture-positive',positive=dict(advantage=1),negative=dict(advantage=-1)),dict(sample_id='fixture-counterexample')]
        self.context=dict(evidence=[dict(evidence_id='actual-complete-feedback',content=dict(representative_groups=self.groups))])
        save(self.folder/'dynamic-context.private.json',self.context)
        self.quota=self.out/'teacher/WINDOW-GROUP-REQUEST-QUOTA.json'
        save(self.quota,dict(revision=self.rev,consumed=True,max_new_requests=1,unknown_no_resend=True))
        save(self.folder/'launch-intent.json',dict(consumed=True,unknown=True))
        self.quota_before=self.quota.read_bytes();self.intent_before=(self.folder/'launch-intent.json').read_bytes()
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()
    def proposal(self,op='compare_evidence_for_competing_roles',pid='different-teacher-ID'):
        return dict(candidate_id=pid,parent_version=PARENT,mechanism='fixture grounded comparison',
            mechanism_family=self.ops['operations'][op]['mechanism_family'],patch=dict(target='instruction_append',value=dict(operation_id=op)),
            source_evidence_ids=['actual-complete-feedback'],semantic_advantage=dict(positive_sample_ids=['fixture-positive'],counterexample_sample_id='fixture-counterexample',explanation='Fixture positive vs negative relational role; counterexample limits applicability.'),
            counterevidence='fixture contrary',falsification='no paired gain',rollback='restore parent',applicability='fixture distinct roles')
    def existing(self,parent=PARENT,status='ready',instruction=None,name='candidate-role-evidence-v1'):
        instruction=instruction or self.ops['operations']['compare_evidence_for_competing_roles']['instruction']
        arm=parent.split('/')[1];folder=self.out/'candidates'/name
        base=self.root/'delivery/s-single-intent-fewshot-v1'/parent
        packet=dict(wire=dict(messages=[dict(content='parent instruction'),dict(content=[dict(text='parent instruction\n\nCandidate-conditioned pipeline context:fixed')])],seed=7,response_format={'type':'json_schema'},audio='fixed fixture'))
        prior=dict(sample_id='branch-1',abstract_wire_sha256=sha(canonical(packet['wire'])))
        mp=base/'manifest.private.jsonl';save(mp,prior)
        save(base/'native-UNBOUND.json',dict(manifest=str(mp.relative_to(self.root)),manifest_sha256=sha(mp.read_bytes())))
        addition='\n\n'+instruction;packet['wire']['messages'][0]['content']+=addition
        packet['wire']['messages'][1]['content'][-1]['text']='parent instruction'+addition+'\n\nCandidate-conditioned pipeline context:fixed'
        payload=folder/arm/'packet.json';save(payload,packet)
        manifest=folder/arm/'manifest.private.jsonl';save(manifest,dict(sample_id='branch-1',payload=str(payload.relative_to(self.root)),packet_sha256=sha(payload.read_bytes())))
        config=folder/arm/'native-UNBOUND.json';save(config,dict(manifest=str(manifest.relative_to(self.root)),manifest_sha256=sha(manifest.read_bytes())))
        save(folder/'experience.json',dict(parent_version=parent,instruction=instruction))
        save(folder/'CPU-PREPARED.json',dict(parent=parent,arm=arm,policy_type='single mechanism instruction'))
        task=dict(task_id=name,parent_version=parent,status=status,prepared=str(folder.relative_to(self.root)),config=str(config.relative_to(self.root)))
        ready=self.out/'READY-TASKS.json';doc=read(ready) if ready.exists() else dict(tasks=[]);doc['tasks'].append(task);save(ready,doc)
        save(self.out/'config'/('teacher-compiled-'+name+'.json'),dict(provenance='main-agent hypothesis; not external teacher TFGRPO update'))
        return task
    def compile(self,proposals):
        save(self.folder/'teacher-final.json',dict(curator=dict(proposals=proposals,experience_entries=[dict(semantic_advantage='fixture analysis')],retirements=[])))
        f.compile_and_prepare(self.rev)
        self.assertEqual(self.quota.read_bytes(),self.quota_before)
        self.assertEqual((self.folder/'launch-intent.json').read_bytes(),self.intent_before)
        return read(self.folder/'LOCAL-REVIEW.json')
    def no_producer(self):
        for name in ('prepare','verify','publish'):self.assertEqual(getattr(f.producer,name).call_count,0)
    def test_existing_candidate_different_proposal_IDs_no_new_tasks(self):
        self.existing();before=(self.out/'READY-TASKS.json').read_bytes()
        proposals=[self.proposal(pid='new-ID-A'),self.proposal(pid='new-ID-B')];r=self.compile(proposals)
        self.assertEqual([d['status'] for d in r['decisions']],['REUSE_EXISTING_POLICY']*2)
        self.assertEqual((self.out/'READY-TASKS.json').read_bytes(),before);self.no_producer()
        self.assertEqual(r['novelty'],'no-novel-candidate')
        self.assertEqual(len(list((self.out/'candidates').iterdir())),1)
        for d,p in zip(r['decisions'],proposals):
            self.assertEqual(d['TFGRPO_learning']['semantic_advantage'],p['semantic_advantage'])
            self.assertEqual(d['policy_reference']['task_id'],'candidate-role-evidence-v1')
            self.assertFalse(d['policy_reference']['new_teacher_policy'])
            self.assertIn('main-agent',d['policy_reference']['provenance'])
        prior=(self.folder/'LOCAL-REVIEW.json').read_bytes();f.compile_and_prepare(self.rev)
        self.assertEqual((self.folder/'LOCAL-REVIEW.json').read_bytes(),prior)
    def test_different_parent_not_reused_but_novel_branch_still_works(self):
        self.existing(parent='prepared-v5-dynamic/single_intent_no_examples')
        self.assertNotEqual(f.instruction_identity(PARENT,'same'),f.instruction_identity('other','same'))
        with patch.object(f.producer,'prepare',return_value=None),patch.object(f.producer,'verify',return_value=None),patch.object(f.producer,'publish',return_value=None):
            r=self.compile([self.proposal()]);self.assertEqual(r['decisions'][0]['status'],'CPU_READY_NOT_PROVEN_GAIN')
            self.assertEqual(f.producer.prepare.call_count,1)
    def test_different_operation_ID_same_actual_instruction_reuses(self):
        self.existing();alias='reviewed_alias_fixture'
        self.ops['operations'][alias]=copy.deepcopy(self.ops['operations']['compare_evidence_for_competing_roles'])
        save(self.out/'config/tfgrpo-operations-v1.json',self.ops)
        r=self.compile([self.proposal(alias,pid='alias-proposal')])
        self.assertEqual(r['decisions'][0]['status'],'REUSE_EXISTING_POLICY');self.no_producer()
    def test_wrong_parent_proposal_fails_admission(self):
        self.existing();p=self.proposal();p['parent_version']='prepared-v5-dynamic/single_intent_no_examples'
        r=self.compile([p]);self.assertEqual(r['decisions'][0]['status'],'NO_ADMISSION');self.no_producer()
    def test_invalid_group_support_cannot_endorse_existing_policy(self):
        self.existing();p=self.proposal();p['semantic_advantage']['positive_sample_ids']=['foreign-group']
        r=self.compile([p]);self.assertEqual(r['decisions'][0]['status'],'NO_ADMISSION');self.no_producer()
    def test_retired_never_admitted_even_prepared(self):
        op='preserve_relational_direction';self.existing(instruction=self.ops['operations'][op]['instruction'],name='relational-direction-v1')
        r=self.compile([self.proposal(op)])
        self.assertEqual(r['decisions'][0]['status'],'NO_ADMISSION');self.assertIn('retired',str(r));self.no_producer()
    def test_preexisting_redundancy_no_admission(self):
        r=self.compile([self.proposal('separate_example_structure_from_values')])
        self.assertEqual(r['decisions'][0]['status'],'NO_ADMISSION');self.no_producer()
    def test_zero_proposals_no_task_created(self):
        r=self.compile([]);self.assertEqual(r['novelty'],'no-novel-candidate')
        self.assertFalse((self.out/'candidates').exists());self.assertFalse((self.out/'READY-TASKS.json').exists());self.no_producer()
    def test_complete_result_attribution_and_pending_score(self):
        task=self.existing();state=self.root/'delivery/s-single-intent-execution-v1/rolling-state'
        current=dict(task=task,status='completed',terminal=dict(status='COMPLETE',completed=1635,total=1635,unknown=0),launch=dict(container_id='fixture-container'))
        save(state/'tasks/candidate-role-evidence-v1.json',current)
        record=f.policy_inventory()[0];self.assertEqual(f.policy_reference(record)['score_status'],'pending')
        metrics=dict(status='COMPLETE_TYPED_STRICT_OFFICIAL_SIDECAR',denominator=545,groups={'all':{'normalized':{'F1':.5}}})
        save(state/'scores'/('omni-s12-score-'+sha(b'fixture-container')[:12]+'.json'),dict(status='completed',stdout=json.dumps(metrics)))
        r=self.compile([self.proposal()]);ref=r['decisions'][0]['policy_reference']
        self.assertEqual(ref['status'],'complete');self.assertEqual(ref['learning_result']['groups'],metrics['groups'])
        self.assertEqual(ref['learning_result_owner_task_id'],task['task_id']);self.no_producer()
    def test_unknown_preserved_no_resend(self):
        task=self.existing();p=self.root/'delivery/s-single-intent-execution-v1/rolling-state/tasks/candidate-role-evidence-v1.json'
        save(p,dict(task=task,status='failed',reason='UNKNOWN_NO_RESEND',terminal=dict(status='FAILED',unknown=1)))
        before=p.read_bytes();r=self.compile([self.proposal()])
        self.assertEqual(r['decisions'][0]['policy_reference']['status'],'unknown_no_resend')
        self.assertEqual(p.read_bytes(),before);self.no_producer();self.assertFalse(f.dispatch_permission(self.folder,self.quota))
    def test_corrupt_actual_wire_fails_closed_without_duplicate(self):
        self.existing();r=f.policy_inventory()[0];packet=self.root/'delivery/s-tfrl-rolling-v1/candidates/candidate-role-evidence-v1/single_intent_fewshot/packet.json'
        d=read(packet);d['wire']['seed']=99;save(packet,d)
        result=self.compile([self.proposal()]);self.assertEqual(result['decisions'][0]['status'],'REUSE_CUSTODY_FAILED_NO_RESEND');self.no_producer()
    def test_teacher_input_reports_pending_rejected_reasons_and_analysis(self):
        self.existing();ctx=f.teacher_policy_context(self.ops)
        by={x['operation_id']:x for x in ctx['operations']}
        self.assertEqual(by['compare_evidence_for_competing_roles']['execution'],'reuse_existing')
        self.assertEqual(by['preserve_relational_direction']['execution'],'not_admissible')
        self.assertTrue(by['preserve_relational_direction']['reason']);self.assertIn('trajectories',ctx['analysis_required'])
    def test_prepared_unpublished_reuses_without_publish(self):
        self.existing();(self.out/'READY-TASKS.json').unlink()
        r=self.compile([self.proposal()])
        self.assertEqual(r['decisions'][0]['status'],'REUSE_EXISTING_POLICY')
        self.assertEqual(r['decisions'][0]['policy_reference']['execution_status'],'prepared')
        self.assertFalse((self.out/'READY-TASKS.json').exists());self.no_producer()
    def test_reference_resolves_pending_to_complete_without_new_task(self):
        task=self.existing();record=f.policy_inventory()[0]
        self.assertEqual(f.policy_reference(record)['status'],'pending')
        state=self.root/'delivery/s-single-intent-execution-v1/rolling-state/tasks/candidate-role-evidence-v1.json'
        save(state,dict(task=task,status='completed',terminal=dict(status='COMPLETE',completed=1,total=1,unknown=0)))
        self.assertEqual(f.policy_reference(record)['status'],'complete');self.no_producer()
    def test_prepared_teacher_message_contains_policy_history_and_independent_template(self):
        self.existing()
        for name in ('curator-v4-reviewed.json','reflector-v3-corrected.json'):save(self.out/'experience'/name,{})
        save(self.out/'config/grpo-v1.json',{'seeds':list(range(8))})
        instruction=self.root/'delivery/s-single-intent-fewshot-v1/config/single-intent.txt'
        instruction.parent.mkdir(parents=True,exist_ok=True);instruction.write_text('parent instruction')
        save(self.root/'delivery/s-single-intent-fewshot-v1/prepared-v5-dynamic/source-policy.json',{})
        for name in ('compact-json-v1','absent-slot-v1'):save(self.out/'candidates'/name/'experience.json',{})
        template=(HERE/'config/training-free-grpo-template-v1.md').read_text()
        (self.out/'config/training-free-grpo-template-v1.md').write_text(template)
        save(self.out/'teacher/revision-v4-parent-unit-correction/executor-config.private.json',dict(analysis_budget={}))
        source=self.out/'fixture-group.json';save(source,dict(representative_groups=self.groups))
        rev=f.prepare_feedback(source,read(source),'group');folder=self.out/'teacher'/rev
        msg=read(folder/'messages.private.json')[0];ctx=json.loads(msg['content'])
        self.assertEqual(ctx['evidence'][0]['content']['representative_groups'],self.groups)
        self.assertEqual(ctx['policy_history']['rejected_strategies'][0]['task_id'],'binding-role-first-card-v1')
        self.assertTrue(ctx['policy_history']['published_task_history'])
        self.assertIn('positive/negative', (folder/'template.md').read_text())
        self.assertIn(template,(folder/'template.md').read_text())
        self.assertFalse((folder/'launch-intent.json').exists())
        for rel,pin in read(folder/'REQUEST-PINS.json').items():self.assertEqual(sha((folder/rel).read_bytes()),pin)
        self.no_producer()
    def test_main_unknown_quota_restart_unchanged(self):
        save(self.out/'REQUESTS/QUEUE-DEFICIT.json',dict(window=dict(end_epoch=time.time()+3600)))
        source=self.root/'delivery/s-single-intent-execution-v1/rolling-state/group-feedback/REFLECTOR-NEXT-INPUT.private.json'
        save(source,dict(samples=545,group_size=8,cross_policy_pooled=False))
        class End(Exception):pass
        real_open=Path.open;guards=[]
        def fixture_open(path,*args,**kwargs):
            handle=real_open(path,*args,**kwargs)
            if path.name=='feedback-teacher.guard':guards.append(handle)
            return handle
        try:
            with patch.object(f,'group_context',return_value=dict(representative_groups=self.groups)),patch.object(f.time,'sleep',side_effect=End),patch.object(Path,'open',fixture_open):
                with self.assertRaises(End):f.main()
        finally:
            for guard in guards:guard.close()
        self.assertEqual(self.quota.read_bytes(),self.quota_before)
        self.assertEqual((self.folder/'launch-intent.json').read_bytes(),self.intent_before)
        self.no_producer()

if __name__=='__main__':unittest.main(verbosity=2)
