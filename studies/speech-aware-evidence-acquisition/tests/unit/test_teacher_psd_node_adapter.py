import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'analysis'))
import teacher_psd_node_adapter as adapter


def test_real_projections_preserve_full_pipeline_and_source():
    root = Path('/mnt/cephfs/xingchao/omni/work/codex/teacher-psd-candidate-integration-v1')
    if not root.exists() or not __import__('os').access(root/'specs/reflexion-decision-fold-a.json', __import__('os').R_OK):
        pytest.skip('real teacher fixture is on the shared research data root')
    repo = root / 'studies/speech-aware-evidence-acquisition'
    for name, node, algorithm in [('reflexion-decision','decision','reflexion'),('reflexion-surface','slot','reflexion'),('reflexion-rescan','slot','reflexion'),('ace-person','slot','ace')]:
        for fold in 'ab':
            spec = json.loads((root/'specs'/f'{name}-fold-{fold}.json').read_text())
            workload, receipt = adapter.compile_candidate(repo, Path('/mnt/cephfs/xingchao/omni'), spec)
            assert receipt['node'] == node and receipt['algorithm'] == algorithm
            assert receipt['historical_source_kind'].startswith('P-to-joint-Solver')
            assert not receipt['candidate_accepted'] and receipt['new_model_calls'] == 0
            assert len(workload['sample_ids']) == 1080
            assert workload['pipeline']['coverage_recovery']['max_extra_contacts'] == 3
            parent = json.loads(Path(spec['parent']['path']).read_text())
            assert {k:v for k,v in workload.items() if k not in ('name','run_id',node+'_policy')} == {k:v for k,v in parent.items() if k not in ('name','run_id',node+'_policy')}
            broken=copy.deepcopy(spec); broken['candidate_sha256']='0'*64
            with pytest.raises(ValueError, match='candidate mismatch'):
                adapter.compile_candidate(repo, Path('/mnt/cephfs/xingchao/omni'), broken)
            text=(repo/workload[node+'_policy']['instruction_template']['relpath']).read_text()
            assert ('Reflexion guidance' if algorithm == 'reflexion' else 'ACE conditional clarification') in text


def test_offline_wire_and_single_node_fixture(tmp_path, monkeypatch):
    import shutil
    study = Path(__file__).resolve().parents[2]
    repo = tmp_path/'repo'
    (repo/'scripts/environment').mkdir(parents=True)
    shutil.copy(study/'scripts/environment/configured_external_learner.py', repo/'scripts/environment/')
    (repo/'configs/probes').mkdir(parents=True)
    (repo/'configs/probes/base.txt').write_text('Decision: choose an existing candidate_id only.\n')
    proposal = {'id':'fixture','evidence_query_ids':['slurp:train:1'], 'hypothesis':'fixture',
        'memory':{'trigger':'compare','action':'compare operation and object','exception':'stay within candidates'},
        'retry_protocol':'fixed', 'transfer_scope':'cross_task_adaptation','validation':'compare'}
    proposals = {'algorithm':'reflexion','candidates':[proposal, {**proposal,'id':'fixture2'}]}
    profile = {'algorithm':'reflexion','min_candidates':2,'max_candidates':6}
    context = {}
    request = {'messages':[{'content':json.dumps({'algorithm':'reflexion','groups':[{'query_id':'slurp:train:1'}]})}]}
    from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha
    text = json.dumps(proposals)
    usage = {'prompt_tokens':10,'completion_tokens':2}
    events = [{'id':'response1','model':'teacher','choices':[{'index':0,'delta':{'content':text},'finish_reason':'stop'}]},
              {'id':'response1','model':'teacher','choices':[],'usage':usage}]
    wire = ''.join('data: '+json.dumps(e)+'\n\n' for e in events)+'data: [DONE]\n\n'
    wire_path=tmp_path/'wire.sse'; wire_path.write_text(wire)
    def put(name, value):
        p=tmp_path/name;p.write_text(json.dumps(value));return {'path':str(p),'sha256':adapter.digest(p)}
    prepared={'executor':{'sha256':adapter.digest(repo/'scripts/environment/configured_external_learner.py')},
              'source':{'sha256':'source'},'files':{}}
    for name,obj in [('profile.json',profile),('context.json',context),('request.json',request)]:
        prepared['files'][name]=put(name,obj)
    response={'finish_reason':'stop','request_sha256':sha(request),'wire_response_sha256':adapter.digest(wire_path),
              'provider_response_id':'response1','model_id':'teacher','provider_usage':usage,'text':text}
    collected={'proposals':proposals,'response':{'response':response},'request_sha256':sha(request)}
    assert adapter.verify_wire_proposals(repo,wire_path,collected,prepared,profile,context,request)==proposals
    altered=copy.deepcopy(collected);altered['proposals']['candidates'][0]['memory']['action']='invent'
    # A new self-signed collected digest cannot change the original SSE.
    altered_pin=put('altered.json',altered)
    assert altered_pin['sha256']==adapter.digest(altered_pin['path'])
    with pytest.raises(ValueError, match='proposals differ'):
        adapter.verify_wire_proposals(repo,wire_path,json.loads(Path(altered_pin['path']).read_text()),prepared,profile,context,request)
    altered=copy.deepcopy(collected);altered['response']['response']['text']=json.dumps(altered['proposals'])+' '
    with pytest.raises(ValueError, match='response differs'):
        adapter.verify_wire_proposals(repo,wire_path,altered,prepared,profile,context,request)

    # Exercise the real compiler and its complete-baseline comparison with a tiny
    # NodeWorkload stand-in; no private roster or Ceph is needed for this test.
    source_pin=put('source.json',{'document':{'model_id':'source-actor'}})
    prepared['source']=source_pin
    prep_pin=put('prepared.json',prepared)
    collected['prepared_sha256']=prep_pin['sha256']
    col_pin=put('collected.json',collected)
    parent={'node_arm':'enhanced-p-slot-decision','pipeline':{'protocol_version':'v2','coverage_recovery':{'max_extra_contacts':3}},
        'sample_ids':['id']*1080,'population':{'dataset_sha256':'fold'},'material':{}}
    parent_pin=put('parent.json',parent)
    audit_pin=put('audit.json',{'source_model_id':'source-actor','source':{'sha256':source_pin['sha256']},
        'folds':{'A':{'sha256':'fold','runtime_view_sha256':'view','intersection_counts':{'query':0}}}})
    base={'phase_material':{'solver':{'instruction_template':'probes/base.txt','response_format_id':'slurp-solver-paired-label-json-v1'},
                            'slot':{'instruction_template':'probes/slot.txt'}},
          'thinking_policy':{'solver':{'enable_thinking':True},'slot':{'enable_thinking':True}},
          'knowledge_stores':{'locked':'same'}, 'native_contact_fit':{'run_id':'old-FIT'},
          'native_episode':{'preserve':'complete'}}
    class FakeJob:
        def __init__(self,c): self.c=c
        def derive(self,*args):
            b=copy.deepcopy(base)
            if 'decision_policy' in self.c:
                b['phase_material']['solver']['instruction_template']=self.c['decision_policy']['instruction_template']['relpath'][8:]
                b['native_contact_fit']['run_id']=self.c['run_id']+'-FIT'
            return b,None,None,{'runtime_view_sha256':'view'},{'model_fragment':'runtime'}
    monkeypatch.setattr(adapter,'NodeWorkload',FakeJob)
    spec={'parent':parent_pin,'collected':col_pin,'prepared':prep_pin,'source':source_pin,'exposure_audit':audit_pin,
          'prepared_files':prepared['files'],'algorithm':'reflexion','node':'decision','candidate_index':0,
          'candidate_sha256':adapter.canonical(proposal),'wire':{'path':str(wire_path),'sha256':adapter.digest(wire_path)},
          'fold':'A','name':'candidate','run_id':'new','template_relpath':'configs/probes/candidate.txt'}
    workload, receipt=adapter.compile_candidate(repo,tmp_path,spec)
    assert workload['pipeline']==parent['pipeline'] and receipt['node']=='decision'
    assert 'compare operation and object' in (repo/spec['template_relpath']).read_text()
    class BadJob(FakeJob):
        def derive(self,*args):
            b,*rest=super().derive(*args)
            if 'decision_policy' in self.c: b['knowledge_stores']['locked']='changed'
            return (b,*rest)
    monkeypatch.setattr(adapter,'NodeWorkload',BadJob)
    with pytest.raises(ValueError,match='full baseline changed'):
        adapter.compile_candidate(repo,tmp_path,spec)
