"""Configured actual slot-store replacement on fixed H2 subsets; shared Flight."""
import argparse
import copy
import json
import math
import re
import hashlib
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis')]
from gepa_pair_workload import PairWorkload
import tfrl_h1_observation as observation
from tfrl_hardset_a0_registration import pin,pinned
from tfrl_hardset_a0_workload import read,sha
from external_slot_candidate import compile_candidate, verify_response, checked, authored_release
from speech_aware_evidence_acquisition.rails import consumption_module as cm
from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks

KIND='external-slot-knowledge-pair'
SCHEMA='saea-external-slot-knowledge-workload-v1'

class ExternalSlotWorkload(PairWorkload):
    CONTACTS=3
    ARMS=['learned','lock']
    RESULT_SCHEMA='saea-external-slot-knowledge-pair-read-v1'
    LABEL='Source-bound slot knowledge paired development'
    PROGRAM='One shared P, original and candidate slot knowledge with identical Solver prompts.'

    def __init__(self,config):
        if set(config)!={'schema','name','run_id','sample_ids','runtime_binding','execution','material'} or config['schema']!=SCHEMA:
            raise ValueError('exact configured external experience workload required')
        # Reuse identity/client-width bounds only; no GEPA scientific identity.
        super().__init__({**config,'schema':'saea-gepa-pair-workload-v1'})
        # Runtime SLURP load_runtime_view sorts by sample_id, independent of
        # the target/control grouping used to choose an unchanged subset.
        if config['sample_ids'] != sorted(config['sample_ids']):
            raise ValueError('external consumers must use runtime sample_id order before review and release')
        self.config=copy.deepcopy(config);self.KIND=KIND

    def material_checked(self,repo,data,material_pin):
        material=pinned(repo,material_pin)
        if set(material)!={'schema','compiler','candidate','release','review','measurement'} or material['schema']!='saea-external-slot-knowledge-material-v1':
            raise ValueError('exact slot material required')
        cfg=pinned(data,material['compiler'])
        original,base,contract,dynamic,forbidden=(read(checked(cfg[k])) for k in ('original','base','contract','dynamic','forbidden_sample_ids'))
        proposals,custody=verify_response(cfg['teacher'],dynamic)
        candidate=compile_candidate(original,base,contract,dynamic,proposals['candidates'][cfg['candidate_index']],custody,forbidden)
        candidate=json.loads(json.dumps(candidate))
        saved=pinned(data,material['candidate'])
        # Reviewed bytes precede strengthened utterance-level isolation diagnostics.
        # Compare every scientific field; separately enforce the stronger check above.
        if {k:v for k,v in saved.items() if k!='isolation'}!={k:v for k,v in candidate.items() if k!='isolation'}:
            raise ValueError('candidate differs from actual teacher compilation')
        if saved.get('isolation',{}).get('new_optimizer_forbidden_overlap')!=[]:
            raise ValueError('saved candidate reports optimizer overlap')
        review=pinned(data,material['review'])
        if (review['decision']!='approve_bounded_development_only' or review['promotion'] is not False
                or review['reviewed_candidate_sha256'].get(str(cfg['candidate_index']))!=material['candidate']['sha256']):
            raise ValueError('exact root development review required')
        release=pinned(data,material['release'])
        authored, rebuilt=authored_release(candidate)
        if release['snapshot_sha256']!=rebuilt['snapshot_sha256'] or pinned(data,release['source'])!=authored:
            raise ValueError('release not bound to actual authored candidate')
        measurement=pinned(data,material['measurement'])
        if measurement!=measure(repo,data,candidate,self.runtime_binding(repo),measurement['snapshot']):
            raise ValueError('measurement runtime or tokenizer binding differs')
        return material,{'candidate':candidate,'release':release,'measurement':measurement}

    def derive(self,repo,data,material_pin):
        from speech_aware_evidence_acquisition.data.splits import prefixed,split_identity_hash
        from tfgrpo_h2_paired import cumulative_supply_bound
        from speech_aware_evidence_acquisition.core.reasoning_knobs import validate_thinking_policy
        material,checked=self.material_checked(repo,data,material_pin);candidate=checked['candidate'];measurement=checked['measurement']
        binding=self.runtime_binding(repo)
        h2,view=observation.population(repo,{'schema':'saea-h2-perception-thinking-v1','arm':'baseline','admitted_level':96})
        by_id={r['sample_id']:r for r in h2['items']};ids=self.config['sample_ids']
        if not set(ids)<=set(by_id):raise ValueError('fixed original H2 consumers required')
        items=[copy.deepcopy(by_id[s]) for s in ids]
        roster={'schema':'saea-roster-receipt-v1','carrier_lock_key':'slurp','roster_id':self.RUN,'roster_role':'construction','split_role':'discovery','count':self.ROWS,'items':items,'split_identity_hash':split_identity_hash(prefixed('slurp',ids)),'strata_counts':dict(sorted(Counter(r['stratum'] for r in items).items())),'parent':pin(repo,observation.H2),'selection_rule':'Exact existing targeted H2 subset; new optimizer source excluded at utterance level.','claims_benefit':False}
        base=read(repo/observation.SOURCE_BASE);exp=read(repo/observation.SOURCE_EXP)
        base['thinking_policy']={r:copy.deepcopy(observation.THINKING_ARMS['baseline']) for r in ('perception','solver')}
        validate_thinking_policy(base['thinking_policy'],read(repo/f"configs/model/{binding['model_fragment']}.json"))
        block=copy.deepcopy(base['native_episode']['consumption_module']);before=cm.compile_module(block,carrier_module_hooks(repo_root=repo),repo_root=repo)
        original=base['knowledge_stores']['slot-contracts']['snapshot']
        if original['sha256']!=candidate['original_snapshot_sha256']:
            raise ValueError('candidate original differs from actual deployed slot snapshot')
        bound=cumulative_supply_bound(base)
        alternate={'release':material['release'],'snapshot':measurement['snapshot']}
        block['columns']=[{'column_id':'learned','kind':'supply','node':'solver','max_supplied_items':bound,
            'release_swap':{'store_id':'slot-contracts','release':material['release']['relpath']}}]
        block.pop('template_frame',None)
        block['supply_frame']={'delivering_max_supplied_items':bound,'twin':False,
            'alternates':{'slot-contracts':[alternate]},'fit_estimate':measurement['fit_estimate']}
        compiled=cm.compile_module(block,carrier_module_hooks(repo_root=repo),repo_root=repo)
        if compiled.program_ceiling!=3:raise ValueError('exact one P two Solver program required')
        base=compiled.apply_to(base);base['native_episode']['consumption_module']=block
        from speech_aware_evidence_acquisition.rails.knowledge_stores import load_knowledge_store_binding
        binding_check=load_knowledge_store_binding(base,data_root=data)
        for store in binding_check.stores.values():
            if {':'.join(x.split(':')[:3]) for x in store.snapshot.source_sample_ids}&{':'.join(x.split(':')[:3]) for x in ids}:
                raise ValueError('consumer intersects knowledge source')
        base.pop('prepared_phase_supply',None);base['serving_slot_tokens']=measurement['fit_estimate']['slot_tokens']
        base['native_contact_fit'].update(run_id=self.RUN+'-FIT',attempt_id='fit-v1',contact_cap=3*self.ROWS)
        base['what_this_is']=self.PROGRAM+' Fixed H2 development; no adoption claim.'
        cm.verify_declared_module(block,base,repo_root=repo)
        cell,=exp['cells'];cell.update(cell_id=self.NAME,baseline_fragment=self.NAME,model_fragment=binding['model_fragment'],run_id=self.RUN,attempt_id='generation-v1',allowed_execution_stages=['check','prepare-supply','worker','campaign'],readiness={'state':'held','requirements':[],'reason':'Prepare/register/admit exact external experience pair before dispatch.'})
        exp.update(experiment_id=self.RUN,purpose=base['what_this_is'],service_session=binding['service_session'])
        exp['population'].update(count=self.ROWS,roster_receipt=self.ROSTER,split_identity_hash=roster['split_identity_hash'])
        exp['populations']=[{'dataset_fragment':observation.DATASET,'ids':ids}]
        exp['execution_plan'].update(planned_model_calls=6*self.ROWS,planned_speech_audio_seconds=math.ceil(6*math.fsum(r['audio_seconds'] for r in items)))
        exp['stage3_algorithm']={'id':'ace-external-slot-construction-offline-v1','operators_used':['context_application'],'note':'Single-slot actual knowledge replacement from disjoint TRAIN teacher evidence; ACE-inspired, not full ACE; no solver append prompt.'}
        exp['stage3_measurement'].update(rows=self.ROWS,read_roster=roster['roster_id'],claims_benefit=False,expected_effect_pp=0,note=base['what_this_is'])
        return base,exp,roster,view,binding



def measure(repo,data,candidate,binding,snapshot):
    from transformers import AutoTokenizer
    from speech_aware_evidence_acquisition.rails.knowledge import KnowledgeSnapshot
    repo=Path(repo);data=Path(data)
    receipt=read(repo/'docs'/binding['runtime_receipt']);model_dir=Path(receipt['model']['model_dir']).resolve()
    if not model_dir.is_relative_to(data.resolve()/'models'):raise ValueError('owned tokenizer required')
    expected={x['path']:x['sha256'] for x in receipt['model']['files']};files={}
    for name in ('config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','added_tokens.json','special_tokens_map.json'):
        p=model_dir/name
        if p.exists():
            rel=str(p.relative_to(data.resolve()))
            if expected.get(rel)!=sha(p):raise ValueError('resident tokenizer file changed')
            files[rel]=sha(p)
    if not {'tokenizer_config.json','vocab.json','merges.txt'}<={Path(p).name for p in files}:raise ValueError('complete resident tokenizer required')
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
    estimate_source='configs/evaluation/gepa-seed-material-v1/complete.json'
    estimate=copy.deepcopy(read(repo/estimate_source)['fit_estimate'])
    model=read(repo/f"configs/model/{binding['model_fragment']}.json")
    estimate['slot_tokens']=int(model['server_args'][model['server_args'].index('--max-model-len')+1])
    # Evaluated node budget is baseline P/S 3072 thinking + 1024 answer.
    estimate['completion_ceiling']=4096
    measured={'fit_estimate':estimate,'runtime_binding':binding,'tokenizer_files':files,
              'estimate_source':pin(repo,estimate_source),'native_fit_completed':False,'new_model_calls':0}
    if KnowledgeSnapshot.from_frozen_json(read(data/snapshot['relpath'])).content_hash()!=snapshot['sha256'] or snapshot['sha256']!=authored_release(candidate)[1]['snapshot_sha256']:
        raise ValueError('measured snapshot differs')
    maximum=max(len(tokenizer.encode(v['content'],add_special_tokens=False)) for v in candidate['snapshot']['values'])
    source=read(repo/observation.SOURCE_BASE)['knowledge_stores']['slot-contracts']['snapshot']
    old=KnowledgeSnapshot.from_frozen_json(read(data/source['relpath']))
    if old.content_hash()!=candidate['original_snapshot_sha256']:raise ValueError('measurement original snapshot differs')
    oldvalues={v.value_id:v.content for v in old.values}
    increments=[max(0,len(tokenizer.encode(v['content'],add_special_tokens=False))-len(tokenizer.encode(oldvalues[v['value_id']],add_special_tokens=False))) for v in candidate['snapshot']['values'] if v['value_id'] in candidate['changed_value_ids']]
    # Exactly one existing card is replaced. Charge its positive growth once;
    # do not pretend every one of 56 supplied values is this largest slot card.
    measured['fit_estimate']['base_prompt_tokens']+=sum(increments)
    measured['slot_text_growth_tokens']=sum(increments)
    measured.update(schema='saea-external-slot-measurement-v1',snapshot=snapshot,max_slot_value_tokens=maximum)
    return measured


def validate(repo,data,name,spec,exp,view):
    job=ExternalSlotWorkload(spec['job_config'])
    if spec['material']!=job.config['material']:raise ValueError('slot material identity differs')
    return job.validate(repo,data,name,spec,exp,view)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['stage','prepare','register'])
    for k in ('repo','data','config'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('out','stage-record','prepared'):p.add_argument('--'+k,type=Path)
    a=p.parse_args();job=ExternalSlotWorkload(read(a.config))
    if a.mode=='stage':result=job.stage(a.repo,a.data,job.config['material'],a.out)
    elif a.mode=='prepare':result=job.prepare(a.repo,a.data,a.stage_record,a.out)
    else:result=job.register(a.repo,a.data,a.stage_record,a.prepared)
    print(json.dumps(result,sort_keys=True))
