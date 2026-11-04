"""Configured external experience evaluation on fixed H2 subsets; shared Flight."""
import argparse
import copy
import json
import math
import re
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis')]
from gepa_pair_workload import PairWorkload
import tfrl_h1_observation as observation
from tfrl_hardset_a0_registration import pin,pinned
from tfrl_hardset_a0_workload import read,sha
from speech_aware_evidence_acquisition.rails import tfgrpo_consumption as consumption
from speech_aware_evidence_acquisition.rails import consumption_module as cm
from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks

KIND='external-experience-pair'
SCHEMA='saea-external-experience-workload-v1'

class ExternalExperienceWorkload(PairWorkload):
    CONTACTS=3
    ARMS=['learned','lock']
    RESULT_SCHEMA='saea-external-experience-pair-read-v1'
    LABEL='External reviewed experience paired development'
    PROGRAM='One shared P, reviewed external experience Solver and unchanged lock Solver.'

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
        if set(material)!={'schema','params','measurement'} or material['schema']!='saea-external-experience-material-v1':
            raise ValueError('explicit external experience material required')
        params=pinned(data,material['params']);measurement=pinned(data,material['measurement'])
        if params.get('source_kind') not in ('external-tfgrpo-v1','external-ace-v1','external-reflexion-v1','external-tfgrpo-reference-v1') or params.get('target_arm') is not None:
            raise ValueError('external reviewed unpaired source params required')
        consumption.verify(params,repo=repo,data=data,model_id=self.config['runtime_binding']['model_fragment'],consumer_ids=self.config['sample_ids'])
        expected=measure(repo,data,params,self.runtime_binding(repo))
        if expected!=measurement:raise ValueError('source-bound actual tokenizer measurement changed')
        return material,{'params':params,'measurement':measurement}

    def derive(self,repo,data,material_pin):
        from speech_aware_evidence_acquisition.data.splits import prefixed,split_identity_hash
        from tfgrpo_h2_paired import cumulative_supply_bound
        from speech_aware_evidence_acquisition.core.reasoning_knobs import validate_thinking_policy
        material,checked=self.material_checked(repo,data,material_pin);params=checked['params'];measurement=checked['measurement']
        binding=self.runtime_binding(repo)
        h2,view=observation.population(repo,{'schema':'saea-h2-perception-thinking-v1','arm':'baseline','admitted_level':96})
        by_id={r['sample_id']:r for r in h2['items']};ids=self.config['sample_ids']
        if not set(ids)<=set(by_id):raise ValueError('fixed original H2 consumers required')
        items=[copy.deepcopy(by_id[s]) for s in ids]
        roster={'schema':'saea-roster-receipt-v1','carrier_lock_key':'slurp','roster_id':self.RUN,'roster_role':'construction','split_role':'discovery','count':self.ROWS,'items':items,'split_identity_hash':split_identity_hash(prefixed('slurp',ids)),'strata_counts':dict(sorted(Counter(r['stratum'] for r in items).items())),'parent':pin(repo,observation.H2),'selection_rule':'Exact configured H2 subset; learning/selection query, utterance and audio excluded by external release.','claims_benefit':False}
        base=read(repo/observation.SOURCE_BASE);exp=read(repo/observation.SOURCE_EXP)
        base['thinking_policy']={r:copy.deepcopy(observation.THINKING_ARMS['baseline']) for r in ('perception','solver')}
        validate_thinking_policy(base['thinking_policy'],read(repo/f"configs/model/{binding['model_fragment']}.json"))
        block=copy.deepcopy(base['native_episode']['consumption_module']);before=cm.compile_module(block,carrier_module_hooks(repo_root=repo),repo_root=repo)
        if before.values()['phase_material']['solver']['instruction_template']!=params['base_template']['relpath']:
            raise ValueError('external experience changed base Solver template')
        block['components'].append({'component_id':consumption.COMPONENT,'state':'on','admitted_by':'development-evaluation','record':self.AUTH,'params':{**copy.deepcopy(params),'target_arm':'learned'}})
        block['columns']=[{'column_id':'learned','kind':'template','node':'solver','template':{'relpath':'configs/'+params['template']['relpath'],'sha256':params['template']['sha256']},'template_tokens':measurement['token_counts']['template']}]
        block['template_frame']={'max_supplied_items':cumulative_supply_bound(base),'delivering_template_tokens':measurement['token_counts']['base_template'],'twin':False,'fit_estimate':measurement['fit_estimate']}
        compiled=cm.compile_module(block,carrier_module_hooks(repo_root=repo),repo_root=repo)
        if compiled.program_ceiling!=3:raise ValueError('exact one P two Solver program required')
        base=compiled.apply_to(base);base['native_episode']['consumption_module']=block
        base.pop('prepared_phase_supply',None);base['serving_slot_tokens']=measurement['fit_estimate']['slot_tokens']
        base['native_contact_fit'].update(run_id=self.RUN+'-FIT',attempt_id='fit-v1',contact_cap=3*self.ROWS)
        base['what_this_is']=self.PROGRAM+' Fixed H2 development; no adoption claim.'
        cm.verify_declared_module(block,base,repo_root=repo)
        cell,=exp['cells'];cell.update(cell_id=self.NAME,baseline_fragment=self.NAME,model_fragment=binding['model_fragment'],run_id=self.RUN,attempt_id='generation-v1',allowed_execution_stages=['check','prepare-supply','worker','campaign'],readiness={'state':'held','requirements':[],'reason':'Prepare/register/admit exact external experience pair before dispatch.'})
        exp.update(experiment_id=self.RUN,purpose=base['what_this_is'],service_session=binding['service_session'])
        exp['population'].update(count=self.ROWS,roster_receipt=self.ROSTER,split_identity_hash=roster['split_identity_hash'])
        exp['populations']=[{'dataset_fragment':observation.DATASET,'ids':ids}]
        exp['execution_plan'].update(planned_model_calls=6*self.ROWS,planned_speech_audio_seconds=math.ceil(6*math.fsum(r['audio_seconds'] for r in items)))
        exp['stage3_algorithm']=algorithm_metadata(params['source_kind'])
        exp['stage3_measurement'].update(rows=self.ROWS,read_roster=roster['roster_id'],claims_benefit=False,expected_effect_pp=0,note=base['what_this_is'])
        return base,exp,roster,view,binding


def algorithm_metadata(source_kind):
    if source_kind == 'external-tfgrpo-reference-v1':
        return {'id':'tfgrpo-reference-multistage-external-dense-offline-v1',
                'operators_used':['context_application'],
                'note':'Original multistage reference with external teacher; continuous reward and offline rollout reuse adaptations.'}
    if source_kind == 'external-reflexion-v1':
        return {'id':'reflexion-external-memory-transfer-offline-v1','operators_used':['context_application'],
            'note':'Reflexion-inspired offline cross-task memory, not native online retries'}
    if source_kind == 'external-ace-v1':
        return {'id':'ace-external-additive-offline-v1','operators_used':['context_application'],
                'note':'ACE-inspired offline Reflector/Curator additive proposal; not native online ACE.'}
    if source_kind == 'external-tfgrpo-v1':
        return {'id':'tfgrpo-external-relative-feedback-v1','operators_used':['context_application'],
                'note':'Fused offline TF-GRPO-inspired candidate; not native multi-stage reference implementation.'}
    raise ValueError('unknown external experience algorithm')


def measure(repo,data,params,binding):
    """Measure exact templates with the resident tokenizer; native FIT remains separate."""
    from transformers import AutoTokenizer
    # Reuse the pinned configured template capacity estimate already deployed on this lane.
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
    counts={k:len(tokenizer.encode(consumption.checked_pin(repo/'configs',params[k]).read_text(),add_special_tokens=False)) for k in ('base_template','template')}
    estimate_source='configs/evaluation/gepa-seed-material-v1/complete.json'
    estimate=copy.deepcopy(read(repo/estimate_source)['fit_estimate'])
    model=read(repo/f"configs/model/{binding['model_fragment']}.json")
    estimate['slot_tokens']=int(model['server_args'][model['server_args'].index('--max-model-len')+1])
    estimate['completion_ceiling']=model['decoding_params']['max_tokens']
    return {'schema':'saea-external-experience-template-measurement-v1','token_counts':counts,'fit_estimate':estimate,'runtime_binding':binding,'tokenizer_files':files,'base_template':params['base_template'],'template':params['template'],'estimate_source':pin(repo,estimate_source),'native_fit_completed':False,'new_model_calls':0}


def validate(repo,data,name,spec,exp,view):
    job=ExternalExperienceWorkload(spec['job_config'])
    if spec['material']!=job.config['material']:raise ValueError('external material identity differs')
    return job.validate(repo,data,name,spec,exp,view)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['stage','prepare','register'])
    for k in ('repo','data','config'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('out','stage-record','prepared'):p.add_argument('--'+k,type=Path)
    a=p.parse_args();job=ExternalExperienceWorkload(read(a.config))
    if a.mode=='stage':result=job.stage(a.repo,a.data,job.config['material'],a.out)
    elif a.mode=='prepare':result=job.prepare(a.repo,a.data,a.stage_record,a.out)
    else:result=job.register(a.repo,a.data,a.stage_record,a.prepared)
    print(json.dumps(result,sort_keys=True))
