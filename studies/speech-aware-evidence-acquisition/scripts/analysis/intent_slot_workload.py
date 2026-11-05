"""Registered independent native graph arms over one immutable development roster."""
import argparse, copy, json, math
from pathlib import Path
from gepa_pair_workload import PairWorkload
import tfrl_h1_observation as observation
from tfrl_hardset_a0_registration import pin,pinned
from tfrl_hardset_a0_workload import read,sha
from prepare_intent_slot_candidate import derive as split_graph

KIND='intent-slot-node-ablation'
SCHEMA='saea-intent-slot-workload-v1'
ARMS=('old-p-joint-s','enhanced-p-joint-s','enhanced-p-slot-decision')

class NodeWorkload(PairWorkload):
    MAX_ROWS=1080
    RESULT_SCHEMA='saea-intent-slot-arm-read-v1'
    LABEL='Fixed development native intent Slot Decision ablation'
    def __init__(self,config):
        fields={'schema','name','run_id','sample_ids','runtime_binding','execution','material','node_arm'}
        if not fields <= set(config) <= fields | {'population','perception_policy','slot_policy'} or config['schema']!=SCHEMA or config['node_arm'] not in ARMS:
            raise ValueError('exact native node workload required')
        arm=config['node_arm']
        super().__init__({k:v for k,v in {**config,'schema':'saea-gepa-pair-workload-v1'}.items() if k not in ('node_arm','population','perception_policy','slot_policy')})
        if not config.get('population') and (len(config['sample_ids'])!=80 or config['sample_ids']!=sorted(config['sample_ids'])):
            raise ValueError('historical stable development80 required')
        if config.get('population'):
            pop=config['population']
            allowed={'slurp-speech-native-evo-fold-a-1080','slurp-speech-native-evo-fold-b-1080'}
            if set(pop)!={'dataset','dataset_sha256','roster','source_baseline'} or pop['dataset'] not in allowed:
                raise ValueError('explicit fixed fold dataset/roster/source baseline required')
            self.DATASET=pop['dataset']
        self.config=copy.deepcopy(config);self.KIND=KIND;self.ARMS=[arm];self.DELIVERED_ARM=arm
        self.CONTACTS=3 if arm=='enhanced-p-slot-decision' else 2
        self.PROGRAM=arm+' independent original-audio graph; no shared responses across runs.'

    def material_checked(self,repo,data,material_pin):
        material=pinned(repo,material_pin)
        if self.config.get('population'):
            pop=self.config['population'];roster=pinned(repo,pop['roster']);view=pinned(repo,{'relpath':f'configs/dataset/{self.DATASET}.json','sha256':pop['dataset_sha256']});ids=self.config['sample_ids']
            if len(ids)!=1080 or roster.get('count',len(roster['items']))!=len(ids) or view['count']!=len(ids) or [r['sample_id'] for r in roster['items']]!=ids or [r['sample_id'] for r in view['items']]!=ids or roster['split_identity_hash']!=view['split_identity_hash']:
                raise ValueError('fixed fold roster, order, or audio identity mismatch')
            if set(material)!={'schema','source_baseline','source_roster','dataset'} or material['schema']!='saea-intent-slot-fold-material-v1' or material['source_baseline']!=pop['source_baseline'] or material['source_roster']!=pop['roster'] or material['dataset']!={'relpath':f'configs/dataset/{self.DATASET}.json','sha256':pop['dataset_sha256']}:
                raise ValueError('fold material pins differ from population')
            import hashlib, json
            from speech_aware_evidence_acquisition.data.splits import prefixed, split_identity_hash
            body={k:v for k,v in view.items() if k!='runtime_view_sha256'}
            encoded=json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
            if hashlib.sha256(encoded).hexdigest()!=view['runtime_view_sha256'] or split_identity_hash(prefixed('slurp',ids))!=view['split_identity_hash']:
                raise ValueError('fold view or split digest differs')
            for r, v in zip(roster['items'],view['items']):
                if (r['audio_seconds']!=v['audio_seconds'] or not v.get('audio_sha256') or not v.get('packet_id') or not v.get('media_relpath') or v.get('sample_rate_hz')!=16000):
                    raise ValueError('fold audio identity or duration differs')
            source=pinned(repo,pop['source_baseline'])
            if source.get('policy') != 'phase-native-module-v1-c2':
                from derive_lock_only_baseline import derive as lock_only
                source=lock_only(source,repo=repo)
            queries=[source['knowledge_stores']['intent-boundaries']['phase_supply']['perception']['query_source'],source['knowledge_stores']['train-exemplars']['phase_supply']['solver']['query_source']]
            for query in queries:
                path=data/query['relpath']
                if sha(path)!=query['sha256']:
                    raise ValueError('fold query source digest differs')
                with path.open() as stream:
                    records=[json.loads(line) for line in stream if line.strip()]
                if [r['id'] for r in records]!=ids:
                    raise ValueError('fold query source population differs')
                if query['kind']=='sealed-segment-vectors' and any(r['audio_sha256']!=v['audio_sha256'] or r['source_sample_rate_hz']!=v['sample_rate_hz'] for r,v in zip(records,view['items'])):
                    raise ValueError('fold query audio identity differs')
            fold_roster=copy.deepcopy(roster);fold_roster.update(roster_id=self.RUN,count=len(ids),selection_rule='Exact existing fold; no resampling.',claims_benefit=False);return material,{'roster':fold_roster,'view':view,'source_baseline':source}
        if set(material)!={'schema','source_baseline','source_roster','partition'} or material['schema']!='saea-intent-slot-material-v1':
            raise ValueError('exact source-bound node material required')
        if material['source_baseline']!=pin(repo,observation.SOURCE_BASE):
            raise ValueError('frozen baseline changed')
        original=pinned(repo,material['source_roster'])
        if original['count']!=80 or [r['sample_id'] for r in original['items']]!=self.config['sample_ids']:
            raise ValueError('must reuse exact existing development roster')
        roster,view=self.population(repo,material['partition'])
        if roster['items']!=original['items']:
            raise ValueError('source roster identity/audio differs')
        return material,{'roster':roster,'view':view}

    def derive(self,repo,data,material_pin):
        from speech_aware_evidence_acquisition.core.reasoning_knobs import validate_thinking_policy
        material,checked=self.material_checked(repo,data,material_pin);binding=self.runtime_binding(repo)
        if self.config.get('population'):
            from derive_lock_only_baseline import derive as lock_only
            source=checked['source_baseline']
            base=copy.deepcopy(source) if source.get('policy')=='phase-native-module-v1-c2' else lock_only(source, repo=repo)
        else:
            base=read(repo/observation.SOURCE_BASE)
        exp=read(repo/observation.SOURCE_EXP)
        base['thinking_policy']={r:copy.deepcopy(observation.THINKING_ARMS['baseline']) for r in ('perception','solver')}
        arm=self.config['node_arm']
        if arm!='old-p-joint-s':
            expanded=split_graph(base)
            if arm=='enhanced-p-slot-decision':
                base=expanded
                if self.config.get('population'):
                    base['native_episode']['intent_slot_pipeline'].update(evidence_policy='unique_source_alias_v1',duplicate_candidate_policy='identical_v1')
                base.pop('carrier_lock_key',None)  # Dataset owns the carrier in composed configuration.
                base['thinking_policy']['slot']=copy.deepcopy(observation.THINKING_ARMS['baseline'])
            else:
                base['phase_material']['perception']=expanded['phase_material']['perception']
                if self.config.get('population'):
                    base['native_episode']['matched_perception_context']={**expanded['native_episode']['intent_slot_pipeline'],'evidence_policy':'unique_source_alias_v1','duplicate_candidate_policy':'identical_v1'}
                    base['native_episode'].pop('prompt_contact_limits',None)
                base['native_episode'].pop('consumption_module',None)
                base['native_evidence_projection']=expanded['native_evidence_projection']
                base['native_evidence_projection_params']={**expanded['native_evidence_projection_params'],'consumer_roles':['solver']}
        for role in ('perception','slot'):
            policy=self.config.get(role+'_policy')
            if policy is None:
                continue
            if role=='slot' and arm!='enhanced-p-slot-decision':
                raise ValueError('Slot policy requires the split graph')
            if (not isinstance(policy,dict) or set(policy)!={'thinking','instruction_template'}
                or not isinstance(policy['thinking'],dict)
                or not isinstance(policy['instruction_template'],dict)
                or set(policy['instruction_template'])!={'relpath','sha256'}):
                raise ValueError('source-pinned node policy required')
            template=policy['instruction_template']
            from tfrl_hardset_a0_workload import confined
            if (not isinstance(template['relpath'],str)
                or not template['relpath'].startswith('configs/probes/')
                or '..' in Path(template['relpath']).parts):
                raise ValueError('Perception template must be under configs/probes')
            path=confined(repo,template['relpath'])
            if not path.is_file() or sha(path)!=template['sha256']:
                raise ValueError('Perception instruction template pin differs')
            base['phase_material'][role]['instruction_template']=template['relpath'][len('configs/'):]
            base['thinking_policy'][role]=copy.deepcopy(policy['thinking'])
        model=read(repo/f"configs/model/{binding['model_fragment']}.json")
        validate_thinking_policy(base['thinking_policy'],model)
        slot=int(model['server_args'][model['server_args'].index('--max-model-len')+1])
        base.pop('prepared_phase_supply',None);base['serving_slot_tokens']=slot
        base['native_contact_fit'].update(run_id=self.RUN+'-FIT',attempt_id='fit-v1',contact_cap=self.CONTACTS*self.ROWS)
        base['what_this_is']=self.PROGRAM+' Fixed TRAIN development, no promotion claim.'
        roster=checked['roster'];roster['parent']=self.config['population']['roster'] if self.config.get('population') else material['source_roster']
        roster['selection_rule']='Reuse exact existing fold; no sampling or TEST tuning.' if self.config.get('population') else 'Reuse exact existing development80; no sampling or TEST tuning.'
        cell,=exp['cells'];cell.update(cell_id=self.NAME,baseline_fragment=self.NAME,model_fragment=binding['model_fragment'],dataset_fragment=self.DATASET,run_id=self.RUN,attempt_id='generation-v1',allowed_execution_stages=['check','prepare-supply','worker','campaign'],readiness={'state':'held','requirements':[],'reason':'Registered native graph requires fresh preparation/admission.'})
        exp.update(experiment_id=self.RUN,purpose=base['what_this_is'],service_session=binding['service_session'])
        if self.config.get('population'):
            exp['split_id']=checked['view']['split_name']
        exp['population'].update(count=self.ROWS,roster_receipt=self.ROSTER,split_identity_hash=roster['split_identity_hash'])
        exp['populations']=[{'dataset_fragment':self.DATASET,'ids':self.config['sample_ids']}]
        exp['population']['dataset_fragment']=self.DATASET
        cap=2*self.CONTACTS
        exp['execution_plan'].update(planned_model_calls=cap*self.ROWS,planned_speech_audio_seconds=math.ceil(cap*math.fsum(r['audio_seconds'] for r in roster['items'])))
        exp['stage3_algorithm']={'id':'none','operators_used':[],'note':'Node architecture ablation, not a new learning algorithm.'}
        exp['stage3_measurement'].update(rows=self.ROWS,read_roster=roster['roster_id'],claims_benefit=False,expected_effect_pp=0,note=base['what_this_is'])
        return base,exp,roster,checked['view'],binding

def validate(repo,data,name,spec,exp,view):
    job=NodeWorkload(spec['job_config'])
    if spec['material']!=job.config['material']:raise ValueError('node material differs')
    return job.validate(repo,data,name,spec,exp,view)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['stage','prepare','register'])
    for k in ('repo','data','config'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('out','stage-record','prepared'):p.add_argument('--'+k,type=Path)
    a=p.parse_args();job=NodeWorkload(read(a.config))
    if a.mode=='stage':result=job.stage(a.repo,a.data,job.config['material'],a.out)
    elif a.mode=='prepare':result=job.prepare(a.repo,a.data,a.stage_record,a.out)
    else:result=job.register(a.repo,a.data,a.stage_record,a.prepared)
    print(json.dumps(result,sort_keys=True))
