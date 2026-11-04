"""Registered independent native graph arms over one immutable development roster."""
import argparse, copy, json, math
from pathlib import Path
from gepa_pair_workload import PairWorkload
import tfrl_h1_observation as observation
from tfrl_hardset_a0_registration import pin,pinned
from tfrl_hardset_a0_workload import read
from prepare_intent_slot_candidate import derive as split_graph

KIND='intent-slot-node-ablation'
SCHEMA='saea-intent-slot-workload-v1'
ARMS=('old-p-joint-s','enhanced-p-joint-s','enhanced-p-slot-decision')

class NodeWorkload(PairWorkload):
    RESULT_SCHEMA='saea-intent-slot-arm-read-v1'
    LABEL='Fixed development native intent Slot Decision ablation'
    def __init__(self,config):
        fields={'schema','name','run_id','sample_ids','runtime_binding','execution','material','node_arm'}
        if set(config)!=fields or config['schema']!=SCHEMA or config['node_arm'] not in ARMS:
            raise ValueError('exact native node workload required')
        arm=config['node_arm']
        super().__init__({k:v for k,v in {**config,'schema':'saea-gepa-pair-workload-v1'}.items() if k!='node_arm'})
        if len(config['sample_ids'])!=80 or config['sample_ids']!=sorted(config['sample_ids']):
            raise ValueError('complete stable sorted development80 required')
        self.config=copy.deepcopy(config);self.KIND=KIND;self.ARMS=[arm];self.DELIVERED_ARM=arm
        self.CONTACTS=3 if arm=='enhanced-p-slot-decision' else 2
        self.PROGRAM=arm+' independent original-audio graph; no shared responses across runs.'

    def material_checked(self,repo,data,material_pin):
        material=pinned(repo,material_pin)
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
        base=read(repo/observation.SOURCE_BASE);exp=read(repo/observation.SOURCE_EXP)
        base['thinking_policy']={r:copy.deepcopy(observation.THINKING_ARMS['baseline']) for r in ('perception','solver')}
        arm=self.config['node_arm']
        if arm!='old-p-joint-s':
            expanded=split_graph(base)
            if arm=='enhanced-p-slot-decision':
                base=expanded
                base.pop('carrier_lock_key',None)  # Dataset owns the carrier in composed configuration.
                base['thinking_policy']['slot']=copy.deepcopy(observation.THINKING_ARMS['baseline'])
            else:
                base['phase_material']['perception']=expanded['phase_material']['perception']
                base['native_episode'].pop('consumption_module',None)
                base['native_evidence_projection']=expanded['native_evidence_projection']
                base['native_evidence_projection_params']={**expanded['native_evidence_projection_params'],'consumer_roles':['solver']}
        model=read(repo/f"configs/model/{binding['model_fragment']}.json")
        validate_thinking_policy(base['thinking_policy'],model)
        slot=int(model['server_args'][model['server_args'].index('--max-model-len')+1])
        base.pop('prepared_phase_supply',None);base['serving_slot_tokens']=slot
        base['native_contact_fit'].update(run_id=self.RUN+'-FIT',attempt_id='fit-v1',contact_cap=self.CONTACTS*self.ROWS)
        base['what_this_is']=self.PROGRAM+' Fixed TRAIN development, no promotion claim.'
        roster=checked['roster'];roster['parent']=material['source_roster']
        roster['selection_rule']='Reuse exact existing development80; no sampling or TEST tuning.'
        cell,=exp['cells'];cell.update(cell_id=self.NAME,baseline_fragment=self.NAME,model_fragment=binding['model_fragment'],run_id=self.RUN,attempt_id='generation-v1',allowed_execution_stages=['check','prepare-supply','worker','campaign'],readiness={'state':'held','requirements':[],'reason':'Registered native graph requires fresh preparation/admission.'})
        exp.update(experiment_id=self.RUN,purpose=base['what_this_is'],service_session=binding['service_session'])
        exp['population'].update(count=self.ROWS,roster_receipt=self.ROSTER,split_identity_hash=roster['split_identity_hash'])
        exp['populations']=[{'dataset_fragment':observation.DATASET,'ids':self.config['sample_ids']}]
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
