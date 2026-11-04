"""Configure a fixed-roster output-protocol pair through the shared Flight workflow."""
import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path
from gepa_pair_workload import PairWorkload
import tfrl_h1_observation as observation
from tfrl_hardset_a0_registration import pin,pinned
from tfrl_hardset_a0_workload import read
from speech_aware_evidence_acquisition.rails import consumption_module as cm
from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks
from speech_aware_evidence_acquisition.rails.slurp_solver_protocol import FORMAT_ID

KIND='solver-output-protocol-pair'
SCHEMA='saea-solver-protocol-workload-v1'


class ProtocolWorkload(PairWorkload):
    CONTACTS=3
    ARMS=['constrained','lock']
    RESULT_SCHEMA='saea-solver-protocol-pair-read-v1'
    LABEL='Solver output protocol paired development'
    PROGRAM='One shared P then schema-constrained S and unchanged S; output protocol only.'

    def __init__(self,config):
        if set(config)!={'schema','name','run_id','sample_ids','runtime_binding','execution','material'} or config['schema']!=SCHEMA:
            raise ValueError('exact protocol workload required')
        super().__init__({**config,'schema':'saea-gepa-pair-workload-v1'})
        if len(config['sample_ids'])!=80 or config['sample_ids']!=sorted(config['sample_ids']):
            raise ValueError('reuse complete sorted development80 roster')
        self.config=copy.deepcopy(config);self.KIND=KIND

    def material_checked(self,repo,data,material_pin):
        material=pinned(repo,material_pin)
        if set(material)!={'schema','source_roster','source_baseline','frame'} or material['schema']!='saea-solver-protocol-material-v1':
            raise ValueError('exact protocol material required')
        roster=pinned(repo,material['source_roster'])
        if [r['sample_id'] for r in roster['items']]!=self.config['sample_ids'] or roster['count']!=80:
            raise ValueError('protocol must reuse unchanged previous80 roster')
        if material['source_baseline']!=pin(repo,observation.SOURCE_BASE):
            raise ValueError('frozen baseline changed')
        return material,{'roster':roster}

    def derive(self,repo,data,material_pin):
        from tfgrpo_h2_paired import cumulative_supply_bound
        from speech_aware_evidence_acquisition.core.reasoning_knobs import validate_thinking_policy
        material,checked=self.material_checked(repo,data,material_pin)
        binding=self.runtime_binding(repo)
        h2,view=observation.population(repo,{'schema':'saea-h2-perception-thinking-v1','arm':'baseline','admitted_level':96})
        by_id={r['sample_id']:r for r in h2['items']};ids=self.config['sample_ids']
        if not set(ids)<=set(by_id):raise ValueError('protocol roster must be existing H2 development')
        items=[copy.deepcopy(by_id[s]) for s in ids]
        if checked['roster']['items']!=items:raise ValueError('source roster audio/identity differs from H2')
        roster=copy.deepcopy(checked['roster']);roster.update(roster_id=self.RUN,parent=material['source_roster'],selection_rule='Reuse unchanged exposed development80; protocol-only experiment.',claims_benefit=False)
        base=read(repo/observation.SOURCE_BASE);exp=read(repo/observation.SOURCE_EXP)
        model=read(repo/f"configs/model/{binding['model_fragment']}.json")
        base['thinking_policy']={r:copy.deepcopy(observation.THINKING_ARMS['baseline']) for r in ('perception','solver')}
        validate_thinking_policy(base['thinking_policy'],model)
        frame=copy.deepcopy(material['frame'])
        if frame['max_supplied_items']!=cumulative_supply_bound(base):raise ValueError('protocol shared supply bound differs')
        if frame['fit_estimate']['completion_ceiling']!=4096:raise ValueError('frozen 3072 thinking plus1024 answer required')
        slot=int(model['server_args'][model['server_args'].index('--max-model-len')+1])
        if frame['fit_estimate']['slot_tokens']!=slot:raise ValueError('protocol serving slot differs')
        block=copy.deepcopy(base['native_episode']['consumption_module'])
        block['columns']=[{'column_id':'constrained','kind':'protocol','node':'solver','response_format_id':FORMAT_ID}]
        block['protocol_frame']=frame
        c=cm.compile_module(block,carrier_module_hooks(repo_root=repo),repo_root=repo)
        if c.program_ceiling!=3:raise ValueError('one P two S required')
        original=copy.deepcopy(base)
        base=c.apply_to(base);base['native_episode']['consumption_module']=block
        for key in ('knowledge_stores','phase_material','thinking_policy','phase_tools','deliver_normaliser'):
            if base.get(key)!=original.get(key):raise ValueError('protocol compile changed frozen '+key)
        if base['phase_material']['solver'].get('response_format_id') is not None:
            raise ValueError('protocol lock must retain its original unconstrained format')
        base.pop('prepared_phase_supply',None);base['serving_slot_tokens']=slot
        base['native_contact_fit'].update(run_id=self.RUN+'-FIT',attempt_id='fit-v1',contact_cap=3*self.ROWS)
        base['what_this_is']=self.PROGRAM+' Previously exposed development, not promotion evidence.'
        cm.verify_declared_module(block,base,repo_root=repo)
        cell,=exp['cells'];cell.update(cell_id=self.NAME,baseline_fragment=self.NAME,model_fragment=binding['model_fragment'],run_id=self.RUN,attempt_id='generation-v1',allowed_execution_stages=['check','prepare-supply','worker','campaign'],readiness={'state':'held','requirements':[],'reason':'Prepare/register/admit exact protocol pair before dispatch.'})
        exp.update(experiment_id=self.RUN,purpose=base['what_this_is'],service_session=binding['service_session'])
        exp['population'].update(count=self.ROWS,roster_receipt=self.ROSTER,split_identity_hash=roster['split_identity_hash'])
        exp['populations']=[{'dataset_fragment':observation.DATASET,'ids':ids}]
        exp['execution_plan'].update(planned_model_calls=6*self.ROWS,planned_speech_audio_seconds=math.ceil(6*math.fsum(r['audio_seconds'] for r in items)))
        exp['stage3_algorithm']={'id':'none','operators_used':[],'note':'Protocol axis engineering ablation, not a TFRL learning algorithm.'}
        from speech_aware_evidence_acquisition.core.stage3_algorithms import validate_stage3_algorithm_label
        validate_stage3_algorithm_label(exp['stage3_algorithm'], cell=self.RUN)
        exp['stage3_measurement'].update(rows=self.ROWS,read_roster=roster['roster_id'],claims_benefit=False,expected_effect_pp=0,note=base['what_this_is'])
        return base,exp,roster,view,binding


def validate(repo,data,name,spec,exp,view):
    return ProtocolWorkload(spec['job_config']).validate(repo,data,name,spec,exp,view)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['stage','prepare','register'])
    for k in ('repo','data','config'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('out','stage-record','prepared'):p.add_argument('--'+k,type=Path)
    a=p.parse_args();job=ProtocolWorkload(read(a.config))
    if a.mode=='stage':result=job.stage(a.repo,a.data,job.config['material'],a.out)
    elif a.mode=='prepare':result=job.prepare(a.repo,a.data,a.stage_record,a.out)
    else:result=job.register(a.repo,a.data,a.stage_record,a.prepared)
    print(json.dumps(result,sort_keys=True))
