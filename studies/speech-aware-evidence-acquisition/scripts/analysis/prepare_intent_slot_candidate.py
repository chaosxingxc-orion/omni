"""Derive a configuration-driven candidate graph from an existing two-call base.

Produces preparation input only: existing roster/supply pins are retained; callers
must generate fresh preparation/FIT/registration identities before scheduling.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def derive(source, mode='batched'):
    from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
    from speech_aware_evidence_acquisition.rails.native_episode_config import load_native_episode_config
    if source['policy'] != 'phase-native-module-v1-c2':
        raise ValueError('source must be the already frozen single P / single S base')
    out=copy.deepcopy(source)
    out.setdefault("carrier_lock_key", "slurp")
    native=out['native_episode']
    for key in ('consumption_module','prompt_contact_limits'):
        native.pop(key,None)
    native['intent_slot_pipeline']={'schema':'saea-intent-slot-pipeline-v1','slot_mode':mode,
                                   'max_candidates':3,'candidate_outside_recovery':False}
    native['roles']['slot']={'phase_id':'slot','max_iterations':1}
    native['episode_id']='native-intent-slot-'+mode+'-v1'
    native['finite_completion_policy']='deliver-latest-or-abstain'
    out['policy']='phase-native-intent-slot-'+mode.replace('_','-')+'-v1'
    materials=out['phase_material']
    materials['perception']['instruction_template']='probes/slurp_native_candidate_perception_v1.txt'
    materials['perception']['fill_params'].update(min_candidates=1,max_candidates=3,
                                                 max_evidence_refs=8,max_unresolved_fields=8)
    materials['slot']={'instruction_template':'probes/slurp_native_candidate_slot_v1.txt',
                       'instruction_fill':None,'state_fill':None,'predecessor_binding':'none',
                       'response_format_id':'slurp-candidate-slots-v1','fill_params':{}}
    materials['solver'].update(instruction_template='probes/slurp_native_candidate_decision_v1.txt',
                               response_format_id='slurp-solver-paired-label-json-v1',fill_params={})
    for store in out.get('knowledge_stores',{}).values():
        for supply_key in ('candidate_supply','phase_supply','context_supply'):
            supply=store.get(supply_key,{})
            if 'solver' in supply:
                supply['slot']=copy.deepcopy(supply['solver'])
    if 'slot-contracts' in out.get('knowledge_stores', {}):
        if out.get('native_evidence_projection') not in (None, 'native-candidate-slot-view-v1'):
            raise ValueError('source evidence projection requires explicit composition before node derivation')
        out['native_evidence_projection'] = 'native-candidate-slot-view-v1'
        out.setdefault('native_evidence_projection_params', {'store_id':'slot-contracts',
                                                           'consumer_roles':['solver']})
    if out.get('native_evidence_projection') == 'native-candidate-slot-view-v1':
        consumers = out['native_evidence_projection_params']['consumer_roles']
        if 'slot' not in consumers:
            consumers.append('slot')
    load_native_episode_config(out,default_phase_program(out['policy']))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--slot-mode',choices=('batched','per_candidate'),default='batched')
    args=parser.parse_args()
    raw=args.source.read_bytes()
    source=json.loads(raw)
    baseline=source.get('candidate_baseline',source)
    candidate=derive(baseline,args.slot_mode)
    result={'schema':'saea-intent-slot-preparation-v1','status':'PREPARATION_ONLY',
            'source_sha256':hashlib.sha256(raw).hexdigest(),'candidate_baseline':candidate,
            'new_model_calls':0,'evaluation':{'fixed_development_samples':80,
                'controls':['old-P+joint-S','enhanced-P+joint-S'],
                'candidate':'enhanced-P+Slot+Decision','freeze_before_test':True,
                'full_test_samples':13078},
            'remaining':['fresh preparation/FIT/registration','fixed development paired comparison',
                         'frozen full TEST13078 reevaluation']}
    with args.out.open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')

if __name__=='__main__':main()
