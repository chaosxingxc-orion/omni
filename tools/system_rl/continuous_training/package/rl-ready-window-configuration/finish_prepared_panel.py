"""Turn a completed frozen DATA panel into training inputs with the existing stages."""
import argparse
import fcntl
from pathlib import Path
from build import pin,save
from prepare_s1_recordings import checked,read,needed
from source.config_profile import stage_config
from s1_panel_selection import partition,observed,sample
from prepare_s1_training_inputs import prepare
from seal_s1_training_roster import seal

def finish(config):
    cfg=read(config);out=Path(cfg['output']);out.mkdir(mode=0o700,parents=True,exist_ok=True)
    from multimodal_jev.local_runtime_storage import check_paths
    check_paths({'output':out},min_free_bytes=(20<<30)+cfg['storage_budget_bytes'],min_free_fraction=.1)
    with (out/'finish.guard').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        endpoint=pin(cfg['collection']);v=read(checked(read(checked(endpoint))['version']))
        req=read(checked(v['dataset_request_ref']));data=read(checked(v['inputs_ref']))
        needed(req['scope']=='dataset_preparation' and req['stage']=='screen' and
               req['panel']==cfg['panel']==data['panel_ref'] and req['count']==len(data['rows']) and
               req['endpoint_witness_ref']==v['endpoint_witness_ref'] and req['base_manifest']==v['base_manifest']
               and v['mode']=='live' and not v['synthetic'] and v['status']=='SEALED' and
               v['unresolved_requests']==0,'different or incomplete frozen screen')
        design=read(checked(cfg['experiment']))
        values=dict(S1_INPUTS=v['inputs_ref'],S1_PARTITION_OUTPUT=str(out/'PARTITION.json'),
            S1_TRAIN_INPUT_OUTPUT=str(out/'static'),S1_OUTCOME_OUTPUT=str(out/'OUTCOMES.json'),
            S1_CLOSED_COLLECTIONS=[endpoint],S1_FROZEN_SYSTEM_VERSION=v['system_version'],
            S1_FROZEN_EXECUTION_VERSION=v['execution_version'],S1_ROSTER_OUTPUT=str(out/'roster'))
        def stage(name):return stage_config(design,dict(schema='omni-local-profile-v1',values=values),name)
        c=stage('s1-partition-config-v1')
        split=dict(partition(c),configuration=c);values['S1_PARTITION']=save(Path(c['output']),split)
        c=stage('s1-static-training-input-config-v1');values['S1_PREPARED_INPUTS']=prepare(c,Path(c['output']))
        c=stage('s1-outcome-config-v1');found,audit=observed(c,split)
        needed(len(found)==req['count'],'unobserved recordings in closed collection')
        outcome=sample(c,split,found)
        values['S1_OBSERVED_OUTCOMES']=save(Path(c['output']),dict(outcome,configuration=c,audit=audit))
        roster=seal(stage('s1-screen-roster-config-v1'))
        return save(out/'COMPLETE.json',dict(roster=roster,pool=pin(out/'roster/NODE-POOL.json'),
                 sampling=pin(out/'roster/SAMPLING.json'),panel=req['count'],selected=len(outcome['selected']),
                 models_called=0,optimizer_steps=0))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config',type=Path);print(finish(p.parse_args().config))
