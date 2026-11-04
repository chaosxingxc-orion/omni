"""Configured GEPA Perception parent/child plus AA/lock, each with its own S."""
import argparse
import json
from pathlib import Path
from gepa_pair_workload import PairWorkload, read
from tfrl_hardset_a0_registration import pinned

KIND='gepa-perception-template-pair'
SCHEMA='saea-gepa-perception-workload-v1'


class PerceptionWorkload(PairWorkload):
    CONTACTS=8
    ARMS=['parent','child','aa','lock']
    RESULT_SCHEMA='saea-gepa-perception-pair-read-v1'
    LABEL='GEPA Perception independent P to S comparison'
    PROGRAM='Four independent P-to-S branches: parent, child, original AA and original lock. Frozen Solver and original output protocol.'

    def __init__(self,config):
        if config.get('schema')!=SCHEMA:raise ValueError('explicit perception workload required')
        super().__init__({**config,'schema':'saea-gepa-pair-workload-v1'})
        self.config['schema']=SCHEMA
        self.KIND=KIND

    def material_checked(self,repo,data,material_pin):
        from gepa_perception_material import verify
        material=pinned(repo,material_pin)
        checked=verify(repo,data,material)
        ids=set(self.config['sample_ids']);contract=checked['contract']
        learned=set(contract['feedback_query_ids'])
        if self.config.get('evaluation_partition','feedback')=='selection' and ids&learned:
            raise ValueError('exposed teacher feedback cannot be independent selection')
        # Parent population validator receives the already verified partition document.
        checked['source_pins']['partition']={'relpath':'configs/evaluation/gepa-inputs-v1/partition.json',
            'sha256':checked['source_pins']['partition']['sha256'],'root':'repo'}
        return material,checked

    def derive(self,repo,data,material_pin):
        base,exp,roster,view,binding=super().derive(repo,data,material_pin)
        base['what_this_is']=self.PROGRAM+' Configured development comparison; no adoption claim.'
        exp['purpose']=base['what_this_is']
        exp['stage3_algorithm']['note']=self.PROGRAM
        exp['stage3_measurement']['note']=base['what_this_is']
        # Every contact consumes the original speech; include fallback headroom.
        exp['execution_plan']['planned_speech_audio_seconds']*=2
        return base,exp,roster,view,binding


def validate(repo,data,name,spec,exp,view):
    job=PerceptionWorkload(spec['job_config'])
    if spec['material']!=job.config['material']:raise ValueError('configured material differs')
    return job.validate(repo,data,name,spec,exp,view)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['stage','prepare','register'])
    for key in ('repo','data','config'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('out','stage-record','prepared'):p.add_argument('--'+key,type=Path)
    a=p.parse_args();job=PerceptionWorkload(read(a.config))
    if a.mode=='stage':result=job.stage(a.repo,a.data,job.config['material'],a.out)
    elif a.mode=='prepare':result=job.prepare(a.repo,a.data,a.stage_record,a.out)
    else:result=job.register(a.repo,a.data,a.stage_record,a.prepared)
    print(json.dumps(result,sort_keys=True))
