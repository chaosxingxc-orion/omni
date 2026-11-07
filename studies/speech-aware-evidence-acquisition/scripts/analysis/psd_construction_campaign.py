#!/usr/bin/env python3
"""Compile and run CPU preparation for independently registered PSD rollouts.

No model is invoked. Each compiled member is an ordinary NodeWorkload; execution
and recovery remain the responsibility of the existing registered Flight path.
"""
import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/analysis')]
from intent_slot_workload import NodeWorkload, read
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import canonical, sha


def compile_campaign(config):
    keys = {'schema', 'name', 'run_id', 'workload', 'collection', 'deployment_policy'}
    if set(config) != keys or config['schema'] != 'saea-psd-construction-campaign-v1':
        raise ValueError('exact construction campaign required')
    name, run = config['name'], config['run_id']
    if not isinstance(name, str) or not re.fullmatch('[a-z0-9][a-z0-9-]*', name) or not isinstance(run, str) or not re.fullmatch('[A-Z0-9][A-Z0-9-]*', run):
        raise ValueError('safe campaign identities required')
    collection = config['collection']
    if (not isinstance(collection, dict) or set(collection) != {'temperature', 'seeds'}
            or not isinstance(collection['seeds'], list) or not 2 <= len(collection['seeds']) <= 100
            or any(type(seed) is not int or not 0 <= seed <= 2**32-2 for seed in collection['seeds'])
            or len(set(collection['seeds'])) != len(collection['seeds'])):
        raise ValueError('explicit finite K and distinct registered seeds required')
    if not isinstance(config['deployment_policy'], dict) or not config['deployment_policy']:
        raise ValueError('separate deployment policy declaration required')
    template = copy.deepcopy(config['workload'])
    p_only = template.get('schema') == 'saea-perception-only-workload-v1'
    if p_only and template.get('learning_mode', 'tf-collection') != 'tf-collection':
        raise ValueError('candidate evaluation is not a TF rollout campaign')
    if (template.get('population', {}).get('role') != 'construction'
            or (not p_only and (template.get('node_arm') != 'enhanced-p-slot-decision'
                                or not isinstance(template.get('pipeline'), dict)))
            or 'collection_decoding' in template):
        raise ValueError('construction PSD v2 SDK workload template required')
    if p_only:
        from p_only_workload import PerceptionOnlyWorkload
    workload_class = PerceptionOnlyWorkload if p_only else NodeWorkload
    policy = {'workload': {k:v for k,v in template.items() if k not in ('name','run_id')},
              'temperature': collection['temperature'],
              'seed_rule': 'ordered-explicit-seeds-v1'}
    members = []
    for index, seed in enumerate(collection['seeds'], 1):
        job = copy.deepcopy(template)
        job.update(name=f'{name}-k{index:03}', run_id=f'{run}-K{index:03}',
                   collection_decoding={'temperature':collection['temperature'], 'seed':seed})
        parsed = workload_class(job)
        budgets = parsed.contact_budgets()
        members.append({'index':index, 'seed':seed, 'job_config':job,
                        'fit_run_id':job['run_id']+'-FIT', 'fit_attempt_id':'fit-v1',
                        'generation_attempt_id':'generation-v1',
                        'normal_generation_calls':budgets['normal_generation_calls'],
                        'generation_cap':budgets['fallback_inclusive_cap'],
                        'fit_cap':budgets['fit_calls']})
    source_exp = read(ROOT/'configs/experiment/slurp-tfrl-hardset-a0-observation-r1-v1.json')
    gpu_hours = source_exp['execution_plan']['planned_gpu_hours']
    result = {'schema':'saea-psd-construction-campaign-compiled-v1', 'campaign':config,
              'policy':policy, 'policy_sha256':sha(policy), 'members':members,
              'budget':{'normal_generation_calls':sum(m['normal_generation_calls'] for m in members),
                        'generation_cap':sum(m['generation_cap'] for m in members),
                        'fit_cap':sum(m['fit_cap'] for m in members),
                        'gpu_hours_each_generation_and_fit':gpu_hours,
                        'gpu_hours_all_generation_and_fit':2*gpu_hours*len(members)},
              'status':'prepared-not-admitted', 'new_model_calls':0}
    result['content_sha256'] = sha(result)
    return result


def write_compiled(config_path, out):
    doc = compile_campaign(read(config_path))
    out.mkdir(parents=True, exist_ok=False)
    for m in doc['members']:
        publish_new(out / (m['job_config']['name'] + '.json'), canonical(m['job_config']))
    publish_new(out / 'campaign.json', canonical(doc))
    return doc


def check_compiled(doc):
    if doc != compile_campaign(doc['campaign']):
        raise ValueError('compiled campaign differs from source configuration')
    return doc


def cpu_step(doc, index, mode, repo, data, out, stage_record=None, prepared=None):
    check_compiled(doc)
    if type(index) is not int or not 1 <= index <= len(doc['members']):
        raise ValueError('unknown campaign member')
    config = doc['members'][index-1]['job_config']
    if config.get('schema') == 'saea-perception-only-workload-v1':
        from p_only_workload import PerceptionOnlyWorkload
        job = PerceptionOnlyWorkload(config)
    else:
        job = NodeWorkload(config)
    if mode == 'stage':
        return job.stage(repo, data, job.config['material'], out)
    if mode == 'prepare':
        return job.prepare(repo, data, stage_record, out)
    if mode == 'register':
        return job.register(repo, data, stage_record, prepared)
    raise ValueError('unknown preparation step')


def queue_manifest(doc, queue_id, state_dir, launches):
    """Compose the existing queue schema from externally prepared launch pins.

    The caller supplies ordinary configured-job-launch pins after registration;
    this function neither creates launch permissions nor dispatches containers.
    """
    check_compiled(doc)
    if len(launches) != len(doc['members']):
        raise ValueError('one pinned launch per campaign member required')
    if not Path(state_dir).is_absolute():
        raise ValueError('absolute queue state directory required')
    jobs = []
    for member, item in zip(doc['members'], launches):
        if not isinstance(item, dict) or set(item) != {'launch','preflight'}:
            raise ValueError('launch and preflight pins required')
        for pin in (item['launch'], item['preflight']):
            if not isinstance(pin, dict) or set(pin) != {'path','sha256'} or not Path(pin['path']).is_absolute() or not re.fullmatch('[a-f0-9]{64}',pin['sha256']):
                raise ValueError('absolute immutable launcher/preflight pins required')
        jobs.append({'id':member['job_config']['name'], 'launch':item['launch'],
                     'preflight':item['preflight'], 'after':None, 'depends_on':[]})
    return {'schema':'saea-configured-job-queue-v1','queue_id':queue_id,
            'state_dir':state_dir,'monitor_interval_seconds':600,'jobs':jobs}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['compile','stage','prepare','register'])
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--repo', type=Path)
    p.add_argument('--data', type=Path)
    p.add_argument('--member', type=int)
    p.add_argument('--stage-record', type=Path)
    p.add_argument('--prepared', type=Path)
    a = p.parse_args()
    if a.mode == 'compile': result = write_compiled(a.config, a.out)
    else: result = cpu_step(read(a.config), a.member, a.mode, a.repo, a.data, a.out, a.stage_record, a.prepared)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
