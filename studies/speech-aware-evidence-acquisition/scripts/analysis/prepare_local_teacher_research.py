"""Prepare a pinned research batch offline and emit the existing local queue config.

Algorithm templates and editable context are supplied as files. This command never
calls a model; it only reuses the existing external_exploration.prepare adapter.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from speech_aware_evidence_acquisition.rails.external_exploration import checked, pin, prepare, read, request_payload


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utterance(sid):
    parts = sid.split(':')
    require(len(parts) >= 4, 'structured sample identity required')
    return ':'.join(parts[:3])


def audit_partition(ids, excluded, dataset):
    require(isinstance(ids, list) and ids and len(ids) == len(set(ids)), 'unique explicit learning roster required')
    require(all(s.startswith('slurp:train:') for s in ids), 'TRAIN-only learning roster required')
    index = {r['sample_id']: r for r in dataset['items']}
    require(set(ids) <= set(index), 'learning IDs absent from pinned dataset')
    require(set(excluded) <= set(index), 'exclusion audio identities absent from pinned datasets')
    known = list(excluded)
    overlap = {
        'query': len(set(ids) & set(excluded)),
        'utterance': len({utterance(s) for s in ids} & {utterance(s) for s in excluded}),
        'audio': len({index[s]['audio_sha256'] for s in ids} & {index[s]['audio_sha256'] for s in known}),
    }
    require(not any(overlap.values()), 'learning overlaps excluded query, utterance or audio')
    return {'learning_rows': len(ids), 'excluded_rows': len(excluded), 'overlap': overlap,
            'excluded_audio_ids_resolved': len(known),
            'scope': 'Learning query/utterance/audio isolation; shared historical knowledge is not an unseen-data guarantee.'}


def build(config_path):
    cfg = read(config_path)
    require(set(cfg) == {'schema','output_root','queue_id','max_parallel','datasets','excluded_ids','jobs'}, 'exact batch config required')
    require(cfg['schema'] == 'saea-local-teacher-research-batch-v1', 'batch schema required')
    require(type(cfg['max_parallel']) is int and 1 <= cfg['max_parallel'] <= 8, 'bounded concurrency required')
    require(isinstance(cfg['jobs'], list) and cfg['jobs'], 'nonempty job list required')
    require(isinstance(cfg['datasets'], list) and cfg['datasets'], 'pinned dataset inventories required')
    dataset = {'items': [item for binding in cfg['datasets'] for item in read(checked(binding))['items']]}
    require(len({r['sample_id'] for r in dataset['items']}) == len(dataset['items']), 'duplicate dataset identities')
    excluded = read(checked(cfg['excluded_ids']))
    require(isinstance(excluded, list) and len(excluded)==len(set(excluded)), 'unique exclusion IDs required')
    jobs=[];names=set()
    # Validate all inputs before creating any preparation output.
    for job in cfg['jobs']:
        require(set(job)=={'id','job','prior_prepared'}, 'exact job binding required')
        name=job['id'];require(isinstance(name,str) and name.replace('-','').replace('_','').isalnum() and name not in names, 'unique safe job ID required');names.add(name)
        source_job=read(checked(job['job']))
        require(source_job['schema']=='saea-external-exploration-job-v1', 'existing exploration job required')
        profile=read(checked(source_job['profile']));checked(profile['template'],checked(source_job['profile']).parent)
        source=read(checked(source_job['source']));checked(source_job['context'])
        ids=source_job['query_ids'];audit=audit_partition(ids,excluded,dataset)
        require(set(ids)<={g['query_id'] for g in source['document']['groups']}, 'missing evidence groups')
        require(profile['max_tokens']>=131072, '128K output budget required')
        priors=[checked(p) for p in job['prior_prepared']]
        for p in priors:require(p.name=='prepared.json','prior prepared manifest required')
        jobs.append((job,source_job,profile,audit,priors))
    out=Path(cfg['output_root']);require(out.is_absolute() and not out.exists(),'fresh absolute output required')
    out.mkdir(parents=True,mode=0o700)
    module_path=ROOT/'scripts/environment/configured_local_teacher_queue.py'
    spec=importlib.util.spec_from_file_location('local_queue',module_path);queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
    cli=ROOT/'scripts/analysis/external_exploration.py'
    queue.persist(out/'implementation.json',queue.implementation_inventory(cli))
    queue_jobs=[];audits=[]
    for job,source_job,profile,audit,priors in jobs:
        directory=out/job['id'];directory.mkdir(mode=0o700)
        derived=dict(source_job,output_root=str(directory/'prepared'))
        queue.persist(directory/'job.json',derived)
        manifest=prepare(directory/'job.json');prepared=directory/'prepared/prepared.json'
        request=read(prepared.parent/'request.json');payload=request_payload(request)
        require([g['query_id'] for g in payload['groups']]==derived['query_ids'],'prepared roster differs')
        prior_ids=[]
        for path in priors:
            record=read(path);request_path=checked(record['files']['request.json'])
            prior_ids.append(read(request_path)['identity'])
        require(request['identity'] not in prior_ids,'duplicate prior research request')
        audit.update(algorithm=profile['algorithm'],groups=manifest['groups'],rollouts=manifest['rollouts'],
                     request_bytes=(prepared.parent/'request.json').stat().st_size,
                     request_sha256=pin(prepared.parent/'request.json')['sha256'],request_identity=request['identity'],
                     template_sha256=pin(prepared.parent/'template.md')['sha256'],
                     max_tokens=profile['max_tokens'],prior_request_identities=prior_ids,
                     new_model_calls=0,prepared=pin(prepared))
        audits.append(audit)
        queue_jobs.append({'id':job['id'],'kind':'external-exploration','cli':pin(cli),
                           'prepared':pin(prepared),'implementation':pin(out/'implementation.json'),'depends_on':[]})
    queue_config={'schema':queue.SCHEMA,'queue_id':cfg['queue_id'],'state_dir':str(out/'queue-state'),
                  'max_parallel':cfg['max_parallel'],'jobs':queue_jobs}
    queue.persist(out/'queue.json',queue_config)
    # Use the production queue validator offline, including implementation/source identities.
    queue.load(out/'queue.json')
    result={'schema':'saea-local-teacher-research-preparation-v1','new_model_calls':0,
            'configuration':pin(Path(config_path).resolve()),'jobs':audits,'queue':pin(out/'queue.json'),
            'dispatch_started':False,'candidate_accepted':False}
    queue.persist(out/'input-audit.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path)
    print(json.dumps(build(p.parse_args().config),ensure_ascii=False))
