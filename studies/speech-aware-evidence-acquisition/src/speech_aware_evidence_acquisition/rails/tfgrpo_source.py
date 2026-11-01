"""Load a pinned, source-audited dynamic TF-GRPO batch, without model calls."""
import hashlib
import json
from pathlib import Path
from .tfgrpo_learning import LearningError, sha, validate_experience
from .tfgrpo_prompts import SOURCE_SHA256, UPSTREAM_COMMIT
from .tfgrpo_runtime import validate_registration

SCHEMA='saea-prepared-tfgrpo-dynamic-v1'


def read_pin(pin, *, repo_root, data_root):
    if not isinstance(pin,dict) or set(pin)!={'root','relpath','sha256'} or pin['root'] not in ('repository','data'):
        raise LearningError('exact source pin required')
    root=Path(repo_root if pin['root']=='repository' else data_root).resolve()
    path=(root/pin['relpath']).resolve()
    if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=pin['sha256']:
        raise LearningError('learning source pin changed or escaped root')
    return json.loads(path.read_bytes())


def load_dynamic_source(values, *, plan, repo_root, data_root, sample_ids, roster_sha256, replay=None):
    spec=values.get('prepared_training_feedback')
    if not isinstance(spec,dict) or set(spec)!={'schema','source'} or spec['schema']!=SCHEMA:
        raise LearningError('dynamic TF-GRPO source contract required')
    reg=values.get('tfgrpo_learning_registration')
    model_id=reg.get('model_id') if isinstance(reg,dict) else None
    from .tfgrpo_runtime import learning_source_plan
    validate_registration(reg,plan=learning_source_plan(plan,replay=replay,data_root=data_root),model_id=model_id)
    if reg['construction_roster_sha256']!=roster_sha256:
        raise LearningError('dynamic source construction roster differs')
    source=read_pin(spec['source'],repo_root=repo_root,data_root=data_root)
    if (not isinstance(source,dict) or set(source)!={'schema','batch','provenance'}
            or source['schema']!='saea-tfgrpo-learning-source-v1'):
        raise LearningError('exact exported learning source required')
    batch=source['batch']
    if (batch.get('schema')!='saea-tfgrpo-batch-v1' or sha(batch)!=reg['batch_sha256']
            or batch.get('model_id')!=model_id or batch.get('learning_config_sha256')!=reg['learning_config_sha256']
            or batch.get('prompt_sha256')!=SOURCE_SHA256 or batch.get('upstream_commit')!=UPSTREAM_COMMIT
            or batch.get('rollout_kind') not in ('full-same-policy','shared-perception-adaptation')):
        raise LearningError('dynamic source batch or algorithm identity differs')
    validate_experience(batch['parent'],model_id)
    if not isinstance(source['provenance'],list) or not source['provenance']:
        raise LearningError('exported source requires immutable provenance pins')
    for pin in source['provenance']:read_pin(pin,repo_root=repo_root,data_root=data_root)
    imported={}
    if 'summary_import' in batch:
        from .tfgrpo_summary_import import load_import
        imported,_=load_import(batch['summary_import'],data_root=data_root,groups=batch['groups'],common=batch['common'],model_id=model_id)
    expected=[];seen=set();trajectory_ids=set()
    for group in batch['groups']:
        qid=group['query_id']
        if qid not in sample_ids or qid in seen or not qid.startswith('slurp:train:'):
            raise LearningError('learning query outside unique TRAIN roster')
        seen.add(qid)
        rolls=group['rollouts']
        if len(rolls)<2 or any(type(r['reward']) is not int or r['reward'] not in (0,1) for r in rolls):
            raise LearningError('binary same-policy rollouts required')
        for r in rolls:
            rid=r['trajectory_id']
            if rid in trajectory_ids or (r['model_id'],r['policy_sha256'],r['experience_sha256'])!=(model_id,batch['policy_sha256'],batch['parent']['sha256']):
                raise LearningError('rollout provenance differs')
            trajectory_ids.add(rid)
        if {r['reward'] for r in rolls}=={0,1}:
            expected.extend({'stage':'summary','identity':r['trajectory_id'],'sample_id':qid} for r in rolls if r['trajectory_id'] not in imported)
            expected.extend({'stage':s,'identity':qid,'sample_id':qid} for s in ('advantage','group_update'))
    if expected:expected.append({'stage':'batch_update','identity':'batch','sample_id':batch['groups'][0]['query_id']})
    if reg['contacts']!=expected or batch['bounds']['max_calls']!=len(expected):
        raise LearningError('finite learning contact roster differs from source batch')
    if set(sample_ids)!=seen:
        raise LearningError('dynamic source must cover the registered query roster')
    return source, {'schema':'saea-tfgrpo-dynamic-source-receipt-v1','source':spec['source'],
                   'batch_sha256':sha(batch),'contacts':len(expected),'rows':len(seen),
                   'construction_roster_sha256':roster_sha256,'dynamic_states':True}
