"""Load a pinned, source-audited dynamic TF-GRPO batch, without model calls."""
import hashlib
import json
from pathlib import Path
from .tfgrpo_learning import (LearningError, sha, validate_experience, BINARY_REWARD,
    validate_reward_mode, group_has_contrast)
from .tfgrpo_prompts import SOURCE_SHA256, UPSTREAM_COMMIT
from .tfgrpo_runtime import validate_registration

SCHEMA='saea-prepared-tfgrpo-dynamic-v1'


def read_pin(pin, *, repo_root, data_root, verified_context=None):
    if not isinstance(pin,dict) or set(pin)!={'root','relpath','sha256'} or pin['root'] not in ('repository','data'):
        raise LearningError('exact source pin required')
    root=Path(repo_root if pin['root']=='repository' else data_root).resolve()
    path=(root/pin['relpath']).resolve()
    if not path.is_relative_to(root):
        raise LearningError('learning source pin changed or escaped root')
    if verified_context is not None:
        _, document = verified_context.read(pin, kind=pin['root'])
        return document
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=pin['sha256']:
        raise LearningError('learning source pin changed or escaped root')
    return json.loads(blob)


def load_dynamic_source(values, *, plan, repo_root, data_root, sample_ids, roster_sha256,
                        replay=None, verified_context=None):
    spec=values.get('prepared_training_feedback')
    if isinstance(spec,dict) and spec.get('schema')=='saea-prepared-grounded-fusion-v1':
        from .grounded_fusion_runtime import load_runtime_source
        return load_runtime_source(values,plan=plan,repo_root=repo_root,data_root=data_root,
            sample_ids=sample_ids,roster_sha256=roster_sha256,replay=replay)
    if isinstance(spec,dict) and spec.get('schema')=='saea-prepared-grounded-shards-v1':
        from .grounded_synthesis_runtime import load_runtime_source
        return load_runtime_source(values,plan=plan,repo_root=repo_root,data_root=data_root,
            sample_ids=sample_ids,roster_sha256=roster_sha256,replay=replay)
    if not isinstance(spec,dict) or set(spec)!={'schema','source'} or spec['schema']!=SCHEMA:
        raise LearningError('dynamic TF-GRPO source contract required')
    reg=values.get('tfgrpo_learning_registration')
    model_id=reg.get('model_id') if isinstance(reg,dict) else None
    from .tfgrpo_runtime import learning_source_plan
    validate_registration(reg,plan=learning_source_plan(plan,replay=replay,data_root=data_root),model_id=model_id)
    if reg['construction_roster_sha256']!=roster_sha256:
        raise LearningError('dynamic source construction roster differs')
    source=read_pin(spec['source'],repo_root=repo_root,data_root=data_root,
                    verified_context=verified_context)
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
    for pin in source['provenance']:
        read_pin(pin,repo_root=repo_root,data_root=data_root,verified_context=verified_context)
    reward_mode = batch['common'].get('reward_mode', BINARY_REWARD)
    validate_reward_mode(reward_mode)
    if reward_mode != BINARY_REWARD and 'summary_import' in batch:
        raise LearningError('dense reward cannot import binary summaries or stages')
    final_only='final_import' in batch
    if final_only:
        if 'summary_import' in batch:
            raise LearningError('final-only source cannot also import summaries')
        from .tfgrpo_final_import import load_import
        load_import(batch['final_import'],data_root=data_root,groups=batch['groups'],
            common=batch['common'],model_id=model_id,parent=batch['parent'],
            policy_sha256=batch['policy_sha256'],rollout_kind=batch['rollout_kind'],
            verified_context=verified_context)
    imported={}
    if 'summary_import' in batch:
        from .tfgrpo_summary_import import load_import
        imported,_=load_import(batch['summary_import'],data_root=data_root,groups=batch['groups'],common=batch['common'],model_id=model_id,
                               verified_context=verified_context)
    expected=[];seen=set();trajectory_ids=set()
    for group in batch['groups']:
        qid=group['query_id']
        if (not final_only and qid not in sample_ids) or qid in seen or not qid.startswith('slurp:train:'):
            raise LearningError('learning query outside unique TRAIN roster')
        seen.add(qid)
        rolls=group['rollouts']
        if len(rolls)<2:
            raise LearningError('at least two same-policy rollouts required')
        contrast = group_has_contrast(rolls, reward_mode)
        for r in rolls:
            rid=r['trajectory_id']
            if rid in trajectory_ids or (r['model_id'],r['policy_sha256'],r['experience_sha256'])!=(model_id,batch['policy_sha256'],batch['parent']['sha256']):
                raise LearningError('rollout provenance differs')
            trajectory_ids.add(rid)
        if contrast:
            expected.extend({'stage':'summary','identity':r['trajectory_id'],'sample_id':qid} for r in rolls if r['trajectory_id'] not in imported)
            expected.extend({'stage':s,'identity':qid,'sample_id':qid} for s in ('advantage','group_update'))
    if expected:expected.append({'stage':'batch_update','identity':'batch','sample_id':batch['groups'][0]['query_id']})
    if final_only:
        if not expected:raise LearningError('final-only source requires completed mixed groups')
        expected=[expected[-1]]
    if reg['contacts']!=expected or batch['bounds']['max_calls']!=len(expected):
        raise LearningError('finite learning contact roster differs from source batch')
    required_ids={batch['groups'][0]['query_id']} if final_only else seen
    if set(sample_ids)!=required_ids:
        raise LearningError('dynamic source must cover the registered query roster')
    return source, {'schema':'saea-tfgrpo-dynamic-source-receipt-v1','source':spec['source'],
                   'batch_sha256':sha(batch),'contacts':len(expected),'rows':len(required_ids),
                   'construction_roster_sha256':roster_sha256,'dynamic_states':True}
