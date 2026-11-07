"""Configured external teacher for the original multistage TF-GRPO learner.

Continuous reward and offline rollout reuse are explicit local adaptations.
This module never issues native seals or admits a candidate for deployment.
"""
import copy
import hashlib
import json
from pathlib import Path

from .external_exploration import checked, pin, read, save
from .external_teacher import ExternalTeacher
from .grounded_synthesis_fusion import project_group_evidence
from .tfgrpo_learning import (DENSE_REWARD, BINARY_REWARD, ExperienceLearner, LearningError,
    canonical, group_has_contrast, intent_slot_reward, sha, validate_experience,
    semantic_advantage, parse_operations, batch_operation_limit)
from .tfgrpo_prompts import SOURCE_SHA256, UPSTREAM_COMMIT

SCHEMA = 'saea-tfgrpo-external-reference-job-v1'
STAGES = ('summary', 'advantage', 'group_update', 'batch_update')
ROOT = Path(__file__).resolve().parents[3]


def require(ok, message):
    if not ok:
        raise LearningError(message)


def teacher_references(reference_pin, ids, groups=None):
    """Validate the pinned private source against the complete TRAIN population."""
    source = read(checked(reference_pin))
    require(isinstance(source.get('provenance'), dict) and source['provenance'], 'reference provenance required')
    rows = source['document']['groups']
    require(isinstance(rows, list), 'reference groups required')
    index = {}
    for row in rows:
        qid = row['query_id']; evidence = row['evidence']
        require(isinstance(qid, str) and qid.startswith('slurp:train:') and qid not in index,
                'duplicate or non-TRAIN reference identity')
        require(evidence['query']['sample_id'] == qid and evidence['audio_available_to_reflector'] is False,
                'reference sample identity or audio boundary differs')
        feedback = evidence['reference_feedback_private']
        require(isinstance(feedback.get('reference_sentence'), str) and feedback['reference_sentence'].strip()
                and isinstance(feedback.get('intent'), str) and feedback['intent'].strip()
                and feedback.get('actor_input_contains_gold_feedback') is False,
                'original reference sentence and label required')
        label = {key: copy.deepcopy(feedback[key]) for key in
                 ('intent', 'historical_reference_intent', 'label_source_sha256', 'actor_input_contains_gold_feedback') if key in feedback}
        index[qid] = {'reference_sentence': feedback['reference_sentence'], 'original_label_feedback': label,
                      'audio_available_to_reflector': False,
                      'interpretation': 'Reference text is supplied supervision, not independent listening to the audio.'}
    require(len(ids) == len(set(ids)) and set(index) == set(ids), 'reference population missing or outside query IDs')
    if groups is not None:
        require({g['query_id'] for g in groups} == set(ids) and len(groups) == len(ids), 'original group population differs')
        for g in groups:
            require(json.loads(g['answer'])['gold_intent'] == index[g['query_id']]['original_label_feedback']['intent'],
                    'reference label differs from original group label')
    return index


def verified_material(config):
    fields = {'schema', 'source', 'source_batch', 'parent', 'query_ids', 'actor_model_id',
              'teacher', 'template', 'stage_completion_caps', 'bounds', 'objectives', 'output_root'}
    optional_reference = {'teacher_reference'} if isinstance(config, dict) and 'teacher_reference' in config else set()
    require(isinstance(config, dict) and set(config) - optional_reference in (fields, fields | {'psd_target'}, fields | {'psd_target','parent_binding'}, fields | {'p_only_binary'}) and config['schema'] == SCHEMA,
            'exact reference job configuration required')
    p_only = config.get('p_only_binary') is True
    require(not optional_reference or p_only, 'teacher reference is limited to P-only TF-GRPO')
    if optional_reference:
        checked(config['teacher']['research_prompt'])
    require('p_only_binary' not in config or p_only, 'explicit P binary mode required')
    if 'psd_target' in config:
        target = config['psd_target']
        require(target in ('slot','decision'), 'explicit PSD target required')
        policy = ROOT / ('configs/prompts/learner/tfrl/tfgrpo-reference-psd-' + target + '.md')
        require(checked(config['teacher']['research_prompt']).read_bytes() == policy.read_bytes(), 'fixed reference PSD policy required')
        from .tfgrpo_psd_parent import verify as verify_parent
        verify_parent(read(checked(config['parent'])), target, config['actor_model_id'], config.get('parent_binding'))
    source = read(checked(config['source'])); document = source['document']
    batch = read(checked(config['source_batch'])); parent = read(checked(config['parent']))
    actor = config['actor_model_id']
    validate_experience(parent, actor)
    require(document['source_batch_sha256'] == sha(batch), 'original batch digest differs')
    require(batch['model_id'] == document['model_id'] == actor
            and batch['parent'] == document['parent'] == parent
            and document['rollout_parent_sha256'] == parent['sha256']
            and document['rollout_policy_sha256'] == batch['policy_sha256']
            and document['rollout_kind'] == batch['rollout_kind'], 'original actor, parent or policy differs')
    mode = BINARY_REWARD if p_only else DENSE_REWARD
    require(document['reward_mode'] == batch['common'].get('reward_mode', BINARY_REWARD) == mode, 'source reward mode differs')
    template = config['template']
    require(template == {'upstream_commit': UPSTREAM_COMMIT, 'source_sha256': SOURCE_SHA256},
            'original stage template version differs')
    bounds = config['bounds']
    require(set(bounds) == {'max_calls', 'concurrency', 'max_operations', 'max_items',
                          'max_characters', 'num_experiences', 'max_completion_tokens_total'}
            and all(type(v) is int and v > 0 for v in bounds.values())
            and bounds['num_experiences'] <= bounds['max_operations'], 'explicit positive learning bounds required')
    teacher = config['teacher']
    require(not {'messages', 'output_root'} & teacher.keys(), 'teacher input comes from stage execution')
    caps = config['stage_completion_caps']
    require(set(caps) == set(STAGES) and all(type(v) is int and v >= 131072 for v in caps.values())
            and set(caps.values()) == {teacher['max_tokens']},
            'each stage must explicitly match the fixed teacher completion cap >=128K')
    require(isinstance(teacher.get('analysis_budget'), dict)
            and teacher['analysis_budget'].get('max_reasoning_tokens_target') == 49152,
            'explicit 49152 advisory reasoning budget required')
    objectives = config['objectives']
    require(set(objectives) == {'agent_objective', 'learning_objective'}
            and all(isinstance(v, str) and v.strip() for v in objectives.values()), 'explicit objectives required')
    ids = config['query_ids']
    require(isinstance(ids, list) and ids and all(isinstance(q, str) and q.startswith('slurp:train:') for q in ids)
            and len(ids) == len(set(ids)), 'explicit distinct TRAIN query identities required')
    originals = {g['query_id']: g for g in batch['groups']}
    projected = {g['query_id']: g for g in document['groups']}
    require(len(originals) == len(batch['groups']) and len(projected) == len(document['groups'])
            and set(ids) <= originals.keys() & projected.keys(), 'missing or duplicate original groups')
    groups = []; seen = set()
    for qid in ids:
        group = originals[qid]; expected = projected[qid]
        require(expected['group_sha256'] == sha(group), 'original group digest differs')
        require(expected['evidence'] == project_group_evidence(group, evidence_view='full-query-v1'),
                'full source evidence differs from original trajectory projection')
        require(set(group) == {'query_id', 'query', 'answer', 'rollouts'}, 'exact original group required')
        query = json.loads(group['query'])
        require(query['sample_id'] == qid and query.get('audio_available_to_reflector') is False
                and {'supplied_evidence', 'task_instruction', 'history'} <= query.keys(), 'full actual query required')
        require(len(group['rollouts']) >= 2, 'multiple original trajectories required')
        for row in group['rollouts']:
            require(set(row) == {'trajectory_id', 'trajectory', 'critique', 'reward', 'model_id',
                                'policy_sha256', 'experience_sha256'}, 'exact original rollout required')
            require(row['trajectory_id'] not in seen and row['model_id'] == actor
                    and row['policy_sha256'] == batch['policy_sha256']
                    and row['experience_sha256'] == parent['sha256'], 'rollout identity or provenance differs')
            seen.add(row['trajectory_id'])
            feedback = json.loads(row['critique']); trajectory = json.loads(row['trajectory'])
            require(trajectory['request']['sample_id'] == qid
                    and {'task_instruction', 'supplied_evidence', 'history'} <= trajectory['request'].keys()
                    and isinstance(trajectory['response_text'], str), 'actual source request and response required')
            if p_only:
                require(type(row['reward']) is int and row['reward'] in (0, 1)
                        and type(feedback['recall3']) is bool and int(feedback['recall3']) == row['reward']
                        and isinstance(feedback['legal'], bool) and isinstance(feedback['top1'], bool), 'P recall feedback differs')
                require(trajectory['response_text'] == trajectory['scored_answer_after_fixed_normalization'], 'P delivered response differs')
            else:
                require(feedback['reward_mode'] == DENSE_REWARD and type(row['reward']) in (int, float)
                        and intent_slot_reward(feedback) == feedback['reward'] == row['reward'], 'source reward differs')
                require(hashlib.sha256(trajectory['scored_answer_after_fixed_normalization'].encode()).hexdigest()
                        == feedback['scored_answer_sha256'], 'source scored response digest differs')
        groups.append(copy.deepcopy(group))
    if optional_reference:
        require(set(originals) == set(ids) == set(projected), 'complete original population required for reference')
        teacher_references(config['teacher_reference'], ids, groups)
    contrast = [g for g in groups if group_has_contrast(g['rollouts'], BINARY_REWARD if config.get('p_only_binary') else DENSE_REWARD)]
    counts = {'summary': sum(len(g['rollouts']) for g in contrast), 'advantage': len(contrast),
              'group_update': len(contrast), 'batch_update': int(bool(contrast))}
    calls = sum(counts.values()); allocation = sum(counts[s] * caps[s] for s in STAGES)
    require(calls <= bounds['max_calls'] and allocation <= bounds['max_completion_tokens_total'],
            'complete stage plan exceeds explicit call or completion allocation')
    return groups, parent, batch, {'groups': len(groups), 'contrast_groups': len(contrast),
        'rollouts': len(seen), 'stage_calls': counts, 'planned_calls': calls,
        'maximum_completion_allocation': allocation, 'provider_capacity_verified': False,
        'input_tokens': None, 'duplicate_rollouts_preserved': True}


def prepare(config_path):
    config = read(config_path)
    groups, parent, batch, plan = verified_material(config)
    root = Path(config['output_root']).resolve()
    require(not root.exists(), 'new preparation directory required')
    # Snapshot the actual generic research policy and standard executor before
    # any request. Stage templates remain versioned upstream templates.
    policy = checked(config['teacher']['research_prompt'])
    root.mkdir(parents=True, mode=0o700)
    (root/'policy.md').write_bytes(policy.read_bytes())
    (root/'executor.py').write_bytes((ROOT/'scripts/environment/configured_external_learner.py').read_bytes())
    teacher = {**config['teacher'], 'research_prompt': pin(root/'policy.md')}
    caller = ExternalTeacher(root/'teacher', config=teacher, executor=pin(root/'executor.py'))
    save(root/'job.json', config); save(root/'teacher.json', teacher); save(root/'plan.json', plan)
    save(root/'groups.json', groups)
    manifest = {'schema': 'saea-tfgrpo-external-reference-prepared-v1',
        'files': {n: pin(root/n) for n in ('job.json', 'teacher.json', 'plan.json', 'groups.json', 'policy.md', 'executor.py')},
        'learning_config_sha256': caller.learning_config_sha256,
        'source_batch_sha256': sha(batch), 'parent_sha256': parent['sha256'],
        'actor_model_id': config['actor_model_id'], 'learner_model_id': caller.model_id,
        'template': config['template'], 'plan': plan, 'new_model_calls': 0,
        'algorithm': ('tfgrpo-reference-multistage-external-p-binary-v1' if config.get('p_only_binary') else 'tfgrpo-reference-multistage-external-dense-offline-v1'),
        'evidence_scope': {'summary': 'full original query and losslessly factored original trajectory',
            'advantage': 'original query, TRAIN reference and model summaries with original rewards',
            'group_update': 'parent experiences and model semantic advantage',
            'batch_update': 'parent experiences and proposed operations; no silent grounded augmentation'},
        'native_learning_seal': False, 'deployment_accepted': False}
    save(root/'prepared.json', manifest)
    return manifest


def load(root):
    root = Path(root).resolve(); manifest = read(root/'prepared.json')
    require(manifest['schema'] == 'saea-tfgrpo-external-reference-prepared-v1', 'reference preparation required')
    for name, value in manifest['files'].items():
        require(checked(value) == root/name, 'prepared file path differs')
    config = read(root/'job.json'); groups, parent, batch, plan = verified_material(config)
    require(groups == read(root/'groups.json') and plan == read(root/'plan.json') == manifest['plan'],
            'prepared source or stage plan changed')
    caller = ExternalTeacher(root/'teacher', config=read(root/'teacher.json'), executor=manifest['files']['executor.py'])
    require(caller.learning_config_sha256 == manifest['learning_config_sha256']
            and caller.model_id == manifest['learner_model_id']
            and config['actor_model_id'] == manifest['actor_model_id']
            and sha(batch) == manifest['source_batch_sha256']
            and parent['sha256'] == manifest['parent_sha256']
            and config['template'] == manifest['template'], 'prepared identities changed')
    return root, manifest, config, groups, parent, batch, caller


def execute(root, *, collect_only=False):
    root, manifest, config, groups, parent, batch, caller = load(root)
    def no_dispatch(request):
        raise LearningError('collect never starts missing contacts')
    def complete_response(response):
        require(response.get('finish_reason') == 'stop', 'incomplete external reference response')
        return response
    def call(request):
        return complete_response(caller(request))
    def reconcile(request):
        observed = caller.reconcile(request)
        if observed is not None:
            complete_response(observed['response'])
        return observed
    bounds = config['bounds']
    learner = ExperienceLearner(root/'learner', model_id=caller.model_id,
        metered_call=no_dispatch if collect_only else call,
        preflight=no_dispatch if collect_only else caller.preflight, reconcile=reconcile,
        learning_config_sha256=caller.learning_config_sha256,
        **{k: bounds[k] for k in ('max_calls', 'concurrency', 'max_operations', 'max_items', 'max_characters')})
    if collect_only:
        require((root/'learner/candidate.json').is_file(), 'no completed reference candidate to collect')
    candidate = learner.learn(groups, parent, **config['objectives'], actor_model_id=config['actor_model_id'],
        policy_sha256=batch['policy_sha256'], rollout_kind=batch['rollout_kind'],
        num_experiences=bounds['num_experiences'], reward_mode=BINARY_REWARD if config.get('p_only_binary') else DENSE_REWARD,
        teacher_diagnostics=teacher_references(config['teacher_reference'], config['query_ids'], groups) if 'teacher_reference' in config else None)
    # Reconcile every complete reference contact against original provider wire.
    receipts = []
    for path in sorted((root/'learner/contacts').glob('*.intent.json')):
        request = read(path); response_path = path.with_name(path.name.replace('.intent.json', '.response.json'))
        require(response_path.exists(), 'incomplete reference contact')
        observed = reconcile(request)
        require(observed is not None and observed == read(response_path), 'reference contact differs from provider wire')
        receipts.append({'stage': request['stage'], 'identity': request['identity'],
                         'request_sha256': sha(request), 'wire_response_sha256': observed['response']['wire_response_sha256']})
    result = {'schema': 'saea-tfgrpo-external-reference-collected-v1',
        'prepared': pin(root/'prepared.json'), 'candidate': pin(root/'learner/candidate.json'),
        'algorithm': manifest['algorithm'], 'actor_model_id': config['actor_model_id'],
        'learner_model_id': caller.model_id, 'contacts': receipts, 'accounting': learner.accounting(),
        'candidate_sha256': candidate['sha256'], 'native_learning_seal': False,
        'semantic_validation': 'pending', 'deployment_accepted': False}
    path = root/'collected.json'
    if path.exists():
        require(read(path) == result, 'collected reference bytes changed')
    else:
        save(path, result)
    return result


def run(root):
    return execute(root)


def collect(root):
    return execute(root, collect_only=True)


def audit(root):
    """Read-only custody and stage parsing of a failed or incomplete batch.

    Does not collect, create a learner lock, reconcile into learner contacts,
    issue requests, or claim that unobserved stages have completed.
    """
    root, manifest, config, groups, parent, batch, caller = load(root)
    contacts = root/'learner/contacts'
    intents = sorted(contacts.glob('*.intent.json'))
    batch_path = root/'learner/batch.json'
    require(not intents or batch_path.is_file(), 'contact batch binding missing')
    batch_hash = sha(read(batch_path)) if batch_path.is_file() else None
    expected_ids = {'summary': {r['trajectory_id'] for g in groups for r in g['rollouts']},
                    'advantage': {g['query_id'] for g in groups},
                    'group_update': {g['query_id'] for g in groups}, 'batch_update': {'batch'}}
    for response_path in contacts.glob('*.response.json'):
        require(response_path.with_name(response_path.name.replace('.response.json', '.intent.json')).is_file(),
                'orphan reference response')
    # Audit the exact same batch capacity as execution. Only validated, observed
    # group outputs can contribute capacity; an invalid/missing group cannot.
    proposal_count = 0
    valid_group_ids = set()
    for path in intents:
        request = read(path)
        if request['stage'] != 'group_update':
            continue
        observed = caller.reconcile(request)
        if observed is not None and observed['response']['finish_reason'] == 'stop':
            try:
                proposals = parse_operations(observed['response']['text'], parent['items'],
                                            max_operations=config['bounds']['max_operations'])
            except LearningError:
                continue
            proposal_count += len(proposals)
            valid_group_ids.add(request['identity'])
    receipts = []; counts = {stage: 0 for stage in STAGES}
    usage = dict(calls=0, prompt_tokens=0, completion_tokens=0)
    for path in intents:
        request = read(path)
        require(request['stage'] in counts, 'unknown reference stage')
        require(request['batch_sha256'] == batch_hash
                and request['identity'] in expected_ids[request['stage']], 'reference batch or stage identity differs')
        require(path.stem.removesuffix('.intent') == sha({'stage': request['stage'], 'identity': request['identity']}),
                'reference contact filename differs')
        response_path = path.with_name(path.name.replace('.intent.json', '.response.json'))
        observed = caller.reconcile(request)
        item = {'stage': request['stage'], 'identity': request['identity'],
                'request_sha256': sha(request), 'learner_response_present': response_path.is_file()}
        if observed is None:
            item.update(state='unknown', parse_status='unavailable')
        else:
            if response_path.is_file():
                require(read(response_path) == observed, 'learner response differs from original wire')
            response = observed['response']
            item.update(state='observed', finish_reason=response['finish_reason'],
                wire_response_sha256=response['wire_response_sha256'])
            for k in usage: usage[k] += response['usage'][k]
            try:
                require(response['finish_reason'] == 'stop', 'incomplete external reference response')
                if request['stage'] == 'advantage':
                    semantic_advantage(response['text'])
                elif request['stage'] in ('group_update', 'batch_update'):
                    limit = config['bounds']['max_operations']
                    if request['stage'] == 'batch_update':
                        require(valid_group_ids == {g['query_id'] for g in groups
                                if group_has_contrast(g['rollouts'], BINARY_REWARD if config.get('p_only_binary') else DENSE_REWARD)},
                                'batch requires complete valid group proposals')
                        limit = batch_operation_limit(proposal_count, limit)
                    parse_operations(response['text'], parent['items'],
                                     max_operations=limit)
                item['parse_status'] = 'valid'
            except (LearningError, ValueError) as error:
                item.update(parse_status='invalid', parse_error=str(error))
        counts[request['stage']] += 1
        receipts.append(item)
    unknown = sum(r['state'] == 'unknown' for r in receipts)
    invalid = sum(r['parse_status'] == 'invalid' for r in receipts)
    return {'schema': 'saea-tfgrpo-reference-contact-audit-v1',
        'prepared': pin(root/'prepared.json'), 'plan': manifest['plan'],
        'contacts': receipts, 'stage_contacts': counts, 'known_usage': usage,
        'unknown_contacts': unknown, 'invalid_stage_responses': invalid,
        'unissued_stage_calls': {s: manifest['plan']['stage_calls'][s]-counts[s] for s in STAGES},
        'candidate_present': (root/'learner/candidate.json').is_file(),
        'known_contacts_replayable': unknown == 0 and invalid == 0,
        'new_model_calls': 0, 'source_modified': False, 'automatic_retry': False,
        'deployment_accepted': False,
        'limitation': 'Requires terminal/quiescent process evidence before resume; stage audit is not a completed candidate.'}
