"""Audit a stopped learner's completed stages before one new final merge.

The import is immutable historical input, not replayed generation. Every native
response (including the unusable final response) retains its original cost.
"""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from ..contracts import ExecutionPlan
from ..core.contact_journal import ContactJournal, NativeContactReplay, comparable_request
from .tfgrpo_learning import (LearningError, canonical, parse_operations, render,
                              semantic_advantage, sha, validate_experience)
from .tfgrpo_projection import summary_trajectory
from .tfgrpo_prompts import SOURCE_SHA256, UPSTREAM_COMMIT


def load_import(spec, *, data_root, groups, common, model_id, parent,
                policy_sha256, rollout_kind):
    root = Path(data_root).resolve()

    def pinned(pin):
        if not isinstance(pin, dict) or set(pin) != {'relpath', 'sha256'}:
            raise LearningError('exact final-stage input pin required')
        path = (root / pin['relpath']).resolve()
        if not path.is_relative_to(root):
            raise LearningError('final-stage input escaped data root')
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != pin['sha256']:
            raise LearningError('final-stage input changed')
        return path, json.loads(blob)

    _, doc = pinned(spec)
    if (not isinstance(doc, dict) or set(doc) != {'schema', 'source_plan', 'source_batch',
            'native_sources', 'records', 'source_resume', 'source_recovery'}
            or doc['schema'] != 'saea-tfgrpo-final-import-v1'):
        raise LearningError('exact final-stage import manifest required')
    plan = ExecutionPlan(**doc['source_plan']); plan.validate()
    base = root / 'runs' / plan.run_id / plan.attempt_id / 'tfgrpo'
    batch_path, batch = pinned(doc['source_batch'])
    if (batch_path != base / 'learner/batch.json'
            or batch.get('schema') != 'saea-tfgrpo-batch-v1'
            or batch.get('prompt_sha256') != SOURCE_SHA256
            or batch.get('upstream_commit') != UPSTREAM_COMMIT
            or any(batch.get(k) != v for k, v in {'groups': groups, 'common': common,
                'model_id': model_id, 'parent': parent, 'policy_sha256': policy_sha256,
                'rollout_kind': rollout_kind}.items())):
        raise LearningError('historical final-stage learning inputs differ')
    validate_experience(parent, model_id)
    sources = doc['native_sources']
    if not isinstance(sources, list) or not sources:
        raise LearningError('historical native sources required')
    observed = {}; identities = set(); totals = dict(calls=0, prompt_tokens=0, completion_tokens=0)
    native_plans = []; source_costs = []
    for source in sources:
        if set(source) != {'plan', 'attempt', 'journal_files'}:
            raise LearningError('exact native source required')
        native = ExecutionPlan(**source['plan']); native.validate()
        identity = (native.run_id, native.attempt_id)
        if identity in identities:
            raise LearningError('duplicate native source would double-count cost')
        identities.add(identity); native_plans.append(native)
        ap, attempt = pinned(source['attempt'])
        if (ap != root / 'attempts' / native.run_id / (native.attempt_id + '.json')
                or attempt.get('state') != 'ABORTED' or attempt.get('contact_kind') != 'prepared-text'
                or (attempt.get('run_id'), attempt.get('attempt_id')) != identity
                or attempt.get('carrier_lock_key') != native.carrier_lock_key
                or attempt.get('reserved') != {'calls': native.planned_model_calls,
                    'audio_seconds': float(native.planned_speech_audio_seconds)}):
            raise LearningError('historical final source must be explicitly aborted')
        journal = ContactJournal(root, native, 'prepared-context-text')
        paths = [pinned(pin)[0] for pin in source['journal_files']]
        actual_paths = set(journal.root.glob('*.intent.json')) | set(journal.root.glob('*.response.json'))
        if len(paths) != len(set(paths)) or set(paths) != actual_paths:
            raise LearningError('complete pinned native journal required')
        usage = dict(calls=0, prompt_tokens=0, completion_tokens=0)
        for path in sorted(journal.root.glob('*.intent.json')):
            intent = json.loads(path.read_bytes()); rid = intent['request_id']
            record = journal.inspect(rid)
            if path != journal._path(rid, 'intent') or record['state'] != 'observed':
                raise LearningError('historical generation outcome unknown')
            body = json.loads(record['response_bytes'])
            if intent['payload'].get('prepared_text') != 'saea-prepared-text-v1':
                raise LearningError('historical source must be prepared text')
            key = (rid, record['response']['response_sha256'])
            if key in observed:
                raise LearningError('ambiguous historical native response')
            observed[key] = (record, body, identity)
            usage['calls'] += 1
            for field in ('prompt_tokens', 'completion_tokens'):
                value = body['usage'][field]
                if type(value) is not int or value < 0:
                    raise LearningError('historical token usage invalid')
                usage[field] += value
        if (len(actual_paths) != usage['calls'] * 2
                or attempt.get('actual') != {'calls': usage['calls'], 'audio_seconds': 0.0}):
            raise LearningError('historical native cost or response inventory differs')
        for field in totals: totals[field] += usage[field]
        source_costs.append({'run_id': native.run_id, 'attempt_id': native.attempt_id, 'usage': usage})
    if native_plans[0] != plan:
        raise LearningError('first native source must be the stopped learner')
    resume_path = base.parent / 'resume-declaration.json'
    if resume_path.exists():
        rp, resume = pinned(doc['source_resume']); cp, recovery = pinned(doc['source_recovery'])
        if (rp != resume_path or len(native_plans) != 2
                or resume.get('schema') != 'saea-resume-declaration-v1'
                or resume.get('run_id') != plan.run_id or resume.get('attempt_id') != plan.attempt_id
                or recovery.get('schema') != 'saea-native-recovery-v1'
                or recovery.get('target_plan') != asdict(plan)
                or resume.get('native_recovery_receipt') != {'path': str(cp), 'sha256': doc['source_recovery']['sha256']}):
            raise LearningError('historical recovery ancestry differs')
        replay = NativeContactReplay(root, native_plans[1], plan, 'prepared-context-text')
        binding = replay.source_binding()
        if (recovery.get('source') != binding or resume.get('predecessors') != [{
                'attempt_id': native_plans[1].attempt_id, 'mode': 'native-contact-journal',
                'pin_sha256': binding['attempt_sha256']}]):
            raise LearningError('historical replay source custody differs')
    elif doc['source_resume'] is not None or doc['source_recovery'] is not None or len(native_plans) != 1:
        raise LearningError('unexplained native ancestry')

    summaries = {}; summary_custody = None
    if 'summary_import' in batch:
        from .tfgrpo_summary_import import load_import as load_summaries
        summaries, summary_custody = load_summaries(batch['summary_import'], data_root=root,
            groups=groups, common=common, model_id=model_id)
        if (summary_custody['source_run_id'], summary_custody['source_attempt_id']) in identities:
            raise LearningError('summary source overlaps native cost ancestry')
        for field in totals: totals[field] += summary_custody['original_total_usage'][field]
    rows = {}
    for row in doc['records']:
        if set(row) != {'stage', 'identity', 'intent', 'response', 'bridge_intent', 'bridge_response'}:
            raise LearningError('exact completed stage pins required')
        key = (row['stage'], row['identity'])
        if key in rows: raise LearningError('duplicate historical learning stage')
        rows[key] = row
    expected_paths = set(); used = set(); truncated_summaries = []

    def read_stage(stage, identity, sample_id, messages, *, finished=True):
        row = rows.pop((stage, identity), None)
        if row is None: raise LearningError('historical learning stage missing')
        key = sha({'stage': stage, 'identity': identity})
        ip, request = pinned(row['intent']); rp, envelope = pinned(row['response'])
        bp, bridge = pinned(row['bridge_intent']); brp, bridge_response = pinned(row['bridge_response'])
        if [ip, rp, bp, brp] != [base/'learner/contacts'/(key+'.intent.json'),
                base/'learner/contacts'/(key+'.response.json'), base/'contacts'/(key+'.intent.json'),
                base/'contacts'/(key+'.response.json')]:
            raise LearningError('historical stage filename differs')
        expected_paths.update((ip, rp, bp, brp))
        expected = {'schema': 'saea-tfgrpo-learning-request-v1', 'batch_sha256': sha(batch),
            'model_id': model_id, 'learning_config_sha256': batch['learning_config_sha256'],
            'stage': stage, 'identity': identity, 'messages': messages}
        receipt = envelope.get('response')
        if (request != expected or bridge.get('request') != request
                or envelope != {'request_sha256': sha(request), 'response': receipt, 'response_sha256': sha(receipt)}
                or bridge_response != {'request_sha256': sha(request), 'response': receipt}):
            raise LearningError('historical stage prompt or bridge custody differs')
        prefix = f'{plan.run_id}/{plan.attempt_id}/'
        if not receipt['receipt_id'].startswith(prefix):
            raise LearningError('historical bridge receipt identity differs')
        rid = receipt['receipt_id'][len(prefix):]; native_key = (rid, receipt['response_sha256'])
        if native_key not in observed or native_key in used:
            raise LearningError('historical stage lacks unique native response')
        record, body, _ = observed[native_key]; payload = record['intent']['payload']
        if (receipt['model_id'] != model_id or receipt['text'] != body.get('text')
                or not isinstance(receipt['text'], str) or not receipt['text'].strip()
                or receipt['usage'] != {'calls': 1, **{k: body['usage'][k] for k in ('prompt_tokens', 'completion_tokens')}}
                or receipt['request_sha256'] != record['intent']['request_sha256']
                or payload['sample_id'] != sample_id
                or comparable_request(bridge['payload']) != comparable_request(payload)
                or (finished and body.get('finish_reason') != 'stop'
                    and not (stage == 'summary' and body.get('finish_reason') == 'length'))):
            raise LearningError('historical stage response, native request or usage differs')
        context = json.loads(payload['history'][0]['content'])
        if context.get('source_messages') != messages or context.get('stage') != stage:
            raise LearningError('historical native stage messages differ')
        used.add(native_key)
        # The historical learner accepted nonempty summary text without a stop
        # requirement. Preserve its exact existing input, disclose truncation,
        # and do not silently regenerate a different learning trajectory.
        if stage == 'summary' and body.get('finish_reason') == 'length':
            truncated_summaries.append({'trajectory_id': identity, 'response_sha256': receipt['response_sha256'],
                'finish_reason': 'length', 'text_characters': len(receipt['text'])})
        return receipt['text'], body.get('finish_reason')

    def messages(prefix, **values):
        return [{'role': 'system', 'content': render(prefix+'_SP', **common)},
                {'role': 'user', 'content': render(prefix+'_UP', **values)}]

    mixed = [g for g in groups if {r['reward'] for r in g['rollouts']} == {0, 1}]
    if not mixed: raise LearningError('final import needs completed mixed groups')
    current = '\n'.join(f'[{k}]. {v}' for k, v in sorted(parent['items'].items())) or 'None'
    operations = []
    for group in mixed:
        for roll in group['rollouts']:
            tid = roll['trajectory_id']
            if tid not in summaries:
                summaries[tid], _ = read_stage('summary', tid, group['query_id'], messages(
                    'SINGLE_ROLLOUT_SUMMARY_TEMPLATE', question=group['query'], answer=group['answer'],
                    critique=roll['critique'], trajectory=summary_trajectory(group['query'], roll['trajectory'])))
        formatted = '\n\n'.join(f"Attempt {i+1} (Reward {r['reward']}):\n{summaries[r['trajectory_id']]}"
                                for i, r in enumerate(group['rollouts']))
        advantage, _ = read_stage('advantage', group['query_id'], group['query_id'], messages(
            'SINGLE_QUERY_GROUP_ADVANTAGE', question=group['query'], answer=group['answer'], trajectories=formatted))
        proposed, _ = read_stage('group_update', group['query_id'], group['query_id'], messages(
            'GROUP_EXPERIENCE_UPDATE_TEMPLATE', existing_experiences=current,
            new_experiences=semantic_advantage(advantage)))
        operations.extend(parse_operations(proposed, parent['items'], max_operations=batch['bounds']['max_operations']))
    material = json.dumps({'existing_experiences': parent['items'], 'proposed_operations': operations},
                          ensure_ascii=False, sort_keys=True)
    final_messages = messages('BATCH_EXPERIENCE_UPDATE_TEMPLATE', experiences_and_operations=material)
    final_text, finish = read_stage('batch_update', 'batch', groups[0]['query_id'], final_messages, finished=False)
    if finish != 'length': raise LearningError('final-only correction requires an observed output-limit failure')
    try: parse_operations(final_text, parent['items'], max_operations=batch['bounds']['max_operations'])
    except LearningError: pass
    else: raise LearningError('already parseable final response must not be regenerated')
    actual_paths = {p for directory in (base/'contacts', base/'learner/contacts')
                    for pattern in ('*.intent.json', '*.response.json') for p in directory.glob(pattern)}
    if rows or expected_paths != actual_paths or used != set(observed):
        raise LearningError('unaccounted historical stages or native contacts')
    return operations, {'schema': 'saea-tfgrpo-final-import-custody-v1', 'source': spec,
        'source_run_id': plan.run_id, 'source_attempt_id': plan.attempt_id,
        'source_batch_sha256': sha(batch), 'source_learning_config_sha256': batch['learning_config_sha256'],
        'learned_query_ids': sorted(g['query_id'] for g in mixed), 'mixed_groups': len(mixed),
        'imported_group_updates': len(mixed),
        'imported_stage_contacts': len(used)-1, 'imported_summaries': len(summaries),
        'operations_sha256': sha(operations), 'final_messages_sha256': sha(final_messages),
        'original_total_usage': totals, 'native_source_usage': source_costs,
        'historical_summary_custody': summary_custody, 'source_all_responses_observed': True,
        'historical_truncated_summaries': truncated_summaries,
        'excluded_final_response': {'finish_reason': finish, 'retained_in_historical_cost': True},
        'classification': 'Verified historical inputs; excluded from fresh generation counters.'}


def build_import(*, data_root, source_plan, native_source_plans, output_relpath):
    """Pin the complete stopped batch, then run the same read-only validation."""
    from ..core.atomic_file import publish_new
    root = Path(data_root).resolve()
    plan = source_plan if isinstance(source_plan, ExecutionPlan) else ExecutionPlan(**source_plan)
    def pin(path):
        path = Path(path).resolve()
        if not path.is_relative_to(root): raise LearningError('import pin escaped data root')
        return {'relpath': str(path.relative_to(root)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    base = root/'runs'/plan.run_id/plan.attempt_id/'tfgrpo'
    batch_path = base/'learner/batch.json'; batch = json.loads(batch_path.read_bytes())
    sources = []
    for value in native_source_plans:
        native = value if isinstance(value, ExecutionPlan) else ExecutionPlan(**value)
        journal = ContactJournal(root, native, 'prepared-context-text')
        paths = sorted(set(journal.root.glob('*.intent.json')) | set(journal.root.glob('*.response.json')))
        sources.append({'plan': asdict(native), 'attempt': pin(root/'attempts'/native.run_id/(native.attempt_id+'.json')),
                        'journal_files': [pin(p) for p in paths]})
    records = []
    for path in sorted((base/'learner/contacts').glob('*.intent.json')):
        request = json.loads(path.read_bytes()); key = path.name.removesuffix('.intent.json')
        records.append({'stage': request['stage'], 'identity': request['identity'], 'intent': pin(path),
            'response': pin(path.with_name(key+'.response.json')), 'bridge_intent': pin(base/'contacts'/path.name),
            'bridge_response': pin(base/'contacts'/(key+'.response.json'))})
    resume_path = base.parent/'resume-declaration.json'; resume = recovery = None
    if resume_path.exists():
        resume = pin(resume_path)
        declaration = json.loads(resume_path.read_bytes())['native_recovery_receipt']
        recovery = pin(Path(declaration['path']))
        if recovery['sha256'] != declaration['sha256']: raise LearningError('recovery receipt changed')
    doc = {'schema': 'saea-tfgrpo-final-import-v1', 'source_plan': asdict(plan), 'source_batch': pin(batch_path),
           'native_sources': sources, 'records': records, 'source_resume': resume, 'source_recovery': recovery}
    out = (root/output_relpath).resolve()
    if not out.is_relative_to(root): raise LearningError('import output escaped data root')
    if out.exists():
        if out.read_bytes() != canonical(doc): raise LearningError('existing final import changed')
    else: publish_new(out, canonical(doc))
    spec = pin(out)
    operations, custody = load_import(spec, data_root=root, **{k: batch[k] for k in
        ('groups', 'common', 'model_id', 'parent', 'policy_sha256', 'rollout_kind')})
    return spec, operations, custody
