"""Frozen full-test workload adapter; dispatch uses the existing shared launcher.

The adapter stages, prepares and registers held Direct/A0 cells. It never
dispatches, learns from test labels, resamples the roster or resets past costs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/analysis')]
from speech_aware_evidence_acquisition.core.atomic_file import publish_new

KIND = 'final-report-method'
DATASET = 'slurp-speech-native-test-full'
ROSTER = 'docs/receipts/rosters/slurp-test-full-13078-v1.json'
DATASET_SHA = '127556a475996603862302ef68196d80ec075818cd88a407e37d42387b664e34'
ROSTER_SHA = 'b925e90ab99727e7d0ff9fc4ec6b7c91a0d86bff0a5b0a4746b9cc98237b19c3'
AUTH = 'docs/decisions/2026-10-30-owner-full-test-and-first-system-gain.md'
AUTH_SHA = 'ca1701626e9f5d5d48d39bb9fe5523f6b7c1480feb2cb16a8091957a9f83382f'
POFF_AUTH = 'docs/decisions/Agent-Full-Test-Development-Lock.md'
POFF_AUTH_SHA = 'f8be5615dd0befe90a97f091ee0a1553f6ea6f4926a1d8b99eb5706d4be8ebea'
LANE_AUTH = 'docs/decisions/Resource-Partition-Full-Test.md'
LANE_AUTH_SHA = 'ec1eaba35c633337526668a60bc928e9651ccd72519999b0b0b03b17d1ba6c34'
LANE_PROFILE = 'qwen3-omni-fulltest-1xtp4-s128-ctx48k-v1'
LANE_MODEL = 'bf16-thinking-fulltest-tp4-lane-v1'
DEVELOPMENT_LOCK = 'configs/evaluation/agent-development-lock-v1.json'
DEVELOPMENT_LOCK_SHA = '318f984210e53009f2f2aa72bab1ea8f105492da7d81ffaccc0567499c21ebd0'
SOURCES = {
    'direct': ('configs/baseline/slurp-direct-only-fold-b-baseline-raw-port-v1.json',
               '8470b6fd2ce5b9f0fe907c43efbe5a093aaf6f46a31f3c6510f807305f0adbcf'),
    'a0': ('configs/baseline/slurp-tfrl-hardset-a0-observation-r1-v1.json',
           'a3dcf828852595f68844ed6d778d5b2c6088271c8d4f83a2feb4b12b8edc15dc'),
    'a0-poff': ('configs/baseline/slurp-tfrl-hardset-a0-observation-r1-v1.json',
                'a3dcf828852595f68844ed6d778d5b2c6088271c8d4f83a2feb4b12b8edc15dc'),
}
SOURCES['a0-poff-lane'] = SOURCES['a0-poff']
MODEL_SOURCE = 'configs/model/bf16-thinking-vllm-slurp-module-fold-b-seq4-len24k-budget3072-max4096-closemarker-port-v2.json'
MODEL_SOURCE_SHA = '8d9b3a1d12601bb0f4abbbc3f2709e48afd561a7ccf2c6dbbffd93c9e153309d'
ROWS = 13078
POLICY = {'test_labels_in_runtime': False, 'test_feedback_for_learning': False,
          'complete_official_scoring_jobs': 1, 'subsets_or_best_of': False,
          'preserve_all_historical_costs': True, 'automatic_replay': False}


def read(path):
    return json.loads(Path(path).read_bytes())


def raw(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def confined(root, rel):
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or '..' in Path(rel).parts:
        raise ValueError('relative confined artifact path required')
    path = (Path(root) / rel).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('artifact escapes declared root')
    return path


def pin(root, rel):
    return {'relpath': rel, 'sha256': sha(confined(root, rel))}


def pinned_path(root, binding):
    if not isinstance(binding, dict) or set(binding) != {'relpath', 'sha256'}:
        raise ValueError('exact artifact pin required')
    path = confined(root, binding['relpath'])
    if not path.is_file() or sha(path) != binding['sha256']:
        raise ValueError('required pinned input missing or changed: ' + binding['relpath'])
    return path


def identity(method):
    if method not in SOURCES:
        raise ValueError('only frozen direct, a0 or a0-poff methods are supported')
    return f'slurp-{method}-fulltest-v1', f'SAEA-SLURP-{method.upper()}-FULLTEST-V1'


def authority(method):
    identity(method)
    if method == 'a0-poff-lane':
        return {'relpath': LANE_AUTH, 'sha256': LANE_AUTH_SHA}
    return {'relpath': POFF_AUTH, 'sha256': POFF_AUTH_SHA} if method == 'a0-poff' else {
        'relpath': AUTH, 'sha256': AUTH_SHA}


def development_policy(repo, config, context):
    """Only the separately registered method may adopt the pinned P-off choice."""
    if config['method'] not in ('a0-poff', 'a0-poff-lane'):
        if 'development_lock' in config:
            raise ValueError('historical methods cannot adopt the development lock')
        return None
    binding = {'relpath': DEVELOPMENT_LOCK, 'sha256': DEVELOPMENT_LOCK_SHA}
    if config.get('development_lock') != binding:
        raise ValueError('a0-poff requires the exact frozen development lock')
    lock = read(pinned_path(repo, binding))
    expected = {'perception': {'enable_thinking': False, 'answer_reserve_tokens': 1024},
                'solver': {'enable_thinking': True, 'reasoning_budget_tokens': 3072,
                           'answer_reserve_tokens': 1024}}
    if (lock['thinking_policy'] != expected or lock['test_used_for_tuning'] is not False
            or config['runtime']['model_fragment'] != (LANE_MODEL if config['method'] == 'a0-poff-lane' else lock['model_fragment'])
            or config['runtime']['profile'] != (LANE_PROFILE if config['method'] == 'a0-poff-lane' else lock['deployment_profile'])
            or config['execution']['admitted_level'] != lock['per_instance_concurrency']
            or lock['cards'] != 8 or lock['instances'] != 2 or context != 49152):
        raise ValueError('frozen P-off/S3072 development policy or runtime changed')
    return copy.deepcopy(expected)


def population(repo):
    from speech_aware_evidence_acquisition.data.splits import prefixed, split_identity_hash
    view = read(pinned_path(repo, {'relpath': f'configs/dataset/{DATASET}.json', 'sha256': DATASET_SHA}))
    roster = read(pinned_path(repo, {'relpath': ROSTER, 'sha256': ROSTER_SHA}))
    ids = [r['sample_id'] for r in roster['items']]
    if (len(ids) != ROWS or len(set(ids)) != ROWS or view['count'] != ROWS
            or ids != [r['sample_id'] for r in view['items']]
            or view['split_role'] != 'final-report' or roster['split_role'] != 'final-report'
            or split_identity_hash(prefixed('slurp', ids)) != view['split_identity_hash']
            or roster['split_identity_hash'] != view['split_identity_hash']):
        raise ValueError('exact complete frozen TEST13078 membership required')
    for row in view['items']:
        if (not row['sample_id'].startswith('slurp:test:')
                or not row['media_relpath'].startswith('datasets/slurp/slurp_real/')
                or not re.fullmatch('[0-9a-f]{64}', row['audio_sha256'])
                or type(row['audio_seconds']) not in (int, float)
                or not math.isfinite(row['audio_seconds']) or row['audio_seconds'] <= 0):
            raise ValueError('real complete test media required')
    return roster, view


def runtime_binding(repo, binding, *, single_lane=False):
    if set(binding) != {'model_fragment', 'profile', 'service_session', 'runtime_receipt'}:
        raise ValueError('explicit runtime/model/profile/session binding required')
    for key in ('model_fragment', 'profile', 'service_session'):
        if not isinstance(binding[key], str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', binding[key]):
            raise ValueError('invalid runtime identity')
    model = read(confined(repo, f"configs/model/{binding['model_fragment']}.json"))
    original = read(pinned_path(repo, {'relpath': MODEL_SOURCE, 'sha256': MODEL_SOURCE_SHA}))
    keys = ('model_lock_key', 'transport', 'served_model_name', 'decoding_params',
            'reasoning_budget_tokens', 'answer_reserve_tokens', 'enable_thinking',
            'reasoning_context_policy', 'forced_close_message_sha256',
            'prompt_order', 'reasoning_content_guard')
    if any(model.get(k) != original.get(k) for k in keys):
        raise ValueError('frozen task model/4096 completion/3072 reasoning changed')
    if model['runtime_receipt'] != binding['runtime_receipt']:
        raise ValueError('runtime receipt differs from selected model')
    confined(repo / 'docs', binding['runtime_receipt'])
    def context_and_args(doc):
        args = list(doc['server_args'])
        if args.count('--max-model-len') != 1:
            raise ValueError('explicit context required')
        index = args.index('--max-model-len')
        value = int(args[index + 1])
        if value not in (24576, 49152):
            raise ValueError('unregistered task context')
        del args[index:index + 2]
        return value, args
    if context_and_args(model)[1] != context_and_args(original)[1]:
        raise ValueError('frozen server prompt/decoding arguments changed')
    profile = read(confined(repo, f"configs/deployment/{binding['profile']}.json"))
    if (profile['tensor_parallel_size'] != 4 or profile['max_num_seqs'] != 128
            or profile['max_num_batched_tokens'] != 2048 or len(profile['instances']) != (1 if single_lane else 2)
            or sorted(c for i in profile['instances'] for c in i['cards']) != list(range(4 if single_lane else 8))
            or (single_lane and (binding['profile'] != LANE_PROFILE or profile['instances'][0]['port'] != 8200))):
        raise ValueError('fixed eight-card two-TP4 runtime required')
    deployed = read(confined(repo, f"configs/model/{profile['model_fragment']}.json"))
    if any(deployed.get(k) != model.get(k) for k in keys) or deployed['server_args'] != model['server_args']:
        raise ValueError('deployment differs from task model')
    return model, context_and_args(model)[0]


def allowance(repo, config, roster):
    from speech_aware_evidence_acquisition.contracts import require_owner_split_allowance
    _, run = identity(config['method'])
    auth = authority(config['method'])
    pinned_path(repo, auth)
    binding = config['allowance']
    if not isinstance(binding, dict) or set(binding) != {'path', 'sha256'}:
        raise ValueError('explicit final-report owner allowance required')
    docs = []
    for rid in (run, run + '-FIT', run + '-READ'):
        docs.append(require_owner_split_allowance(run_id=rid,
            execution_profile='bounded-discovery-probe', carrier_lock_key='slurp',
            split_identity_hash=roster['split_identity_hash'], owner_allowance=binding, source_root=repo))
    doc = docs[0]
    split = doc['split']
    if (doc.get('authorization_basis') != auth
            or split.get('roster_receipt_relpath') != ROSTER
            or split.get('roster_receipt_sha256') != ROSTER_SHA
            or split.get('roster_id') != roster['roster_id']
            or doc.get('read_run_ids') != [run, run + '-READ']
            or doc.get('preparation_run_ids') != [run + '-FIT']
            or set(doc['permitted_run_ids']) != {run, run + '-FIT', run + '-READ'}
            or doc.get('evaluation_policy') != POLICY):
        raise ValueError('full-test allowance must bind frozen method, owner authority and no test learning')
    return doc


def create_allowance(repo, data, method):
    """Record already-granted full-test authority, never change old receipts."""
    environment(repo)
    require_unspent(repo, data, method)
    roster, _ = population(repo)
    auth = authority(method)
    pinned_path(repo, auth)
    name, run = identity(method)
    rel = f'docs/receipts/owner/{name}-allowance-v1.json'
    doc = {'schema': 'saea-owner-split-allowance-v1', 'study': 'speech-aware-evidence-acquisition',
        'status': 'authorised', 'authorization_basis': auth,
        'dated_owner_decision': 'Owner 2026-10-30 D1/D2/D3; 2026-11-02 independent frozen full-test baseline priority. This is mechanical registration of granted authority, not a historical signature.',
        'venue': 'Owner authorizes complete-test reporting in the current NPU reproduction study; no paper publication is requested.',
        'split': {'carrier_lock_key': 'slurp', 'split_identity_hash': roster['split_identity_hash'],
                  'roster_id': roster['roster_id'], 'roster_receipt_relpath': ROSTER,
                  'roster_receipt_sha256': ROSTER_SHA, 'recordings': ROWS},
        'permitted_run_ids': [run, run + '-FIT', run + '-READ'],
        'read_run_ids': [run, run + '-READ'], 'preparation_run_ids': [run + '-FIT'],
        'evaluation_policy': POLICY, 'prior_exposure': 'Retain all historical test-sample evaluation and costs. No claim of zero prior exposure or test-derived tuning.'}
    publish_new(repo / rel, raw(doc))
    return {'allowance': {'path': rel, 'sha256': sha(repo / rel)},
            'new_model_calls': 0, 'generation_authorized': False, 'readiness': 'held'}


def execution_binding(repo, config):
    """Use the shared client-width override; keep hashed serving geometry intact."""
    execution = config.get('execution')
    if (not isinstance(execution, dict) or set(execution) != {'admitted_level'}
            or type(execution['admitted_level']) is not int or execution['admitted_level'] != 96):
        raise ValueError('explicit stable C96 admitted_level required')
    profile = read(confined(repo, f"configs/deployment/{config['runtime']['profile']}.json"))
    admission = profile.get('admitted_level', profile['max_num_seqs'])
    if type(admission) is not int or admission < 96 or admission > profile['max_num_seqs']:
        raise ValueError('C96 cannot raise the existing deployment admission')
    from speech_aware_evidence_acquisition.core.flight.width import flight_width
    width = flight_width({**profile, 'admitted_level': 96}, 4)
    instances = 1 if config.get('method') == 'a0-poff-lane' else 2
    if width != 96 * instances:
        raise ValueError('C96 requires the declared instance count')
    return {'admitted_level': 96, 'instances': instances, 'total_client_concurrency': width,
        'environment': {'SAEA_ADMITTED_LEVEL': '96'}}


def check_config(repo, config, roster):
    required = {'schema', 'method', 'source_baseline', 'runtime', 'execution', 'budget', 'allowance'}
    if (not isinstance(config, dict) or not required <= set(config)
            or set(config) - required - {'a0_inputs', 'development_lock'}
            or config['schema'] != 'saea-final-report-job-config-v1'):
        raise ValueError('explicit final-report job configuration required')
    identity(config['method'])
    source, digest = SOURCES[config['method']]
    if config['source_baseline'] != {'relpath': source, 'sha256': digest}:
        raise ValueError('frozen source baseline required')
    base = read(pinned_path(repo, config['source_baseline']))
    budget = config['budget']
    contacts = 1 if config['method'] == 'direct' else 2
    if (set(budget) != {'generation_calls', 'fit_calls', 'gpu_hours'}
            or type(budget['generation_calls']) is not int or budget['generation_calls'] != 2 * contacts * ROWS
            or type(budget['fit_calls']) is not int or budget['fit_calls'] != contacts * ROWS
            or type(budget['gpu_hours']) not in (int, float)
            or not math.isfinite(budget['gpu_hours']) or budget['gpu_hours'] <= 0):
        raise ValueError('exact fallback call budget and explicit positive card-hour cap required')
    allowance(repo, config, roster)
    model, context = runtime_binding(repo, config['runtime'], single_lane=config['method'] == 'a0-poff-lane')
    execution_binding(repo, config)
    development_policy(repo, config, context)
    if config['method'] == 'direct' and 'a0_inputs' in config:
        raise ValueError('Direct must not consume A0 retrieval inputs')
    return base, model, context, contacts


def a0_inputs(repo, data, config, view, source):
    """Reuse typed loaders and shared builder custody, never fabricate vectors."""
    from speech_aware_evidence_acquisition.rails.phase_supply import (
        load_sealed_segment_vectors, load_sealed_vectors, load_sealed_agreement)
    inputs = config.get('a0_inputs')
    if not isinstance(inputs, dict) or set(inputs) != {
            'segment_query', 'content_query', 'agreement', 'content_provenance',
            'agreement_receipt', 'source_receipts', 'query_ledger_row'}:
        raise ValueError('A0 held: real full-test query/content/agreement provenance missing')
    segment = load_sealed_segment_vectors(pinned_path(data, inputs['segment_query']),
        expected_sha256=inputs['segment_query']['sha256'], dimensions=['intent-boundary'])
    content = load_sealed_vectors(pinned_path(data, inputs['content_query']),
        expected_sha256=inputs['content_query']['sha256'], dimensions=['content'])
    agree = load_sealed_agreement(pinned_path(data, inputs['agreement']),
        expected_sha256=inputs['agreement']['sha256'], dimension='agreement:exemplar-intent')
    ids = {r['sample_id'] for r in view['items']}
    if set(segment) != ids or set(content) != ids or set(agree) != ids:
        raise ValueError('A0 inputs must cover all and only real TEST13078 rows')
    for row in view['items']:
        query = segment[row['sample_id']]
        if (query.audio_sha256 != row['audio_sha256'] or len(query.segments) != 1
                or query.segments[0]['intent-boundary'] != content[row['sample_id']]['content']):
            raise ValueError('speech audio binding or value-preserving content derivative differs')
    provenance = read(pinned_path(data, inputs['content_provenance']))
    if (provenance.get('schema') != 'saea-slurp-exemplar-query-vectors-provenance-v1'
            or provenance.get('parent_relpath') != inputs['segment_query']['relpath']
            or provenance.get('parent_sha256') != inputs['segment_query']['sha256']
            or provenance.get('sha256') != inputs['content_query']['sha256']
            or provenance.get('row_count') != ROWS):
        raise ValueError('shared content-derivative provenance differs')
    receipt = read(pinned_path(data, inputs['agreement_receipt']))
    origin = receipt['derived_from']
    contacts = receipt.get('contacts', {})
    if (receipt.get('schema') != 'saea-phase-agreement-receipt-v1'
            or receipt.get('contact_class') != 'model-free'
            or set(contacts) != {'model', 'server', 'encoder', 'embedder', 'scorer', 'network', 'gpu', 'paid_api'}
            or any(type(v) is not int or v != 0 for v in contacts.values())
            or receipt['artifact']['relpath'] != inputs['agreement']['relpath']
            or receipt['artifact']['sha256'] != inputs['agreement']['sha256']
            or receipt['artifact']['rows'] != ROWS
            or origin['split_identity_hash'] != view['split_identity_hash']
            or origin['vote_store'] != 'train-exemplars' or origin['vote_phase'] != 'solver'
            or origin['target_store'] != 'intent-boundaries'
            or origin['vote_snapshot_sha256'] != source['knowledge_stores']['train-exemplars']['snapshot']['sha256']
            or origin['target_snapshot_sha256'] != source['knowledge_stores']['intent-boundaries']['snapshot']['sha256']
            or origin['vote_spec']['query_source']['sha256'] != inputs['content_query']['sha256']):
        raise ValueError('shared agreement builder custody differs')
    if not isinstance(inputs['source_receipts'], list) or not inputs['source_receipts']:
        raise ValueError('real speech encoder/source custody receipts required')
    for binding in inputs['source_receipts']:
        pinned_path(data, binding)
    ledger = (repo / 'docs/exposure-ledger.md').read_text()
    for row in (inputs['query_ledger_row'], receipt.get('artifact_id')):
        if (not isinstance(row, str) or not re.fullmatch(r'SAEA-[A-Z0-9-]+', row)
                or ledger.count('| ' + row + ' |') != 1):
            raise ValueError('A0 source must name a unique real exposure-ledger row')
    return inputs


def derive(repo, data, config):
    roster, view = population(repo)
    source, model, context, contacts = check_config(repo, config, roster)
    base = copy.deepcopy(source)
    policy = development_policy(repo, config, context)
    if policy is not None:
        base['thinking_policy'] = policy
    name, run = identity(config['method'])
    base.pop('prepared_phase_supply', None)
    base['serving_slot_tokens'] = context
    base['scorers'] = ['slurp-slu-official-test', 'cost']
    fit_key = 'live_phase_fit' if config['method'] == 'direct' else 'native_contact_fit'
    base[fit_key].update(run_id=run + '-FIT', attempt_id='fit-v1', contact_cap=contacts * ROWS)
    audio = math.fsum(r['audio_seconds'] for r in view['items'])
    if config['method'] == 'direct':
        if source['policy'] != 'phase-direct-only-v1' or source.get('knowledge_stores') or source.get('native_episode'):
            raise ValueError('frozen one-contact no-retrieval Direct required')
        base['live_phase_fit_audio_seconds_cap'] = math.ceil(audio)
    else:
        from speech_aware_evidence_acquisition.rails import consumption_module as cm
        from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks
        inputs = a0_inputs(repo, data, config, view, source)
        block = base['native_episode']['consumption_module']
        if source['policy'] != 'phase-native-module-v1-c2' or block['columns'] or block['review_loop'] != 'off':
            raise ValueError('frozen experience-free two-contact A0 required')
        boundary = base['knowledge_stores']['intent-boundaries']['phase_supply']['perception']
        exemplar = base['knowledge_stores']['train-exemplars']['phase_supply']['solver']
        for target, key in ((boundary['query_source'], 'segment_query'),
                            (exemplar['query_source'], 'content_query'),
                            (boundary['agreement_source'], 'agreement')):
            target.update(inputs[key])
            target['tool_contact_ledger_row'] = (read(pinned_path(data,
                inputs['agreement_receipt']))['artifact_id'] if key == 'agreement'
                else inputs['query_ledger_row'])
            target['embedded_signal'] = ('derived: ' if key == 'agreement' else 'speech: ') + 'Full original TEST speech-derived input; frozen TRAIN knowledge only, no test label or transcript.'
        components = [c for c in block['components'] if c['component_id'] == 'agree-vote-seed']
        if len(components) != 1 or components[0]['state'] != 'on':
            raise ValueError('frozen A0 agreement component required')
        components[0]['params']['agreement_source'] = copy.deepcopy(boundary['agreement_source'])
        # Keep knowledge provenance; replace only the old roster-specific files.
        base['preparation_source_receipts'] = [r for r in base['preparation_source_receipts']
            if not any(marker in r['relpath'] for marker in ('R01C', 'r01c'))]
        base['preparation_source_receipts'] += [{'root': 'data', **r} for r in
            [*inputs['source_receipts'], inputs['content_provenance'], inputs['agreement_receipt']]]
        base['preparation_source_receipts'].append({'root': 'repository', 'relpath': ROSTER, 'sha256': ROSTER_SHA})
        cm.verify_declared_module(block, base, repo_root=repo)
        if cm.compile_module(block, carrier_module_hooks(repo_root=repo), repo_root=repo).program_ceiling != 2:
            raise ValueError('two-contact A0 program required')
    base['what_this_is'] = 'Frozen ' + config['method'] + ' complete TEST13078 report; no learning or test selection.'
    plan = {'execution_profile': 'bounded-discovery-probe', 'split_role': 'final-report',
            'planned_model_calls': config['budget']['generation_calls'],
            'planned_speech_audio_seconds': math.ceil(2 * contacts * audio),
            'planned_gpu_hours': config['budget']['gpu_hours']}
    exp = {'experiment_id': run, 'execution_profile': plan['execution_profile'],
        'execution_plan': plan, 'service_session': config['runtime']['service_session'],
        'owner_allowance': config['allowance'], 'split_id': 'TEST-FULL',
        'preserve_wire_responses': True, 'purpose': base['what_this_is'],
        'population': {'count': ROWS, 'dataset_fragment': DATASET,
                      'roster_receipt': ROSTER, 'split_identity_hash': roster['split_identity_hash']},
        'populations': [{'dataset_fragment': DATASET, 'ids': [r['sample_id'] for r in roster['items']]}],
        'cells': [{'cell_id': name, 'baseline_fragment': name, 'dataset_fragment': DATASET,
                   'model_fragment': config['runtime']['model_fragment'], 'run_id': run,
                   'attempt_id': 'generation-v1', 'allowed_execution_stages': ['check', 'prepare-supply', 'worker', 'campaign'],
                   'readiness': {'state': 'held', 'requirements': [], 'reason': 'Prepared full test, registered allowance, shared admission and preflight required.'}}],
        'stage3_algorithm': {'id': 'none', 'operators_used': [], 'note': 'Independent frozen final reporting; no learning.'},
        'stage3_measurement': {'rows': ROWS, 'claims_benefit': False, 'expected_effect_pp': 0,
            'measurement_design': 'material-source', 'primary_metric_key': 'slu',
            'read_roster': roster['roster_id'], 'tuning_roster': 'none', 'note': 'All13078, official test scorer, no test tuning.'}}
    return base, exp, roster, view


def scientific_hash(base):
    base = copy.deepcopy(base)
    base.pop('prepared_phase_supply', None)
    return hashlib.sha256(raw(base)).hexdigest()


def require_unspent(repo, data, method):
    name, run = identity(method)
    ledger = (repo / 'docs/exposure-ledger.md').read_text()
    if (repo / 'docs/readiness' / name).exists():
        raise ValueError('existing registration; inspect custody before continuation')
    for rid in (run, run + '-FIT', run + '-READ'):
        if '| ' + rid + ' |' in ledger or any((data / k / rid).exists() for k in ('runs', 'attempts')):
            raise ValueError('existing registration/spend; never blindly repeat')


def environment(repo):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    assert_execution_environment()
    return require_clean_study_repo(repo)


def owned_out(data, out):
    if out.exists() or not out.resolve().is_relative_to((data / 'work/codex').resolve()):
        raise ValueError('fresh owned output required')


def stage(repo, data, config, out):
    head = environment(repo)
    require_unspent(repo, data, config['method'])
    owned_out(data, out)
    base, exp, roster, view = derive(repo, data, config)
    name, run = identity(config['method'])
    docs = {f'configs/baseline/{name}.json': base, f'configs/experiment/{name}.json': exp}
    if any((repo / rel).exists() for rel in docs):
        raise ValueError('staged configuration already exists; no overwrite')
    result = {'schema': 'saea-final-report-stage-v1', 'config': config, 'source_commit': head,
        'rows': ROWS, 'run_id': run, 'dataset': pin(repo, f'configs/dataset/{DATASET}.json'),
        'roster': pin(repo, ROSTER), 'source_baseline': config['source_baseline'],
        'model': pin(repo, f"configs/model/{config['runtime']['model_fragment']}.json"),
        'profile_pin': pin(repo, f"configs/deployment/{config['runtime']['profile']}.json"),
        'source_model': pin(repo, MODEL_SOURCE), 'output_hashes': {k: hashlib.sha256(raw(v)).hexdigest() for k, v in docs.items()},
        'new_model_calls': 0, 'test_labels_read': False, 'generation_authorized': False, 'readiness': 'held'}
    out.mkdir(parents=True)
    publish_new(out / 'intent.json', raw(result))
    for rel, doc in docs.items():
        publish_new(repo / rel, raw(doc))
    publish_new(out / 'complete.json', raw(result))
    return result


def staged_inputs(repo, data, stage_record):
    doc = read(stage_record)
    if (doc.get('schema') != 'saea-final-report-stage-v1' or doc.get('rows') != ROWS
            or doc.get('new_model_calls') != 0 or doc.get('generation_authorized') is not False):
        raise ValueError('complete held model-free stage required')
    for key in ('dataset', 'roster', 'source_baseline', 'model', 'profile_pin', 'source_model'):
        pinned_path(repo, doc[key])
    base, exp, _, _ = derive(repo, data, doc['config'])
    name, run = identity(doc['config']['method'])
    expected = {f'configs/baseline/{name}.json': base, f'configs/experiment/{name}.json': exp}
    if doc['output_hashes'] != {k: hashlib.sha256(raw(v)).hexdigest() for k, v in expected.items()}:
        raise ValueError('stage must preserve frozen method derivation')
    for rel, digest in doc['output_hashes'].items():
        if sha(confined(repo, rel)) != digest:
            raise ValueError('staged configuration changed')
    return doc


def prepare(repo, data, stage_record, out):
    environment(repo)
    doc = staged_inputs(repo, data, stage_record)
    config = doc['config']
    require_unspent(repo, data, config['method'])
    owned_out(data, out)
    name, run = identity(config['method'])
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    from speech_aware_evidence_acquisition.core.preparation import PreparationError
    runtime = build_flight_runtime(repo, data, model=config['runtime']['model_fragment'],
        dataset=DATASET, baseline=name, experiment=name, run_id=run, attempt_id='generation-v1',
        runtime_root=data / 'work/scratch/runtime-root', lock_path=repo.parents[1] / 'docs/datasets.lock.json')
    target = data / 'prepared' / run / 'phase-supply-v1.json'
    supply = None
    if config['method'] == 'direct':
        try:
            runtime.phase_supply_artifact_identity()
        except PreparationError as error:
            if str(error) != 'program has no supplied first phase to prepare':
                raise
        else:
            raise ValueError('Direct unexpectedly declares a knowledge supply')
        runtime.verify_prepared_artifacts()
    else:
        if target.exists():
            raise ValueError('existing preparation must be inspected; no implicit retry')
        runtime.prepare_phase_supply(target, max_workers=16)
        from tfrl_odd40_cells import verify_output
        verify_output(read(target)['output'], [r['sample_id'] for r in population(repo)[0]['items']])
        supply = pin(data, str(target.relative_to(data)))
    out.mkdir(parents=True)
    result = {'schema': 'saea-final-report-prepared-v1', 'stage_sha256': sha(stage_record),
        'rows': ROWS, 'run_id': run, 'method': config['method'], 'supply': supply,
        'preparation_mode': 'no-supplied-first-phase' if supply is None else 'shared-flight-phase-supply',
        'new_model_calls': 0, 'readiness': 'held'}
    publish_new(out / 'complete.json', raw(result))
    return result


def validate(repo, data, name, spec, exp, view):
    config = spec['job_config']
    expected_base, expected_exp, roster, expected_view = derive(repo, data, config)
    expected_name, run = identity(config['method'])
    runtime = config['runtime']
    if (name != expected_name or spec['cell'] != name or spec['experiment'] != name
            or spec['workload_kind'] != KIND or spec['run_id'] != run or spec['attempt_id'] != 'generation-v1'
            or spec['fit_run_id'] != run + '-FIT' or spec['fit_attempt_id'] != 'fit-v1'
            or spec['rows'] != ROWS or view != expected_view
            or spec['split_identity_hash'] != roster['split_identity_hash']
            or spec['authority'] != authority(config['method'])['relpath'] or spec['evaluation_policy'] != POLICY
            or spec.get('execution') != execution_binding(repo, config)
            or any(spec[k] != runtime[k] for k in ('model_fragment', 'service_session', 'profile', 'runtime_receipt'))):
        raise ValueError('complete frozen final-report identity changed')
    for key in ('dataset', 'roster', 'source_baseline', 'model', 'profile_pin', 'source_model'):
        pinned_path(repo, spec[key])
    if (spec['dataset'] != {'relpath': f'configs/dataset/{DATASET}.json', 'sha256': DATASET_SHA}
            or spec['roster'] != {'relpath': ROSTER, 'sha256': ROSTER_SHA}
            or spec['source_baseline'] != config['source_baseline']
            or spec['model']['relpath'] != f"configs/model/{runtime['model_fragment']}.json"
            or spec['profile_pin']['relpath'] != f"configs/deployment/{runtime['profile']}.json"
            or spec['source_model'] != {'relpath': MODEL_SOURCE, 'sha256': MODEL_SOURCE_SHA}):
        raise ValueError('registered method source pins differ')
    current = copy.deepcopy(exp)
    current['cells'][0]['readiness'] = expected_exp['cells'][0]['readiness']
    if current != expected_exp:
        raise ValueError('final-report population, budget, scorer contract or algorithm changed')
    base = read(repo / f'configs/baseline/{name}.json')
    if scientific_hash(base) != scientific_hash(expected_base) or scientific_hash(base) != spec['scientific_baseline_sha256']:
        raise ValueError('frozen method, retrieval or official test scorer changed')
    prep = read(pinned_path(data, spec['preparation']))
    if (prep.get('schema') != 'saea-final-report-prepared-v1' or prep['rows'] != ROWS
            or prep['run_id'] != run or prep['method'] != config['method']
            or prep['stage_sha256'] != spec['stage_sha256'] or prep['new_model_calls'] != 0):
        raise ValueError('verified own complete preparation required')
    if config['method'] != 'direct':
        from tfrl_odd40_cells import verify_output
        current_supply = read(pinned_path(data, base['prepared_phase_supply']))
        original_supply = read(pinned_path(data, prep['supply']))
        if current_supply['output'] != original_supply['output']:
            raise ValueError('A0 prepared output changed instead of identity-only repin')
        verify_output(current_supply['output'], [r['sample_id'] for r in roster['items']])
    elif 'prepared_phase_supply' in base or prep['supply'] is not None or prep['preparation_mode'] != 'no-supplied-first-phase':
        raise ValueError('Direct cannot acquire knowledge supply')
    return ROWS


def register(repo, data, stage_record, prepared):
    from datetime import datetime, timezone
    from speech_aware_evidence_acquisition.contracts import ExecutionPlan
    from speech_aware_evidence_acquisition.core.config import compose, protocol_hash
    head = environment(repo)
    staged = staged_inputs(repo, data, stage_record)
    config = staged['config']
    require_unspent(repo, data, config['method'])
    name, run = identity(config['method'])
    prep = read(prepared)
    if (prep.get('schema') != 'saea-final-report-prepared-v1'
            or prep.get('stage_sha256') != sha(stage_record) or prep.get('rows') != ROWS
            or prep.get('run_id') != run or prep.get('method') != config['method']
            or prep.get('new_model_calls') != 0):
        raise ValueError('matching completed preparation required before registration')
    if not prepared.resolve().is_relative_to((data / 'work/codex').resolve()):
        raise ValueError('preparation evidence must remain in owned data root')
    base, exp, roster, view = derive(repo, data, config)
    if config['method'] != 'direct':
        from tfrl_odd40_cells import verify_output
        verify_output(read(pinned_path(data, prep['supply']))['output'], [r['sample_id'] for r in roster['items']])
        base['prepared_phase_supply'] = prep['supply']
        exp['cells'][0]['readiness']['requirements'] = [{'kind': 'preparation', **prep['supply']}]
    elif prep['supply'] is not None or prep['preparation_mode'] != 'no-supplied-first-phase':
        raise ValueError('Direct has no retrieval preparation')
    bp, ep = [repo / f'configs/{kind}/{name}.json' for kind in ('baseline', 'experiment')]
    # Only staged, unregistered, held files reach this mutation. Later admission
    # refreshes through the existing output-preserving repin implementation.
    bp.write_bytes(raw(base))
    ep.write_bytes(raw(exp))
    cfg = compose(repo, config['runtime']['model_fragment'], DATASET, name, name)
    split = roster['split_identity_hash']
    ph = protocol_hash(cfg.config_hash, cfg.values['arm'], split)
    ident = {'cell': name, 'experiment': name, 'run_id': run, 'attempt_id': 'generation-v1',
             'fit_run_id': run + '-FIT', 'fit_attempt_id': 'fit-v1', 'rows': ROWS,
             'composed_config_hash': cfg.config_hash, 'protocol_hash': ph, 'split_identity_hash': split}
    spec = {**ident, **config['runtime'], 'workload_kind': KIND, 'job_config': config,
        'execution': execution_binding(repo, config),
        'baseline': name, 'authority': authority(config['method'])['relpath'], 'decision_id': name + '-owner-admission',
        'decision_relpath': f'coordination/{name}/root-admission-v1.json',
        'evaluation_policy': POLICY, 'scientific_baseline_sha256': scientific_hash(base),
        'preparation': pin(data, str(prepared.relative_to(data))), 'stage_sha256': sha(stage_record),
        **{k: staged[k] for k in ('dataset', 'roster', 'source_baseline', 'model', 'profile_pin', 'source_model')}}
    validate(repo, data, name, spec, exp, view)
    pkg = repo / 'docs/readiness' / name
    protocol = {'schema': 'saea-final-report-protocol-v1', 'cells': {'arm': ident},
        'method': config['method'], 'rows': ROWS, 'normal_generation_calls': config['budget']['fit_calls'],
        'fallback_inclusive_cap': config['budget']['generation_calls'], 'fit_calls': config['budget']['fit_calls'],
        'profile': config['runtime']['profile'], 'owner_allowance': config['allowance'],
        'execution': execution_binding(repo, config),
        'evaluation_policy': POLICY, 'claimed_runtime_seconds': None, 'claimed_measured_cost': None}
    pp = {'relpath': str((pkg / 'protocol-registered-v1.json').relative_to(repo)),
          'sha256': hashlib.sha256(raw(protocol)).hexdigest()}
    docs = {'candidate.json': spec, 'protocol-registered-v1.json': protocol}
    ledger_rows = []
    date = datetime.now(timezone.utc).date().isoformat()
    for kind, rid, aid, cap in [('generation', run, 'generation-v1', config['budget']['generation_calls']),
                              ('fit', run + '-FIT', 'fit-v1', config['budget']['fit_calls'])]:
        budget = {**exp['execution_plan'], 'planned_model_calls': cap}
        if kind == 'fit' and config['method'] == 'direct':
            budget['planned_speech_audio_seconds'] = base['live_phase_fit_audio_seconds_cap']
        ExecutionPlan(run_id=rid, attempt_id=aid, carrier_lock_key='slurp',
            split_identity_hash=split, protocol_hash=ph, owner_allowance=config['allowance'],
            **budget).validate(source_root=repo)
        doc = {'schema': f'saea-registered-{kind}-plan-v1', 'plan_kind': kind, 'cell': name,
            'experiment': name, 'composed_config_hash': cfg.config_hash,
            'expected_session_id': config['runtime']['service_session'], 'owner_allowance': config['allowance'],
            'execution_plan': {**budget, 'schema': 'saea-execution-plan-v1',
                'study': 'speech-aware-evidence-acquisition', 'run_id': rid, 'attempt_id': aid,
                'carrier_lock_key': 'slurp', 'split_identity_hash': split, 'protocol_hash': ph},
            'protocol_document': pp, 'status': 'Registered; held pending shared admission, preflight and lease.'}
        if prep['supply'] is not None:
            doc['prepared_phase_supply'] = prep['supply']
        rel = f'{kind}-plan-registered-v1.json'
        docs[rel] = doc
        ledger_rows.append([date, rid, budget['execution_profile'], 'Frozen full TEST13078 ' + kind,
            config['runtime']['model_fragment'], 'slurp TEST complete13078', 'final-report', split, 'no',
            str(cap), str(budget['planned_speech_audio_seconds']), str(budget['planned_gpu_hours']),
            'All prior evaluated samples and costs retained; no learning from test', str((pkg / rel).relative_to(repo)),
            'Plan SHA256 ' + hashlib.sha256(raw(doc)).hexdigest() + '; root countersigned under recorded owner amendment, coordinator Codex; ' + spec['authority']])
    recipe = {'schema': 'saea-final-report-analysis-v1', 'read_run_id': run + '-READ',
        'generation_run_id': run, 'attempt_id': 'generation-v1', 'rows': ROWS, 'roster': spec['roster'],
        'scorer': 'slurp-slu-official-test', 'complete_manifest_only': True,
        'new_model_calls': 0, 'evaluation_policy': POLICY,
        'scope': 'Shared campaign scores once after seal; report complete13078, absolute quality and all generation/FIT/history costs. No test-derived updates.'}
    docs['analysis-recipe-v1.json'] = recipe
    ledger_rows.append([date, run + '-READ', 'model-free-check', 'Complete frozen official report',
        'no model', 'slurp TEST complete13078', 'final-report', split, 'no', '0', '0', '0',
        'All prior costs retained', str((pkg / 'analysis-recipe-v1.json').relative_to(repo)),
        'Read sealed complete official score; never score/select a partial subset; ' + spec['authority']])
    pkg.mkdir(parents=True)
    for rel, doc in docs.items():
        publish_new(pkg / rel, raw(doc))
    ledger = repo / 'docs/exposure-ledger.md'
    with ledger.open('a') as stream:
        stream.write('\n## Frozen full-test ' + config['method'] + '\n\n| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n| ' + ' | '.join(['---'] * 15) + ' |\n' + '\n'.join('| ' + ' | '.join(row) + ' |' for row in ledger_rows) + '\n')
        stream.flush()
        __import__('os').fsync(stream.fileno())
    decoding = ('P-off/1024 answer; S3072/1024 answer' if config['method'] in ('a0-poff', 'a0-poff-lane')
                else 'ordinary4096/3072 task decoding')
    publish_new(repo / 'docs/readiness' / (name + '-preregistration.md'),
        ('# Frozen full-test workload\n\nComplete13078 original TEST rows, official test scorer, no test tuning. Frozen source method and ' + decoding + '. Shared FlightRuntime only. Held until committed registration, shared admission/repin, clean preflight and original runtime lease. Preserve unresolved contacts, known responses and historical costs; no blind replay.\n').encode())
    return {'candidate': name, 'rows': ROWS, 'readiness': 'held', 'new_model_calls': 0, 'source_commit': head}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['allowance', 'stage', 'prepare', 'register', 'validate'])
    for key in ('repo', 'data'):
        p.add_argument('--' + key, required=True, type=Path)
    for key in ('config', 'stage-record', 'prepared', 'out'):
        p.add_argument('--' + key, type=Path)
    p.add_argument('--candidate')
    p.add_argument('--method', choices=list(SOURCES))
    a = p.parse_args()
    if a.mode == 'allowance':
        if not a.method: p.error('allowance requires --method')
        result = create_allowance(a.repo, a.data, a.method)
    elif a.mode == 'stage':
        if a.config is None or a.out is None: p.error('stage requires --config and --out')
        result = stage(a.repo, a.data, read(a.config), a.out)
    elif a.mode == 'prepare':
        if a.stage_record is None or a.out is None: p.error('prepare requires --stage-record and --out')
        result = prepare(a.repo, a.data, a.stage_record, a.out)
    elif a.mode == 'register':
        if a.stage_record is None or a.prepared is None: p.error('register requires --stage-record and --prepared')
        result = register(a.repo, a.data, a.stage_record, a.prepared)
    else:
        if not a.candidate: p.error('validate requires --candidate')
        from registered_workload import validate_workload
        result = {'rows': validate_workload(a.repo, a.data, a.candidate,
            read(a.repo / 'docs/readiness' / a.candidate / 'candidate.json'))}
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
