"""Configured external proposals: prepare offline, run once, collect verified output."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from speech_aware_evidence_acquisition.rails.external_teacher import ExternalTeacher
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import canonical, sha
from speech_aware_evidence_acquisition.rails.solver_prompt_candidate import validate_mutation


def read(path):
    return json.loads(Path(path).read_bytes())


def pin(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def checked(value, base=Path('.')):
    if set(value) != {'path', 'sha256'}:
        raise ValueError('exact file pin required')
    path = (base / value['path']).resolve()
    if pin(path)['sha256'] != value['sha256']:
        raise ValueError('pinned bytes changed')
    return path


def save(path, value):
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location('exploration_persist', ROOT / 'scripts/environment/configured_external_learner.py')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    module.persist(path, value)


def expand_evidence(value, dictionary, stack=()):
    if isinstance(value, dict):
        if set(value) == {'development_evidence_ref'}:
            key = value['development_evidence_ref']
            if key in stack or key not in dictionary:
                raise ValueError('missing or cyclic evidence reference')
            return expand_evidence(dictionary[key], dictionary, (*stack, key))
        if set(value) == {'development_canonical_json_string'}:
            return canonical(expand_evidence(value['development_canonical_json_string'], dictionary, stack)).decode()
        return {k: expand_evidence(v, dictionary, stack) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_evidence(v, dictionary, stack) for v in value]
    return value


def request_context(context, algorithm):
    """Keep exclusion identities local; the teacher needs the editable boundary."""
    if algorithm != 'gepa':
        return context
    result = json.loads(json.dumps(context))
    excluded = result['contract'].pop('excluded_query_ids')
    result['contract']['excluded_query_ids_commitment'] = {
        'count': len(excluded), 'sha256': sha(excluded),
        'enforcement': 'full exclusion list retained and checked locally; use supplied TRAIN evidence only'}
    return result


def prepare(config_path):
    config_path = Path(config_path).resolve()
    cfg = read(config_path)
    if set(cfg) != {'schema', 'profile', 'source', 'query_ids', 'context', 'teacher', 'output_root'} or cfg['schema'] != 'saea-external-exploration-job-v1':
        raise ValueError('exact exploration job required')
    profile_path = checked(cfg['profile'], config_path.parent)
    profile = read(profile_path)
    if set(profile) != {'schema', 'algorithm', 'template', 'min_candidates', 'max_candidates', 'max_tokens'} or profile['schema'] != 'saea-external-exploration-profile-v1':
        raise ValueError('exact algorithm profile required')
    if profile['algorithm'] not in ('gepa', 'tfgrpo', 'ace', 'reflexion'):
        raise ValueError('unknown algorithm')
    if not (type(profile['min_candidates']) is int and type(profile['max_candidates']) is int and 2 <= profile['min_candidates'] <= profile['max_candidates'] <= 32):
        raise ValueError('bounded multiple candidates required')
    if type(profile['max_tokens']) is not int or profile['max_tokens'] < 131072:
        raise ValueError('128K output budget required')
    template = checked(profile['template'], profile_path.parent)
    source = checked(cfg['source'], config_path.parent)
    context_path = checked(cfg['context'], config_path.parent)
    ids = cfg['query_ids']
    if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)) or any(not isinstance(q, str) or not q.startswith('slurp:train:') for q in ids):
        raise ValueError('distinct explicit TRAIN query IDs required')
    groups = read(source)['document']['groups']
    by_id = {g['query_id']: g for g in groups}
    if len(by_id) != len(groups) or not set(ids) <= set(by_id):
        raise ValueError('missing or duplicate source groups')
    selected = []
    for query_id in ids:
        group = by_id[query_id]
        evidence = group['evidence']
        dictionary = evidence.get('development_evidence_dictionary', {})
        if dictionary and sha(dictionary) != evidence['development_evidence_dictionary_sha256']:
            raise ValueError('evidence dictionary changed')
        full = expand_evidence({k: v for k, v in evidence.items() if not k.startswith('development_evidence_')}, dictionary)
        if not {'supplied_evidence', 'task_instruction'} <= full['query'].keys() or not full['rollouts'] or any('request' not in r.get('source_trajectory', {}) for r in full['rollouts']):
            raise ValueError('complete actual query and request evidence required')
        # Never drop duplicate rollouts or alter their empirical weight.
        selected.append({'query_id': query_id, 'group_sha256': group['group_sha256'], 'evidence': full})
    context = read(context_path)
    if profile['algorithm'] == 'gepa':
        from speech_aware_evidence_acquisition.rails.solver_prompt_candidate import validate_contract
        contract = context['contract']
        validate_contract(contract, expected_contract_sha256=contract['contract_sha256'])
        if not set(ids) <= set(contract['feedback_query_ids']):
            raise ValueError('selected evidence outside GEPA feedback contract')
    teacher_cfg = dict(cfg['teacher'])
    if {'messages', 'output_root', 'research_prompt'} & teacher_cfg.keys():
        raise ValueError('job supplies messages, root and algorithm template')
    out = Path(cfg['output_root']).resolve()
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    (out / 'template.md').write_bytes(template.read_bytes())
    (out / 'template.md').chmod(0o600)
    executor_source = ROOT / 'scripts/environment/configured_external_learner.py'
    (out / 'executor.py').write_bytes(executor_source.read_bytes())
    executor = pin(out / 'executor.py')
    teacher_cfg.update(max_tokens=profile['max_tokens'], research_prompt=pin(out / 'template.md'))
    caller = ExternalTeacher(out / 'contacts', config=teacher_cfg, executor=executor)
    payload = {'algorithm': profile['algorithm'], 'context': request_context(context, profile['algorithm']), 'groups': selected,
               'min_candidates': profile['min_candidates'], 'max_candidates': profile['max_candidates'],
               'source_sha256': pin(source)['sha256'], 'rollout_multiplicity_preserved': True}
    # Re-factor repeated full evidence losslessly; no token-based truncation.
    from speech_aware_evidence_acquisition.rails.grounded_synthesis_fusion import factor_development_evidence
    packed = factor_development_evidence(payload)
    table = packed.get('development_evidence_dictionary', {})
    restored = expand_evidence({k: v for k, v in packed.items() if not k.startswith('development_evidence_')}, table)
    if restored != payload:
        raise ValueError('evidence packing must round-trip exactly')
    request = {'schema': 'saea-tfgrpo-learning-request-v1', 'model_id': caller.model_id,
               'learning_config_sha256': caller.learning_config_sha256,
               'stage': profile['algorithm'] + '-external-proposals-v1', 'identity': sha(payload),
               'messages': [{'role': 'user', 'content': canonical(packed).decode()}]}
    binding = caller.preflight(request)
    save(out / 'profile.json', profile)
    save(out / 'context.json', context)
    save(out / 'request.json', request)
    save(out / 'teacher.json', teacher_cfg)
    manifest = {'schema': 'saea-external-exploration-prepared-v1', 'executor': executor,
                'files': {name: pin(out / name) for name in ('profile.json', 'context.json', 'request.json', 'teacher.json', 'template.md')},
                'source': pin(source), 'job_sha256': pin(config_path)['sha256'], 'binding': binding,
                'groups': len(selected), 'rollouts': sum(len(g['evidence']['rollouts']) for g in selected),
                'new_model_calls': 0, 'candidate_accepted': False}
    save(out / 'prepared.json', manifest)
    return manifest


def load_prepared(root):
    root = Path(root).resolve()
    m = read(root / 'prepared.json')
    if m['schema'] != 'saea-external-exploration-prepared-v1':
        raise ValueError('prepared exploration required')
    for value in m['files'].values():
        checked(value)
    checked(m['executor'])
    caller = ExternalTeacher(root / 'contacts', config=read(root / 'teacher.json'), executor=m['executor'])
    return root, m, caller, read(root / 'request.json')


def validate_proposals(text, profile, context, evidence_ids=None):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate proposal key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON constant')
    doc = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(doc, dict) or set(doc) != {'algorithm', 'candidates'} or doc['algorithm'] != profile['algorithm']:
        raise ValueError('algorithm-specific proposal envelope required')
    candidates = doc['candidates']
    if not isinstance(candidates, list) or not profile['min_candidates'] <= len(candidates) <= profile['max_candidates']:
        raise ValueError('candidate count outside profile')
    if len({sha(c) for c in candidates}) != len(candidates):
        raise ValueError('duplicate proposals')
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError('candidate object required')
        if profile['algorithm'] == 'gepa':
            contract = context['contract']
            validate_mutation(canonical(candidate).decode(), contract=contract,
                              expected_contract_sha256=contract['contract_sha256'],
                              expected_parent_sha256=context['parent']['candidate_sha256'])
        else:
            fields = {'tfgrpo': {'id', 'target', 'evidence_query_ids', 'relative_feedback', 'hypothesis', 'intervention', 'counterexample', 'validation'},
                      'ace': {'id', 'reflection', 'operations', 'validation'},
                      'reflexion': {'id', 'evidence_query_ids', 'hypothesis', 'memory', 'retry_protocol', 'transfer_scope', 'validation'}}
            if set(candidate) != fields[profile['algorithm']]:
                raise ValueError('algorithm-specific candidate fields required')
            if not isinstance(candidate['id'], str) or not candidate['id'].strip():
                raise ValueError('nonempty candidate identity required')
        refs = (candidate.get('reflection', {}).get('evidence_query_ids')
                if profile['algorithm'] == 'ace' else candidate.get('evidence_query_ids'))
        if not isinstance(refs, list) or not refs or any(not isinstance(q, str) or not q.startswith('slurp:train:') for q in refs):
            raise ValueError('actual TRAIN evidence references required')
        if evidence_ids is not None and not set(refs) <= set(evidence_ids):
            raise ValueError('proposal references evidence not supplied in this request')
    if profile['algorithm'] != 'gepa' and len({c['id'] for c in candidates}) != len(candidates):
        raise ValueError('duplicate candidate identity')
    return doc


def request_payload(request):
    packed = json.loads(request['messages'][0]['content'])
    table = packed.get('development_evidence_dictionary', {})
    if table and sha(table) != packed['development_evidence_dictionary_sha256']:
        raise ValueError('packed request dictionary changed')
    return expand_evidence({k: v for k, v in packed.items() if not k.startswith('development_evidence_')}, table)


def collect(root):
    root, manifest, caller, request = load_prepared(root)
    observed = caller.reconcile(request)
    if observed is None:
        raise ValueError('no verified complete response; never resend unknown requests')
    response = observed['response']
    if response['finish_reason'] != 'stop':
        raise ValueError('incomplete proposal response')
    payload = request_payload(request)
    proposals = validate_proposals(response['text'], read(root / 'profile.json'), read(root / 'context.json'),
                                   evidence_ids=[g['query_id'] for g in payload['groups']])
    result = {'schema': 'saea-external-exploration-collected-v1', 'prepared_sha256': pin(root / 'prepared.json')['sha256'],
              'request_sha256': sha(request), 'response': observed, 'proposals': proposals,
              'semantic_validation': 'pending', 'deployment_compilation': 'pending', 'candidate_accepted': False}
    path = root / 'collected.json'
    if path.exists():
        if read(path) != result:
            raise ValueError('collected output changed')
    else:
        save(path, result)
    return result


def run(root):
    root, manifest, caller, request = load_prepared(root)
    # Existing executor persists intent and rejects a second dispatch.
    caller(request)
    return collect(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'collect'])
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    result = {'prepare': prepare, 'run': run, 'collect': collect}[args.action](args.path)
    print(json.dumps({'schema': result['schema'], 'candidate_accepted': False}))
