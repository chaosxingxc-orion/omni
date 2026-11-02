"""Independent review and native custody required before ACE development use.

Review is an external assessment, not a semantic truth inferred by this code.
This verifier never edits a candidate, sends a request or promotes deployment.
"""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .tfgrpo_learning import sha, validate_experience

SCHEMA = 'saea-learning-candidate-release-v1'
REVIEW_SCHEMA = 'saea-learning-semantic-review-v1'
STRATEGY = 'ace_incremental_offline_v1'
USAGE = ('calls', 'prompt_tokens', 'completion_tokens')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_pin(root, pin):
    from .tfgrpo_consumption import checked_pin
    path = checked_pin(root, pin)
    return path, json.loads(path.read_bytes())


def _usage(value):
    require(isinstance(value, dict) and set(value) == set(USAGE)
            and all(type(v) is int and v >= 0 for v in value.values()), 'exact learning usage required')
    return value


def verify_ace_source(*, repo, data, candidate, candidate_pin, source_manifest, bundle):
    """Recheck real source, full native/FIT inventory and deterministic updates.

    Recovered multi-attempt candidates are deliberately held until a finalized
    ancestry adapter exists; this does not change generation recovery support.
    """
    from ..contracts import ExecutionPlan
    from ..core.config import compose
    from ..core.contact_journal import ContactJournal
    from . import ace_runtime as runtime, ace_incremental as ace
    from .tfgrpo_runtime import contact_key
    plan = ExecutionPlan(**bundle.manifest['plan']).validate()
    base = Path(data) / 'runs' / plan.run_id / plan.attempt_id
    require(not (base / 'resume-declaration.json').exists(),
            'recovered ACE release requires finalized ancestry verification; no implicit fallback')
    require((Path(data) / source_manifest['relpath']).resolve() == base.resolve() / (plan.run_id + '.run-manifest.json')
            and (Path(data) / candidate_pin['relpath']).resolve() == base.resolve() / 'tfgrpo/learner/candidate.json',
            'candidate must belong to its exact finalized attempt')
    fragments = bundle.manifest['config']['fragments']
    cfg = compose(Path(repo), *(fragments[k] for k in ('model', 'dataset', 'baseline', 'experiment')))
    require(cfg.config_hash == bundle.manifest['config']['config_hash'], 'learning configuration changed')
    reg = cfg.values['tfgrpo_learning_registration']
    ids = tuple(dict.fromkeys(row['sample_id'] for row in reg['contacts']))
    loaded, _ = runtime.load_runtime_source(cfg.values, plan=plan, repo_root=Path(repo), data_root=Path(data),
        sample_ids=ids, roster_sha256=reg['construction_roster_sha256'])
    document, source = loaded['document'], loaded['source']
    require(json.loads((base / 'tfgrpo/learner/batch.json').read_bytes()) == document,
            'candidate original learning batch changed')
    require(candidate['batch_sha256'] == sha(document)
            and candidate['parent_sha256'] == document['parent']['sha256']
            and candidate['historical_source_custody'] == source
            and candidate['learned_query_ids'] == sorted(set(document['parent']['learned_query_ids']) | set(ids)),
            'candidate parent, source or learned query identities differ')
    journal = ContactJournal(Path(data), plan, 'prepared-context-text')
    request_ids = set()
    for path in journal.root.glob('*.intent.json'):
        intent = json.loads(path.read_bytes()); rid = intent['request_id']
        require(path == journal._path(rid, 'intent') and rid not in request_ids, 'native request identity differs')
        request_ids.add(rid)
    require(set(journal.root.glob('*.response.json')) == {journal._path(rid, 'response') for rid in request_ids},
            'orphan or missing finalized native response')
    attempt_path = Path(data) / 'attempts' / plan.run_id / (plan.attempt_id + '.json')
    attempt = json.loads(attempt_path.read_bytes())
    require(attempt.get('state') == 'FINALIZED' and attempt.get('contact_kind') == 'prepared-text'
            and attempt.get('run_id') == plan.run_id and attempt.get('attempt_id') == plan.attempt_id
            and attempt.get('actual') == {'calls': len(reg['contacts']), 'audio_seconds': 0.0}
            and len(request_ids) == len(reg['contacts']), 'complete finalized ACE native spend required')
    def assert_observed():
        require(all(journal.inspect(r)['state'] == 'observed' for r in request_ids), 'unknown ACE native contact')
    def assert_target(root, actual_plan, arm):
        require(Path(root).resolve() == Path(data).resolve() and actual_plan == plan
                and arm == 'prepared-context-text', 'finalized source audit identity differs')
    # The shared validator reconstructs every prompt and checks learner/bridge,
    # native response bytes, FIT hashes and usage. This view grants no replay.
    audit = SimpleNamespace(source=journal, source_plan=plan, request_ids=request_ids,
        assert_target=assert_target, assert_all_contacts_observed=assert_observed,
        source_binding=lambda: {'run_id': plan.run_id, 'attempt_id': plan.attempt_id,
            'attempt_sha256': hashlib.sha256(attempt_path.read_bytes()).hexdigest()})
    verified = runtime.verify_recovery_prefix(audit, data_root=Path(data), plan=plan,
        registration=reg, document=document, count_tokens=loaded['count_tokens'])
    require(len(verified['observed_prefix']) == len(reg['contacts'])
            and verified['known_usage'] == candidate['learning_usage'], 'candidate native usage differs')
    local = base / 'tfgrpo/learner/contacts'
    keys = {contact_key(row['stage'], row['identity']) for row in reg['contacts']}
    for directory in (local, base / 'tfgrpo/contacts'):
        for suffix in ('.intent.json', '.response.json'):
            require({p.name.removesuffix(suffix) for p in directory.glob('*' + suffix)} == keys,
                    'complete finalized learner and bridge inventory required')
    require(candidate['contact_response_sha256'] == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(local.glob('*.response.json'))}, 'candidate response inventory changed')
    current = _usage(candidate['learning_usage']); history = _usage(source['original_total_usage'])
    require(candidate['cumulative_known_usage'] == {k: history[k] + current[k] for k in USAGE},
            'candidate cumulative learning cost differs')
    items = copy.deepcopy(document['parent']['items']); operations = []; events = []
    for index, group in enumerate(document['groups']):
        identity = f'{document["input_sha256"]}:group-{index}'
        def text(stage):
            key = contact_key(stage, identity)
            return json.loads((local / (key + '.response.json')).read_bytes())['response']['text']
        reflection = ace.validate_reflection(text(ace.REFLECT), group, items, document['bounds']['max_findings'])
        merged = ace.merge_delta(text(ace.CURATE), items, reflection, document['bounds'])
        events.append({'group_index': index, 'query_id': group['query_id'], 'reflection': reflection,
            'merge': merged, 'before_items_sha256': sha(items), 'after_items_sha256': sha(merged['items'])})
        operations.extend(merged['operations']); items = merged['items']
    require(candidate['items'] == items and candidate['operations'] == operations and candidate['updates'] == events,
            'candidate differs from original observed ACE updates')
    conflicts = [{'query_id': e['query_id'], **c} for e in events for c in e['merge']['unresolved_conflicts']]
    require(candidate['unresolved_conflicts'] == conflicts, 'candidate conflict inventory changed')
    return {'document': document, 'source': source, 'known_usage': current, 'historical_usage': history}


def verify_review(review, *, release, candidate, document):
    fields = {'schema', 'decision', 'reviewer', 'procedure', 'candidate', 'source_manifest',
              'batch_sha256', 'parent_sha256', 'source_sha256', 'consumer_roster', 'items', 'conflicts'}
    require(isinstance(review, dict) and set(review) == fields and review['schema'] == REVIEW_SCHEMA
            and review['decision'] == 'eligible-for-development-evaluation', 'independent development review required')
    for key in ('reviewer', 'procedure'):
        require(isinstance(review[key], str) and review[key].strip(), 'named independent reviewer and procedure required')
    require(all(review[k] == release[k] for k in ('candidate', 'source_manifest', 'consumer_roster'))
            and review['batch_sha256'] == candidate['batch_sha256']
            and review['parent_sha256'] == candidate['parent_sha256']
            and review['source_sha256'] == sha(candidate['historical_source_custody']), 'review belongs to another candidate/source')
    groups = {g['query_id']: g['group_sha256'] for g in document['groups']}
    require(isinstance(review['items'], dict) and set(review['items']) == set(candidate['items']),
            'every retained item requires independent review')
    for item_id, content in candidate['items'].items():
        item = review['items'][item_id]
        require(isinstance(item, dict) and set(item) == {'content_sha256', 'decision', 'evidence', 'reason'}
                and item['content_sha256'] == sha(content) and item['decision'] == 'supported'
                and isinstance(item['reason'], str) and item['reason'].strip()
                and isinstance(item['evidence'], list) and item['evidence'], 'unsupported or changed retained item')
        for evidence in item['evidence']:
            require(isinstance(evidence, dict) and set(evidence) == {'query_id', 'group_sha256'}
                    and groups.get(evidence['query_id']) == evidence['group_sha256'], 'review cites foreign TRAIN evidence')
    conflicts = {sha(c): c for c in candidate.get('unresolved_conflicts', [])}
    require(isinstance(review['conflicts'], dict) and set(review['conflicts']) == set(conflicts),
            'all unresolved conflicts require explicit disposition')
    for decision in review['conflicts'].values():
        require(isinstance(decision, dict) and set(decision) == {'retained_items_affected', 'reason'}
                and decision['retained_items_affected'] is False
                and isinstance(decision['reason'], str) and decision['reason'].strip(),
                'unresolved conflict affects retained candidate')


def verify(params, *, repo, data, candidate, bundle):
    """Revalidate release at the actual consumption boundary, not only staging."""
    _, release = read_pin(data, params['candidate_release'])
    fields = {'schema', 'strategy', 'candidate', 'source_manifest', 'consumer_roster', 'learning_dataset', 'review'}
    require(isinstance(release, dict) and set(release) == fields and release['schema'] == SCHEMA
            and release['strategy'] == candidate.get('strategy') == STRATEGY, 'implemented candidate release strategy required')
    require(release['candidate'] == params['candidate'] and release['source_manifest'] == params['source_manifest'],
            'release candidate or finalized source differs')
    validate_experience(candidate, params['model_id'])
    require(release['consumer_roster']['relpath'] == 'docs/receipts/rosters/slurp-stage3-construction-r01c-h2.json'
            and release['learning_dataset']['relpath'] == 'configs/dataset/slurp-speech-native-stage3-construction-r01c.json',
            'fixed original H2 roster and TRAIN625 audio view required')
    _, roster = read_pin(repo, release['consumer_roster'])
    _, dataset = read_pin(repo, release['learning_dataset'])
    ids = [r['sample_id'] for r in roster['items']]
    require(len(ids) == 312 and len(set(ids)) == 312 and ids == params['consumer_ids']
            and all(i.startswith('slurp:train:') for i in ids), 'complete ordered TRAIN H2 review roster required')
    loaded = verify_ace_source(repo=repo, data=data, candidate=candidate, candidate_pin=params['candidate'],
                               source_manifest=params['source_manifest'], bundle=bundle)
    index = {r['sample_id']: r for r in dataset['items']}
    require(dataset.get('count') == 625 and len(index) == 625, 'complete original TRAIN625 audio view required')
    learned = set(candidate['learned_query_ids'])
    require(len(index) == len(dataset['items']) and learned | set(ids) <= set(index), 'learning and consumer audio custody required')
    def identities(sample_ids):
        utterances, audios = set(), set()
        for sid in sample_ids:
            require(sid.startswith('slurp:train:') and sid.count(':') >= 3, 'original TRAIN utterance identity required')
            utterances.add(':'.join(sid.split(':')[:3]))
            digest = index[sid]['audio_sha256']
            require(isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
                    'immutable real audio hash required')
            audios.add(digest)
        return utterances, audios
    train_u, train_a = identities(learned); eval_u, eval_a = identities(ids)
    require(not learned & set(ids) and not train_u & eval_u and not train_a & eval_a,
            'learning feedback overlaps evaluation query, utterance or audio')
    _, review = read_pin(data, release['review'])
    verify_review(review, release=release, candidate=candidate, document=loaded['document'])
    return release
