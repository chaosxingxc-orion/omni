"""Compile verified external operations into a pending experience-library preview.

Does not fabricate a native learning bundle, expose labels to an actor, or
approve semantic correctness. The existing native release gate is unchanged.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/analysis')]
import external_exploration as external
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha
from speech_aware_evidence_acquisition.rails.tfgrpo_external_proposals import (
    CONTEXT_SCHEMA, STRATEGY, prepare_context, relative_evidence, validate_proposal)


def build_context(source, *, max_operations=2, max_items=48, max_characters=10000):
    document = external.read(source)['document']
    context = {k: document[k] for k in ('model_id', 'parent', 'reward_mode',
               'rollout_policy_sha256', 'rollout_kind', 'offline_stale_policy')}
    context.update(schema=CONTEXT_SCHEMA, bounds={'max_operations': max_operations,
                   'max_items': max_items, 'max_characters': max_characters})
    return prepare_context(context, document)


def compile_preview(spec):
    if not isinstance(spec, dict) or set(spec) != {'schema', 'prepared', 'collected', 'candidate_index'} or spec['schema'] != 'saea-tfgrpo-external-compile-v1':
        raise ValueError('exact external TF-GRPO compilation specification required')
    pp, cp = external.checked(spec['prepared']), external.checked(spec['collected'])
    if pp.name != 'prepared.json' or cp != pp.parent / 'collected.json':
        raise ValueError('original prepared/collected directory required')
    root, manifest, caller, request = external.load_prepared(pp.parent)
    record = external.read(cp)
    if record.get('schema') != 'saea-external-exploration-collected-v1' or record.get('prepared_sha256') != spec['prepared']['sha256']:
        raise ValueError('external collected preparation differs')
    source = external.checked(manifest['source'])
    context = prepare_context(external.read(root / 'context.json'), external.read(source)['document'])
    payload = external.request_payload(request)
    profile = external.read(root / 'profile.json')
    if payload['context'] != context or payload['algorithm'] != 'tfgrpo' or profile['algorithm'] != 'tfgrpo':
        raise ValueError('TF-GRPO source and parent context required')
    ledger = relative_evidence(payload['groups'])
    if ledger != payload.get('relative_evidence'):
        raise ValueError('relative evidence differs from complete supplied trajectories')
    observed = caller.reconcile(request)
    if observed is None or observed != record.get('response') or sha(request) != record.get('request_sha256'):
        raise ValueError('external response custody differs or is unknown')
    if observed['response']['finish_reason'] != 'stop':
        raise ValueError('incomplete external proposal generation')
    proposals = external.validate_proposals(observed['response']['text'], profile, context,
                  [g['query_id'] for g in payload['groups']], relative=ledger)
    if proposals != record.get('proposals') or record.get('candidate_accepted') is not False:
        raise ValueError('collected proposals differ from actual external response')
    index = spec['candidate_index']
    if type(index) is not int or not 0 <= index < len(proposals['candidates']):
        raise ValueError('explicit valid candidate index required')
    proposal = proposals['candidates'][index]
    items = validate_proposal(proposal, context, ledger)
    # Every displayed reference is privileged learner exposure, even if uncited.
    learned_ids = sorted(set(context['parent']['learned_query_ids']) |
                         {g['query_id'] for g in payload['groups']})
    body = {'schema': 'saea-tfgrpo-external-preview-v1', 'strategy': STRATEGY,
            'model_id': context['model_id'], 'parent_sha256': context['parent']['sha256'],
            'items': items, 'operations': proposal['operations'], 'proposal': proposal,
            'learned_query_ids': learned_ids, 'relative_evidence': ledger,
            'rollout_policy_sha256': context['rollout_policy_sha256'],
            'rollout_kind': context['rollout_kind'], 'offline_stale_policy': context['offline_stale_policy'],
            'source': manifest['source'], 'prepared': spec['prepared'], 'collected': spec['collected'],
            'request_sha256': sha(request), 'wire_response_sha256': observed['response']['wire_response_sha256'],
            'teacher_model_id': observed['response']['model_id'], 'provider_usage': observed['response']['provider_usage'],
            'learning_usage': observed['response']['usage'],
            'usage_scope': 'whole external request shared by sibling proposals; count once by request_sha256',
            'semantic_validation': 'pending', 'deployment_accepted': False,
            'native_learning_seal': False, 'candidate_publishable': False,
            'implementation': 'fused offline relative-feedback operation proposal, not native multistage reference TF-GRPO',
            'new_model_calls': 0}
    return {**body, 'sha256': sha(body)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    context_parser = sub.add_parser('context')
    context_parser.add_argument('--source', type=Path, required=True)
    for name, default in [('max-operations', 2), ('max-items', 48), ('max-characters', 10000)]:
        context_parser.add_argument('--' + name, type=int, default=default)
    compile_parser = sub.add_parser('compile')
    compile_parser.add_argument('--spec', type=Path, required=True)
    for p in (context_parser, compile_parser):
        p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('fresh output required; do not overwrite evidence')
    result = (build_context(args.source, max_operations=args.max_operations,
              max_items=args.max_items, max_characters=args.max_characters)
              if args.action == 'context' else compile_preview(external.read(args.spec)))
    external.save(args.out, result)
    print(result['schema'])
