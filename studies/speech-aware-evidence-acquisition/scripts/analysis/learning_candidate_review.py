"""Export native-verified TRAIN evidence for independent review; never approve."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/analysis')]
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.rails import learning_candidate_release as release
from speech_aware_evidence_acquisition.rails import tfgrpo_consumption as consumption
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import canonical, sha, validate_experience

SCHEMA = 'saea-learning-review-export-v1'
ROSTER = 'docs/receipts/rosters/slurp-stage3-construction-r01c-h2.json'
DATASET = 'configs/dataset/slurp-speech-native-stage3-construction-r01c.json'


def pin(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    release.require(path.is_relative_to(root) and path.is_file(), 'owned existing evidence file required')
    return {'relpath': str(path.relative_to(root)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def export_review(repo, data, *, candidate_path, manifest_path, out):
    """Create an explicitly pending review; original source verifier remains mandatory."""
    repo, data, out = Path(repo).resolve(), Path(data).resolve(), Path(out)
    release.require(not out.exists() and out.resolve().is_relative_to(data / 'work/codex'),
                    'fresh owned review directory required')
    cp, mp = pin(data, candidate_path), pin(data, manifest_path)
    _, candidate = release.read_pin(data, cp)
    validate_experience(candidate, candidate['model_id'])
    strategy = candidate.get('strategy')
    release.require(strategy in (release.STRATEGY, release.REFLEXION_STRATEGY),
                    'implemented review strategy required')
    bundle = consumption.verify_learning_seal({'candidate': cp, 'source_manifest': mp},
                                               data=data, candidate=candidate)
    verifier = (release.verify_reflexion_source if strategy == release.REFLEXION_STRATEGY
                else release.verify_ace_source)
    loaded = verifier(repo=repo, data=data, candidate=candidate, candidate_pin=cp,
                      source_manifest=mp, bundle=bundle)
    document = loaded['document']
    roster_pin, dataset_pin = pin(repo, repo / ROSTER), pin(repo, repo / DATASET)
    # Shared release verifier below checks original query, utterance and audio disjointness.
    manifest = {'schema': release.SCHEMA, 'strategy': strategy, 'candidate': cp,
                'source_manifest': mp, 'consumer_roster': roster_pin,
                'learning_dataset': dataset_pin, 'review': None}
    review = {'schema': release.REVIEW_SCHEMA, 'decision': 'pending',
              'reviewer': '', 'procedure': '', 'candidate': cp, 'source_manifest': mp,
              'consumer_roster': roster_pin, 'batch_sha256': candidate['batch_sha256'],
              'parent_sha256': candidate['parent_sha256'],
              'source_sha256': sha(candidate['historical_source_custody']),
              'items': {key: {'content_sha256': sha(content), 'decision': 'pending',
                             'evidence': [], 'reason': ''}
                        for key, content in candidate['items'].items()},
              'conflicts': {sha(c): {'retained_items_affected': None, 'reason': ''}
                            for c in candidate.get('unresolved_conflicts', [])}}
    evidence = {'schema': 'saea-learning-review-evidence-v1', 'strategy': strategy,
                'candidate': cp, 'source_manifest': mp, 'document_sha256': sha(document),
                'items': candidate['items'], 'updates': candidate['updates'],
                'groups': document['groups'], 'parent': document['parent'],
                'unresolved_conflicts': candidate.get('unresolved_conflicts', []),
                'learning_usage': loaded['known_usage'], 'historical_usage': loaded['historical_usage'],
                'cumulative_known_usage': candidate['cumulative_known_usage'],
                'native_source_verified': True, 'semantic_support_verified': False,
                'h2_labels_included': False, 'new_model_calls': 0}
    # Recheck the original bytes before publishing any files.
    for root, record in [(data, cp), (data, mp), (repo, roster_pin), (repo, dataset_pin)]:
        release.read_pin(root, record)
    out.mkdir(parents=True)
    for name, doc in [('evidence.json', evidence), ('review.pending.json', review),
                      ('release.pending.json', manifest)]:
        publish_new(out / name, canonical(doc))
    receipt = {'schema': SCHEMA, 'status': 'pending-independent-semantic-review',
               'strategy': strategy, 'items': len(candidate['items']), 'groups': len(document['groups']),
               'candidate': cp, 'source_manifest': mp,
               'artifacts': {n: pin(data, out / n) for n in
                             ('evidence.json', 'review.pending.json', 'release.pending.json')},
               'semantic_support_verified': False, 'candidate_deployment_accepted': False,
               'new_model_calls': 0,
               'limitations': ['Export does not grant evaluation or deployment authority.',
                               'Evaluation isolation is rechecked by the complete release verifier before use.',
                               'TRAIN text is private; do not publish this packet.']}
    publish_new(out / 'complete.json', canonical(receipt))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'data', 'candidate', 'manifest', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_review(args.repo, args.data, candidate_path=args.candidate,
                                  manifest_path=args.manifest, out=args.out), sort_keys=True))
