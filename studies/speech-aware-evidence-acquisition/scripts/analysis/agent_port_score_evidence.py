"""Read the full Agent's sealed delivered score, never a first-visit branch score."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/analysis'))
import direct_port_score_evidence as shared
from speech_aware_evidence_acquisition.core.atomic_file import publish_new

SCHEMA = 'saea-agent-port-delivered-score-read-v1'


def require_read(repo, recipe_path, read_id, experiment):
    recipe_path = shared.within(recipe_path, repo)
    recipe = json.loads(recipe_path.read_bytes())
    entries = shared.registration.parse_ledger(repo, Path('docs/exposure-ledger.md')).get(read_id, [])
    if (len(entries) != 1 or not shared.registration.is_countersigned(entries[0])
            or entries[0].get('artifacts') != str(recipe_path.relative_to(repo))
            or shared.sha(recipe_path) not in entries[0].get('notes', '')):
        raise ValueError('Agent delivered-score READ must be uniquely registered and hash-pinned')
    if (recipe.get('schema') != SCHEMA or recipe.get('read_run_id') != read_id
            or recipe.get('quality_read') is not True or recipe.get('new_scoring_jobs') != 0
            or recipe.get('score_surface') != 'sealed-final-delivery'
            or recipe.get('experiments') != [experiment]):
        raise ValueError('Agent recipe must read the saved final delivery only')
    return recipe


def build(repo, data, experiment, read_id, recipe_path):
    if not shared.re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', experiment):
        raise ValueError('invalid Agent experiment identity')
    recipe = require_read(repo, recipe_path, read_id, experiment)
    doc = json.loads((repo / 'configs/experiment' / (experiment + '.json')).read_bytes())
    cell, = doc['cells']
    base = json.loads((repo / 'configs/baseline' / (cell['baseline_fragment'] + '.json')).read_bytes())
    if (base.get('policy') != 'phase-native-module-v1-c17'
            or not base.get('native_episode') or doc['family_budget']['official_complete_scoring_jobs'] != 1):
        raise ValueError('requires the complete c17 Agent campaign')
    ids = doc['populations'][0]['ids']
    if len(ids) != len(set(ids)) or len(ids) != doc['population']['count'] or len(ids) != 1080:
        raise ValueError('the complete unique 1080-row Fold B roster is required')
    expected = {'run_id': cell['run_id'], 'attempt_id': cell['attempt_id'], 'rows': len(ids),
                'split_identity_hash': doc['population']['split_identity_hash']}
    if recipe.get('identity') != expected:
        raise ValueError('Agent recipe identity differs from the full campaign')
    directory = data / 'runs' / cell['run_id'] / cell['attempt_id']
    bundle = shared.verify_bundle(directory / (cell['run_id'] + '.run-manifest.json'))
    receipts_path = directory / 'flight-receipts.json'
    receipts = shared.read_flight_receipts(receipts_path)
    preflight = receipts['stages']['preflight']
    if ((receipts['run_id'], receipts['attempt_id']) != (cell['run_id'], cell['attempt_id'])
            or bundle.manifest['run_id'] != cell['run_id']
            or bundle.manifest['study_commit'] != preflight['study_commit']
            or bundle.manifest['config']['config_hash'] != preflight['config_hash']
            or bundle.manifest['protocol_hash'] != preflight['protocol_hash']):
        raise ValueError('Agent executed identity and preflight disagree')
    result = shared.verify_saved_score(bundle, receipts, directory, ids)
    result.update(schema='saea-agent-port-delivered-score-evidence-v1', **expected,
                  read_run_id=read_id, score_surface='sealed-final-delivery', claims_benefit=False)
    result['inputs'][str(receipts_path)] = shared.sha(receipts_path)
    result['inputs'][str(recipe_path)] = shared.sha(recipe_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=REPO)
    for name in ('data-root', 'recipe', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('experiment', 'read-id'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    result = build(args.repo_root, args.data_root, args.experiment, args.read_id, args.recipe)
    publish_new(args.out, (json.dumps(result, indent=2, sort_keys=True) + '\n').encode())
    print(json.dumps({'rows': result['rows'], 'score_surface': result['score_surface'],
                      'new_scoring_jobs': 0}))


if __name__ == '__main__':
    main()
