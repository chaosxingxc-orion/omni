"""Register complete Direct Fold B score reads before any campaign contact."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
from speech_aware_evidence_acquisition.core.atomic_file import publish_new

PKG = 'docs/readiness/2026-10-30-slurp-direct-only-fold-b-closemarker-npu-v1'
READ = 'SAEA-SLURP-DIRECT-ONLY-FOLD-B-CLOSEMARKER-QUALITY-READ-NPU-V1'
AUTH = 'docs/decisions/2026-10-30-owner-full-test-and-first-system-gain.md'


def register(repo, data):
    facts = json.loads((repo / PKG / 'registered-facts-v1.json').read_bytes())
    specs = [facts['identity'][key] for key in ('lock', 'aa', 'ab')]
    identities = {}
    for spec in specs:
        doc = json.loads((repo / 'configs/experiment' / (spec['cell'] + '.json')).read_bytes())
        cell, = doc['cells']
        if cell['readiness']['state'] != 'held':
            raise ValueError('register quality READ while the unspent campaign is held')
        if (cell['run_id'], cell['attempt_id']) != (spec['run_id'], spec['attempt_id']):
            raise ValueError('campaign identity differs from registered facts')
        if doc['population']['count'] != 1080 or doc['family_budget']['official_complete_scoring_jobs'] != 1:
            raise ValueError('three complete Fold B passes are required')
        if any((data / base / cell['run_id'] / cell['attempt_id']).exists() for base in ('runs', 'attempts')):
            raise ValueError('quality read must be registered before campaign attempt')
        identities[spec['cell']] = {key: spec[key] for key in ('run_id', 'attempt_id', 'rows', 'split_identity_hash')}
    recipe = {'schema': 'saea-direct-port-complete-score-read-v1', 'read_run_id': READ,
              'quality_read': True, 'new_scoring_jobs': 0, 'experiments': list(identities),
              'identities': identities, 'authority': AUTH, 'claims_benefit': False,
              'scope': 'Read the saved complete official scores of three registered TRAIN Fold B Direct passes. No model calls, rescoring, test contact or A-D benefit claim.',
              'reader': 'scripts/analysis/direct_port_score_evidence.py'}
    path = repo / PKG / 'quality-analysis-recipe-v1.json'
    ledger = repo / 'docs/exposure-ledger.md'
    text = ledger.read_text()
    if path.exists() or '| ' + READ + ' |' in text:
        raise ValueError('quality READ already exists; never replay registration')
    raw = (json.dumps(recipe, indent=2, sort_keys=True) + '\n').encode()
    digest = hashlib.sha256(raw).hexdigest()
    fields = ['2026-10-30', READ, 'model-free-check', recipe['scope'], 'no model contact',
              'slurp TRAIN Fold B 1080', 'discovery', specs[0]['split_identity_hash'], 'no', '0', '0', '0',
              'Same three registered Direct passes; scores reused from their complete campaign jobs.',
              str(path.relative_to(repo)),
              f'REGISTERED BEFORE CONTACT; root countersigned under {AUTH}; recorded by Codex. Plan SHA256 {digest}. Zero new scoring jobs.']
    publish_new(path, raw)
    with ledger.open('a') as stream:
        stream.write('\n## Direct complete Fold B quality read\n\n| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n| ' + ' | '.join(['---'] * 15) + ' |\n| ' + ' | '.join(fields) + ' |\n')
    return {'recipe': str(path), 'read_run_id': READ, 'new_scoring_jobs': 0, 'model_contacts': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=REPO)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(register(args.repo_root, args.data_root)))


if __name__ == '__main__':
    main()
