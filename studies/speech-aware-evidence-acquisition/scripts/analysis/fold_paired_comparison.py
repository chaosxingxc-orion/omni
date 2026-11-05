"""Compare two complete sealed same-fold TRAIN runs, without model or scorer calls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/analysis')]
from speech_aware_evidence_acquisition.analysis.paired import paired_contrast
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.core.bundle import verify_bundle
from speech_aware_evidence_acquisition.contracts import RUN_ID_PATTERN
from slurp_paired_roster_read import parse_args, run_stages
from tfrl_h1_observation_read import summarize_scores
from tfrl_current_half_result import score_table
from h1_thinking_comparison import native_costs

SCHEMA = 'saea-fold-paired-comparison-v1'
SCORER = 'slurp-slu-official-train'
IDENTITY = ('evaluator_commit', 'scorer_version', 'split_id', 'projection', 'filler_normalization')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_tables(roster, entries):
    ids = [x['sample_id'] for x in roster['items']]
    if len(ids) != 1080 or len(set(ids)) != 1080:
        raise ValueError('full unique Fold A roster required')
    identities = [{k: e['aggregate'][k] for k in IDENTITY} for e in entries]
    if identities[0] != identities[1] or identities[0]['split_id'] != 'train':
        raise ValueError('scorer or normalizer identity differs')
    tables = [score_table(e, ids) for e in entries]
    strict = [{sid: {'strict': float(t[sid]['scenario_action_exact'] == 1 and
               t[sid]['slot_f1_exact_fp'] == 0 and t[sid]['slot_f1_exact_fn'] == 0)}
               for sid in ids} for t in tables]
    return ids, identities[0], tables, strict

def run(config_path, output, *, draws=None):
    c = json.loads(Path(config_path).read_bytes())
    if set(c) != {'schema','repo','data','roster','roster_sha256','split_identity_hash','arms','seed','draws','comparison_spec'} or c['schema'] != SCHEMA:
        raise ValueError('exact comparison configuration required')
    if set(c['arms']) != {'control','split'}:
        raise ValueError('exact two arm identities required')
    spec = c['comparison_spec']
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError('explicit comparison specification required')
    repo, data, output = Path(c['repo']), Path(c['data']), Path(output)
    roster_path = repo / c['roster']
    if sha(roster_path) != c['roster_sha256']:
        raise ValueError('roster pin differs')
    roster = json.loads(roster_path.read_bytes())
    if roster['split_identity_hash'] != c['split_identity_hash']:
        raise ValueError('split pin differs')
    entries, sources, summaries = [], {}, {}
    for label in ('control','split'):
        arm = c['arms'][label]
        if set(arm) != {'run_id','attempt_id'}:
            raise ValueError('explicit run and attempt required')
        run, attempt = arm['run_id'], arm['attempt_id']
        if any(not isinstance(v,str) or RUN_ID_PATTERN.fullmatch(v) is None for v in (run,attempt)):
            raise ValueError('confined run and attempt identities required')
        directory = data / 'runs' / run / attempt
        manifest = directory / (run + '.run-manifest.json')
        bundle = verify_bundle(manifest)
        if bundle.attempt_state != 'FINALIZED' or bundle.manifest['status'] != 'COMPLETED':
            raise ValueError('complete finalized arm required')
        if bundle.manifest['split']['identity_hash'] != c['split_identity_hash']:
            raise ValueError('fold identity differs')
        if bundle.manifest.get('run_id') != run or bundle.manifest.get('attempt', {}).get('attempt_id') != attempt:
            raise ValueError('sealed run or attempt identity differs')
        score_path = directory / (run + '.scores.json')
        score_meta = bundle.manifest.get('scores') or {}
        if Path(score_meta.get('path', '')).resolve() != score_path.resolve() or score_meta.get('sha256') != sha(score_path):
            raise ValueError('scores must be bound by sealed manifest')
        scores = json.loads(score_path.read_bytes())[SCORER]
        if len(bundle.outputs_rows) != 1080 or {r['sample_id'] for r in bundle.outputs_rows} != {r['sample_id'] for r in roster['items']}:
            raise ValueError('sealed outputs differ from full roster')
        entries.append(scores)
        summaries[label] = summarize_scores(scores, roster['items'], expected_rows=1080)
        dispositions, role_sequences = Counter(), Counter()
        for row in bundle.outputs_rows:
            receipt = row.get('native_episode_receipt') or {}
            disposition = receipt.get('terminal_disposition') or {}
            dispositions[disposition.get('reason') or 'no_terminal_reason'] += 1
            roles = tuple(v.get('stage') for v in receipt.get('role_visits', []))
            role_sequences[' -> '.join(roles) or 'none'] += 1
        summaries[label]['terminal_dispositions'] = dict(dispositions)
        summaries[label]['role_sequences'] = dict(role_sequences)
        if bundle.manifest.get('predecessor_attempt') or bundle.manifest.get('predecessor_attempts'):
            summaries[label]['native_role_costs'] = None
            summaries[label]['native_role_costs_unavailable'] = (
                'recovered rows use attempt-scoped predecessor traces; reconcile each attempt journal separately')
        else:
            summaries[label]['native_role_costs'] = native_costs(bundle.outputs_rows, bundle.trace_records)
        sources[label] = {'run_id':run, 'attempt_id':attempt, 'manifest_sha256':sha(manifest),
                          'scores_sha256':sha(score_path), 'cost':bundle.manifest['cost'],
                          'predecessor_attempt':bundle.manifest.get('predecessor_attempt'),
                          'predecessor_attempts':bundle.manifest.get('predecessor_attempts')}
    ids, identity, tables, strict = validate_tables(roster, entries)
    clusters = {x['sample_id']: x['utterance_id'] for x in roster['items']}
    n = c['draws'] if draws is None else draws
    contrast = paired_contrast(strict[0], strict[1], metrics=['strict'], sample_ids=ids,
        seed=c['seed'], n_replicates=n, ci_level=.95, cluster_of=clusters.__getitem__)['strict'].as_dict()
    if output.exists():
        raise ValueError('fresh private output required')
    args = parse_args(['--stage','all','--left-run',c['arms']['control']['run_id'],
        '--left-attempt',c['arms']['control']['attempt_id'],'--left-label','control',
        '--right-run',c['arms']['split']['run_id'],'--right-attempt',c['arms']['split']['attempt_id'],
        '--right-label','split','--scorer-id',SCORER,'--roster',c['roster'],
        '--analysis-root',str(output),'--repo-root',str(repo),'--data-root',str(data),
        '--seed',str(c['seed']),'--draws',str(n),'--expected-split-role','discovery',
        '--split-identity-hash',c['split_identity_hash']])
    run_stages(args)
    result = {'schema':SCHEMA,'rows':len(ids),'scorer_identity':identity,'arms':summaries,
              'strict_paired':contrast,'sources':sources,'new_model_calls':0,'new_scoring_jobs':0,
              'comparison_spec':spec,
              'limitations':['Slot thinking changes effective answer budget: off 4096; on 3072 reasoning plus 1024 answer',
                             'independent requests; sample uncertainty is not run-to-run uncertainty']}
    publish_new(output / 'fold-comparison.json', (json.dumps(result,indent=2,sort_keys=True)+'\n').encode())
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); run(a.config,a.output)
