"""Compile the real odd40 candidate branches without authorizing generation.

This is the candidate portion only. The shared compiler produces four contacts;
the conditional adjudicator is intentionally not claimed by that ceiling.
"""
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src')]
from speech_aware_evidence_acquisition.rails import consumption_module as cm
from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks
from speech_aware_evidence_acquisition.rails.supply_variants import ODD40_OVERRIDE_PATH, ODD40_OVERRIDE_VALUE


def derive(repo, original):
    """Replace K4 diagnostic sampling with B, A/A and A on the same lock."""
    original_block = original['native_episode']['consumption_module']
    cm.verify_declared_module(original_block, original, repo_root=repo)
    if 'tfrl_v2_bank' in original_block:
        raise ValueError('unbanked source required for first odd40 construction')
    if any(c.get('kind') != 'decoding' for c in original_block['columns']) or len(original_block['columns']) != 4:
        raise ValueError('verified four-candidate source required')
    phase = original['knowledge_stores']['train-exemplars']['phase_supply']['solver']
    if (phase['k'] != 20 or phase['fuse_policy'] != 'saea-fuse-supplyfilter-v1'
            or phase['fuse_coverage'] != dict(m=20, c=20, top_l=20)
            or phase['rank_weights'] != dict(w_cos=1., w_prior=0., w_agree=0.)):
        raise ValueError('original cosine top20 evidence required')
    block = copy.deepcopy(original_block)
    frame = block.pop('decoding_frame')
    prefix = 'knowledge_stores.train-exemplars.phase_supply.solver.'
    block['columns'] = [{'column_id': 'b', 'kind': 'supply', 'node': 'solver',
        'max_supplied_items': frame['max_supplied_items'], 'overrides': {
            ODD40_OVERRIDE_PATH: ODD40_OVERRIDE_VALUE, prefix+'k': 40,
            **{prefix+'fuse_coverage.'+k: 40 for k in ('m', 'c', 'top_l')}}}]
    block['supply_frame'] = {'twin': True, 'fit_estimate': copy.deepcopy(frame['fit_estimate']),
        'delivering_max_supplied_items': frame['max_supplied_items']}
    compiled = cm.compile_module(block, carrier_module_hooks(repo_root=repo), repo_root=repo)
    base = compiled.apply_to(original)
    base['native_episode']['consumption_module'] = block
    expected = cm.policy_owned_values(original)
    expected['policy'] = compiled.program_id
    solver = expected['native_episode']['roles']['solver']
    solver.pop('decoding_variants')
    solver['supply_variants'] = copy.deepcopy(base['native_episode']['roles']['solver']['supply_variants'])
    if cm.policy_owned_values(base) != expected:
        raise ValueError('candidate compilation changed unrelated policy values')
    if compiled.program_ceiling != 4:
        raise ValueError('shared P plus B, A/A, A must compile to four contacts')
    # Never inherit consumed attempt identities or an old prepared supply.
    base.pop('prepared_phase_supply', None)
    base.pop('native_contact_fit', None)
    base['what_this_is'] = 'Unregistered odd40 candidate portion; conditional adjudicator integration outstanding.'
    cm.verify_declared_module(block, base, repo_root=repo)
    from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
    from speech_aware_evidence_acquisition.rails.native_episode_config import load_native_episode_config
    parsed = load_native_episode_config(
        {**base, 'carrier_lock_key': 'slurp'}, default_phase_program(base['policy']))
    if parsed.max_contacts != compiled.program_ceiling:
        raise ValueError('candidate compiler and runtime contact ceilings differ')
    return base, {'candidate_program_ceiling': compiled.program_ceiling,
        'candidate_runtime_config_loaded': True,
        'candidate_program_id': compiled.program_id,
        'candidate_arm_order': ['b', 'aa', 'lock'], 'a_arm': 'lock',
        'adjudicator_compiled': False, 'full_method_budget_compiled': False,
        'real_request_fit_completed': False, 'generation_authorized': False}


def derive_method(repo, original):
    """Compile the candidate and conditional-choice program; never admit it."""
    from speech_aware_evidence_acquisition.rails.whole_answer_selection import POLICY_ID
    from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
    from speech_aware_evidence_acquisition.rails.native_episode_config import load_native_episode_config
    base, proof = derive(repo, original)
    block = copy.deepcopy(base['native_episode']['consumption_module'])
    block['answer_selection'] = POLICY_ID
    compiled = cm.compile_module(block, carrier_module_hooks(repo_root=repo), repo_root=repo)
    result = compiled.apply_to(base)
    result['native_episode']['consumption_module'] = block
    result['what_this_is'] = 'Unregistered odd40 whole-answer selection; real FIT and admission outstanding.'
    expected = cm.policy_owned_values(base)
    expected['policy'] = compiled.program_id
    expected['native_episode']['answer_selection'] = POLICY_ID
    if cm.policy_owned_values(result) != expected:
        raise ValueError('whole-answer compilation changed candidate requests')
    parsed = load_native_episode_config({**result, 'carrier_lock_key': 'slurp'}, default_phase_program(result['policy']))
    if compiled.program_ceiling != 5 or parsed.max_contacts != 5:
        raise ValueError('conditional adjudicator must add exactly one contact')
    return result, {**proof, 'adjudicator_compiled': True,
        'method_program_ceiling': 5, 'method_program_id': compiled.program_id,
        'full_method_budget_compiled': True, 'normal_generation_min_per_row': 4,
        'normal_generation_max_per_row': 5, 'fallback_inclusive_cap_per_row': 10,
        'fit_cap_per_row': 5}


def prepare(repo, source, data, out, with_selection=False):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    assert_execution_environment()
    head = require_clean_study_repo(repo)
    source_head = require_clean_study_repo(source)
    if out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex'):
        raise ValueError('fresh owned output directory required')
    def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    pins, outputs, seen = {}, {}, set()
    for half, count in [('h1', 313), ('h2', 312)]:
        name = 'slurp-tfrl-current-seed-i1-'+half+'-v1'
        bp, ep = 'configs/baseline/'+name+'.json', 'configs/experiment/'+name+'.json'
        original = json.loads((source/bp).read_bytes())
        experiment = json.loads((source/ep).read_bytes())
        cell, = experiment['cells']
        population, = experiment['populations']
        ids = population['ids']
        dp = 'configs/dataset/'+cell['dataset_fragment']+'.json'
        view = json.loads((source/dp).read_bytes())
        items = {r['sample_id']: r for r in view['items']}
        if (len(view['items']) != 625 or len(items) != 625 or len(ids) != count
                or len(set(ids)) != count or set(ids)&seen or not set(ids) <= items.keys()):
            raise ValueError('exact disjoint real construction halves required')
        for sid in ids:
            row = items[sid]
            if (not sid.startswith('slurp:train:') or '/slurp_real/' not in row['media_relpath']
                    or not row['audio_sha256'] or not (data/row['media_relpath']).is_file()):
                raise ValueError('real training media binding required')
        seen.update(ids)
        for rel in (bp, ep, dp): pins[rel] = sha(source/rel)
        baseline, proof = (derive_method if with_selection else derive)(repo, original)
        outputs[half] = {'baseline': baseline, 'proof': proof, 'rows': count,
            'split_identity_hash': experiment['population']['split_identity_hash']}
    if len(seen) != 625: raise ValueError('complete625 required')
    if require_clean_study_repo(repo) != head or require_clean_study_repo(source) != source_head:
        raise ValueError('source changed during compilation')
    if any(sha(source/rel) != digest for rel,digest in pins.items()):
        raise ValueError('source bytes changed during compilation')
    out.mkdir(parents=True)
    def write(p, d): publish_new(p, (json.dumps(d, indent=2, sort_keys=True)+'\n').encode())
    for half, d in outputs.items(): write(out/(half+'-candidate-baseline.json'), d['baseline'])
    result = {'schema': 'saea-odd40-candidate-compilation-v1', 'rows':625,
        'implementation_commit':head, 'source_commit':source_head, 'source_hashes':pins,
        'halves':{h:{k:v for k,v in d.items() if k!='baseline'} for h,d in outputs.items()},
        'output_hashes':{h:sha(out/(h+'-candidate-baseline.json')) for h in outputs},
        'new_model_calls':0, 'new_tokenizer_calls':0, 'new_scoring_jobs':0,
        'generation_authorized':False, 'remaining':(
            ['real request FIT', 'own supply', 'registration', 'admission and preflight',
             'partial-contact recovery validation'] if with_selection else
            ['conditional adjudicator runtime', 'full method budget compilation', 'own supply',
             'registration and real FIT', 'admission and preflight'])}
    write(out/'complete.json', result)
    return result


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('repo','source','data','out'): p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--with-selection', action='store_true')
    print(json.dumps(prepare(**vars(p.parse_args()))))
