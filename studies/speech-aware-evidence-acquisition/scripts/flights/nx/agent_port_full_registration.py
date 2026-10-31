"""Register the unspent full Agent successor after accepted repair proof."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[3]
for directory in (REPO / 'src', Path(__file__).parent, REPO / 'scripts/flights/direct', REPO / 'scripts/analysis'):
    sys.path.insert(0, str(directory))
from speech_aware_evidence_acquisition.core.config import compose, protocol_hash
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from l338_port_deployment_binding import validate_binding
import l338_port_family as family
from direct_port_refresh_registration import accepted_proof
from agent_port_score_evidence import SCHEMA as READ_SCHEMA

from agent_port_campaign_family import selected

FAMILY = selected()
PKG = FAMILY.package
SOURCE = 'docs/readiness/2026-10-28-slurp-l338-lever-column-fold-b-npu-port-v1'
AUTH = 'docs/decisions/2026-10-30-owner-full-test-and-first-system-gain.md'
BRANCH_READ = FAMILY.branch_read
BRANCH_RECIPE = PKG + '/column-read-recipe-v1.json'


def read(path): return json.loads(path.read_bytes())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def raw(doc): return (json.dumps(doc, indent=2, sort_keys=True) + '\n').encode()


def register(repo, data, binding_path, archive, *, dry_run=False):
    repo, data, archive = repo.resolve(), data.resolve(), archive.resolve()
    if archive.exists() or not archive.is_relative_to(data) or archive.is_relative_to(repo):
        raise ValueError('fresh external registration archive required')
    if subprocess.check_output(['git', '-c', 'safe.directory=' + str(repo), 'status', '--porcelain'], cwd=repo).strip():
        raise ValueError('registration requires a clean study tree')
    spec = read(repo / PKG / 'successor-draft-v1.json')
    if (repo / PKG / 'root-admission-draft-v1.json').exists():
        raise ValueError('Agent already registered; never replay registration')
    ledger_path = repo / 'docs/exposure-ledger.md'
    ledger_before = ledger_path.read_bytes()
    for run in (spec['run_id'], spec['fit_run_id'], spec['read_run_id'], BRANCH_READ):
        if f'| {run} |' in ledger_before.decode() or any((data / directory / run).exists()
                                                      for directory in ('runs', 'attempts', 'prepared', 'analysis')):
            raise ValueError('Agent identity already registered or spent')
    if (data / spec['decision_relpath']).exists():
        raise ValueError('Agent receipt already minted')
    binding = validate_binding(read(binding_path), repo, data)
    proof, proof_pins = accepted_proof(repo, data, binding)
    exp_rel = f"configs/experiment/{spec['cell']}.json"
    base_rel = f"configs/baseline/{spec['baseline']}.json"
    exp, base = read(repo / exp_rel), read(repo / base_rel)
    cell, = exp['cells']
    if cell['readiness']['state'] != 'held' or base.get('prepared_phase_supply'):
        raise ValueError('unprepared held Agent required')
    predecessor = exp['baseline_predecessor']
    previous = read(repo / 'configs/experiment' / (predecessor['cell'] + '.json'))
    previous_base = read(repo / 'configs/baseline' / (previous['cells'][0]['baseline_fragment'] + '.json'))
    if exp['population'] != previous['population'] or exp['populations'] != previous['populations']:
        raise ValueError('full Agent roster changed')
    if {key for key in base.keys() | previous_base.keys() if base.get(key) != previous_base.get(key)} != {
        'prepared_phase_supply', 'native_contact_fit', 'what_this_is'}:
        raise ValueError('full Agent frozen baseline semantics changed')
    ids = exp['populations'][0]['ids']
    if len(ids) != len(set(ids)) or len(ids) != 1080:
        raise ValueError('full unique 1080-row roster required')
    if (cell['run_id'], cell['attempt_id'], base['native_contact_fit']['run_id'],
            base['native_contact_fit']['attempt_id']) != (spec['run_id'], spec['attempt_id'], spec['fit_run_id'], spec['fit_attempt_id']):
        raise ValueError('Agent spend identity differs')
    if exp['service_session'] != binding['session_id'] or cell['model_fragment'] != Path(binding['model_fragment']['relpath']).stem:
        raise ValueError('Agent deployment differs from accepted proof')
    hours = family.provisional_card_hours(1080,
        lock_rows_per_minute_per_instance=binding['lock_rows_per_minute_per_instance'],
        tensor_parallel_size=binding['tensor_parallel_size'])
    exp['execution_plan']['planned_gpu_hours'] = hours
    exp['family_budget']['gpu_hours_fit_inclusive'] = hours
    exp['generation_budget']['definition'] = (
        f"Full1080-row c17 program:18360 nominal generation contacts,36720 including fallback;"
        f"18360 inline FIT contacts under {spec['fit_run_id']};107264audio seconds;{hours} reserved NPU card-hours. "
        "Ceilings are not targets; cumulative across attempts. Stop on failure without automatic replay.")
    exp['generation_budget']['accelerator_bound'] = {
        'status': 'MEASURED-SYNTHETIC-ALLOWANCE', 'card_hours': hours,
        'limitation': 'Synthetic equivalent rate, not a four-row throughput extrapolation.'}
    with tempfile.TemporaryDirectory(prefix='agent-register-') as temporary:
        stage = Path(temporary)
        shutil.copytree(repo / 'configs', stage / 'configs')
        (stage / exp_rel).write_bytes(raw(exp))
        cfg = compose(stage, cell['model_fragment'], cell['dataset_fragment'], cell['baseline_fragment'], spec['cell'])
    split = exp['population']['split_identity_hash']
    protocol = protocol_hash(cfg.config_hash, cfg.values['arm'], split)
    head = subprocess.check_output(['git', '-c', 'safe.directory=' + str(repo), 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    now = datetime.now(timezone.utc).isoformat()
    identity = {**{key: spec[key] for key in ('cell', 'run_id', 'attempt_id', 'fit_run_id', 'fit_attempt_id')},
        'experiment': spec['cell'], 'rows': 1080, 'split_identity_hash': split,
        'composed_config_hash': cfg.config_hash, 'protocol_hash': protocol,
        'model_fragment': cell['model_fragment'], 'baseline_fragment': cell['baseline_fragment']}
    original_facts = read(repo / SOURCE / 'registered-facts-v1.json')
    measurement = original_facts['stage3_measurement'][predecessor['run_id']]
    bound = {'status': 'MEASURED-SYNTHETIC-ALLOWANCE', 'measurement': binding,
        'caps': {'arm': {'generation_calls': 36720, 'fit_endpoint_calls': 18360,
                        'speech_audio_seconds': 107264, 'card_hours': hours}},
        'formula': 'ceil_0.1(1.3 * 1080 * 8.52 * tensor_parallel_size / lock_rows_per_minute_per_instance / 60)',
        'proof_inputs': proof_pins,
        'limitation': 'Full task cost remains measured during execution; smoke is not a saturated throughput estimate.'}
    outputs = {exp_rel: raw(exp)}
    def document(name, value):
        rel = PKG + '/' + name
        outputs[rel] = raw(value)
        return {'path': rel, 'sha256': hashlib.sha256(outputs[rel]).hexdigest()}
    protocol_pin = document('protocol-registered-v1.json', {
        'schema': 'saea-registered-protocol-v1', 'cells': {'arm': identity}, 'registered_at_utc': now,
        'sampling_arms': original_facts['sampling_arms'], 'thinking_knobs': original_facts['thinking_knobs'],
        'accelerator_bound': bound, 'proof': proof, 'failure_policy': exp['failure_policy'],
        'score_surfaces': {'headline': 'sealed-final-delivery', 'secondary': 'first-solver-level2 branch diagnostics'}})
    facts_pin = document('registered-facts-v1.json', {
        'schema': 'saea-registered-facts-v1', 'identity': {'arm': identity},
        'stage3_measurement': {spec['run_id']: measurement}, 'registered_at_utc': now,
        'study_commit_at_registration': head, 'composed_config_hashes': {'arm': cfg.config_hash},
        'sampling_arms': original_facts['sampling_arms'], 'thinking_knobs': original_facts['thinking_knobs']})
    plans = {}
    for kind, run, attempt, calls in [('generation', spec['run_id'], spec['attempt_id'], 36720),
                                      ('fit', spec['fit_run_id'], spec['fit_attempt_id'], 18360)]:
        plan = copy.deepcopy(read(repo / SOURCE / (kind + '-plan-registered-v1.json')))
        plan.update(cell=spec['cell'], experiment=spec['cell'], composed_config_hash=cfg.config_hash,
            expected_session_id=binding['session_id'], preparation_base_commit=head, registered_at_utc=now,
            accelerator_bound=bound, family_budget=exp['family_budget'],
            protocol_document={'relpath': protocol_pin['path'], 'sha256': protocol_pin['sha256']},
            prepared_phase_supply={'relpath': f"prepared/{spec['run_id']}/phase-supply-v1.json", 'sha256': 'MINTED_AT_THE_FLIP'},
            status='REGISTERED-BEFORE-CONTACT; HELD until mint, own supply and preflight')
        plan['execution_plan'] = dict(exp['execution_plan'], schema='saea-execution-plan-v1',
            study='speech-aware-evidence-acquisition', run_id=run, attempt_id=attempt, carrier_lock_key='slurp',
            protocol_hash=protocol, split_identity_hash=split, planned_model_calls=calls)
        plan['serving'] = {'session_id': binding['session_id'], 'model_fragment': cell['model_fragment'], 'binding': binding}
        plan['execution_route'] = ('Shared campaign, one complete official score of final delivery. '
            'Secondary first-visit columns require their registered derived-copy read.' if kind == 'generation'
            else 'Shared inline native FIT under independent identity; no generation.')
        plans[kind] = document(kind + '-plan-registered-v1.json', plan)
    column_recipe = copy.deepcopy(read(repo / SOURCE / 'analysis-recipe-v1.json'))
    proof_exp = read(repo / 'configs/experiment/slurp-delivery-assets-proof-4-npu-v1.json')
    proof_cell, = proof_exp['cells']
    proof_base = read(repo / 'configs/baseline' / (proof_cell['baseline_fragment'] + '.json'))
    column_recipe.update(read_run_id=BRANCH_READ,
        run_ids={'read': BRANCH_READ, 'arm': spec['run_id'], 'fit': spec['fit_run_id'],
                 'smoke': proof_cell['run_id'], 'smoke_fit': proof_base['native_contact_fit']['run_id']},
        authority=AUTH, registered_at_utc=now, score_surface='first-solver-level2',
        identity={key: identity[key] for key in ('run_id', 'attempt_id', 'rows', 'split_identity_hash')},
        official_complete_scoring_jobs=6,
        stages={'columns': 'agent_port_column_read.py --stage columns: five complete branch scores and l1-support-sample0',
                'read': 'agent_port_column_read.py --stage read: unchanged branch rules and replicate noise model'},
        final_delivery_read=spec['read_run_id'],
        repeat_custody={'reader': 'scripts/analysis/agent_port_repeat_evidence.py',
            'copies': ['lock', 'aa', 'lock-rep2'], 'requires_all_final_output_equality': True,
            'requires_complete_procedure': True,
            'on_unqualified': 'register complete-system repeats; keep branch noise separate'})
    column_recipe['port_extensions']['status'] = 'REGISTERED-BEFORE-CONTACT; model-free secondary branch diagnostics'
    column_recipe['port_extensions']['open_items'] = [
        'System-level D/A comparison is separately registered; branch nomination is not A0 promotion.']
    column_pin = document('column-read-recipe-v1.json', column_recipe)
    recipe_pin = document('analysis-recipe-v1.json', {
        'schema': READ_SCHEMA, 'read_run_id': spec['read_run_id'], 'quality_read': True,
        'new_scoring_jobs': 0, 'score_surface': 'sealed-final-delivery', 'experiments': [spec['cell']],
        'identity': {key: identity[key] for key in ('run_id', 'attempt_id', 'rows', 'split_identity_hash')},
        'authority': AUTH, 'claims_benefit': False,
        'headline': 'Use final delivered complete official score as A0, paired against Direct in a separately registered comparison.',
        'secondary_column_recipe': column_pin,
        'secondary_column_status': 'Separately registered six complete official column scores; no branch scoring by the delivered-score reader.'})
    document('root-admission-draft-v1.json', {
        'schema': 'saea-campaign-decision-v1', 'status': 'DRAFT-NOT-MINTED', 'decision_id': spec['decision_id'],
        'eligible_run_ids': [spec['run_id'], spec['fit_run_id']],
        'eligible_attempt_ids': [spec['attempt_id'], spec['fit_attempt_id']],
        'authorization_basis': AUTH, 'owner': 'root', 'recorded_at_utc': None,
        'decision': 'Run the full frozen Agent baseline after accepted delivery repair proof; lock D0/A0 before TFRL.',
        'antecedents': {'protocol': protocol_pin, 'registered_facts': facts_pin, 'analysis_recipe': recipe_pin, 'column_recipe': column_pin,
                        'plans': plans, 'registration_base_commit': head},
        'budget': {**exp['family_budget'], 'accelerator_bound': bound},
        'population': {'count': 1080, 'split_role': 'discovery', 'split_identity_hash': split, 'test_contact_authorized': False},
        'service': {'expected_session_id': binding['session_id'], 'no_child_start_stop': True},
        'not_authorized': ['old attempt replay', 'automatic retry', 'test contact', 'TFRL promotion before D0/A0 lock']})
    document('flight-steps-v1.json', {'schema': 'saea-flight-steps-manifest-v1', 'package': PKG,
        'roster': exp['population']['roster_receipt'], 'stages': {'flight': {'experiment': spec['cell'], 'rows': 1080}}})
    lines = []
    for kind, run, calls in [('generation', spec['run_id'], 36720), ('fit', spec['fit_run_id'], 18360), ('read', spec['read_run_id'], 0), ('columns', BRANCH_READ, 0)]:
        is_read = kind in {'read', 'columns'}
        pin = {'read': recipe_pin, 'columns': column_pin}.get(kind) if is_read else plans[kind]
        fields = ['2026-10-30', run, 'model-free-check' if is_read else 'bounded-discovery-probe',
            {'read': 'Complete Agent final-delivery read', 'columns': 'Six complete official branch and stream scores; mechanism diagnostics'}.get(kind, 'Full frozen Agent ' + kind),
            'none' if is_read else cell['model_fragment'], 'slurp TRAIN Fold B 1080', 'discovery', split,
            'no', str(calls), '0' if is_read else '107264', '0' if is_read else str(hours),
            'Previous contacts remain accounted; predecessor attempts never reopened.', pin['path'],
            f"REGISTERED BEFORE CONTACT; root countersigned under {AUTH}; recorded by Codex. Plan SHA256 {pin['sha256']}; admission {spec['decision_relpath']}; no automatic retry."]
        lines.append('| ' + ' | '.join(fields) + ' |')
    outputs['docs/exposure-ledger.md'] = ledger_before + ('\n## Full Agent natural-end successor\n\n'
        '| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n'
        + '| ' + ' | '.join(['---'] * 15) + ' |\n' + '\n'.join(lines) + '\n').encode()
    outputs[PKG + '-preregistration.md'] = (
        '# Full Agent Fold B registration\n\nRegister the complete frozen 1,080-row c17 Agent under '
        + AUTH + '. All previous attempts remain closed. Headline A0 is the sealed final delivered official score, '
        'read without rescoring. First-visit branch diagnostics are distinct from system returns.\n\n'
        + '```json\n' + json.dumps({'stage3_measurement': {spec['run_id']: measurement}}, indent=2) + '\n```\n').encode()
    if ledger_path.read_bytes() != ledger_before or subprocess.check_output(['git', '-c', 'safe.directory=' + str(repo), 'status', '--porcelain'], cwd=repo).strip():
        raise ValueError('source tree changed during registration staging')
    if dry_run:
        return {'package': PKG, 'dry_run': True, 'files': len(outputs), 'card_hours': hours,
                'model_contacts': 0, 'outputs': {path: hashlib.sha256(blob).hexdigest() for path, blob in outputs.items()}}
    archive.mkdir(parents=True)
    publish_new(archive / 'intent.json', raw({'commit': head, 'proof': proof_pins, 'binding_sha256': sha(binding_path),
        'outputs': {path: hashlib.sha256(blob).hexdigest() for path, blob in outputs.items()}, 'model_contacts': 0}))
    for rel, blob in outputs.items():
        target = repo / rel
        if target.exists():
            previous_path = archive / 'before' / rel
            previous_path.parent.mkdir(parents=True, exist_ok=True)
            publish_new(previous_path, target.read_bytes())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
    publish_new(archive / 'complete.json', raw({'files': len(outputs), 'card_hours': hours, 'model_contacts': 0}))
    return {'registered': PKG, 'held': True, 'card_hours': hours, 'model_contacts': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=REPO)
    parser.add_argument('--dry-run', action='store_true')
    for name in ('data-root', 'binding', 'archive'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(register(args.repo_root, args.data_root, args.binding, args.archive, dry_run=args.dry_run)))


if __name__ == '__main__': main()
