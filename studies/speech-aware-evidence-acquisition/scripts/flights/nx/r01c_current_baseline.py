"""Bind the completed R01C agreement to the actual two-contact A0 method.

Prepare a held construction baseline, with original prompt/contact semantics and
current real retrieval inputs. No admission, inference or full-A0 gain claim.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / 'src'), str(Path(__file__).parent)]
import r01c_current_comparator as comparator
import r01c_current_agreement as agreement
from speech_aware_evidence_acquisition.rails import consumption_module as cm
from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
read, sha, raw = comparator.read, comparator.sha, comparator.raw
CELL = 'slurp-a0-r01c-current-c2-v1'
RUN = 'SAEA-SLURP-A0-R01C-CURRENT-C2-V1'


def derive(repo, source_base, source_exp, retrieval_pins, agreement_sha, receipt_sha):
    for value in (agreement_sha, receipt_sha):
        if not isinstance(value, str) or len(value) != 64 or set(value) - set('0123456789abcdef') or value == '0' * 64:
            raise ValueError('actual agreement and receipt pins required')
    base, exp = comparator.derive(repo, source_base, source_exp, retrieval_pins)
    components = source_base['native_episode']['consumption_module']['components']
    original = [c for c in components if c['component_id'] == 'agree-vote-seed']
    if len(original) != 1 or original[0]['state'] != 'on':
        raise ValueError('actual enabled A0 agreement required')
    replacement = copy.deepcopy(original[0])
    pin = replacement['params']['agreement_source']
    pin.update(relpath=agreement.OUTPUT + '/agreement.jsonl', sha256=agreement_sha,
               tool_contact_ledger_row=agreement.ARTIFACT)
    pin['embedded_signal'] = pin['embedded_signal'].replace('TRAIN Fold B', 'TRAIN R01C')
    module = base['native_episode']['consumption_module']
    for i, c in enumerate(module['components']):
        if c['component_id'] == 'agree-vote-seed': module['components'][i] = replacement
    compiled = cm.compile_module(module, carrier_module_hooks(repo_root=repo), repo_root=repo)
    expected = compiled.values_on_roster_of(base)['knowledge_stores']['intent-boundaries']['phase_supply']['perception']
    phase = base['knowledge_stores']['intent-boundaries']['phase_supply']['perception']
    for key in ('agreement_source', 'k', 'rank_weights'): phase[key] = copy.deepcopy(expected[key])
    base['preparation_source_receipts'].append({'root': 'data', 'relpath': agreement.OUTPUT + '/receipt.json', 'sha256': receipt_sha})
    base['native_contact_fit'].update(run_id=RUN + '-FIT', attempt_id='fit-v1', contact_cap=1250)
    base['what_this_is'] = 'Actual two-contact A0 on full R01C625; held for full A0 gain lock, supply and independent admission.'
    cm.verify_declared_module(module, base, repo_root=repo)
    def semantics(block):
        block = copy.deepcopy(block); block.pop('lock')
        for c in block['components']:
            if c['component_id'] == 'agree-vote-seed': c['params'].pop('agreement_source')
        return block
    if semantics(module) != semantics(source_base['native_episode']['consumption_module']):
        raise ValueError('A0 module semantics changed')
    if base['native_episode']['roles'] != source_base['native_episode']['roles']:
        raise ValueError('A0 role prompts or decoding changed')
    cell, = exp['cells']
    cell.update(cell_id=CELL, baseline_fragment=CELL, run_id=RUN, attempt_id='generation-v1',
        readiness={'state': 'held', 'requirements': [], 'reason': 'Full A0 gain lock, own supply, registration and admission pending.'})
    exp.update(experiment_id=RUN, purpose='Frozen A0 material-source construction baseline on all625 R01C train rows; no test claim.')
    exp['stage3_algorithm'] = {'id': 'none', 'operators_used': [], 'note': 'Two-contact construction control; RL sampling is a separate successor.'}
    exp['stage3_measurement'].update(claims_benefit=False, note='Construction material source; first-system gain requires full Fold B read.')
    if 'execution_plan' in exp:
        import math
        view = read(repo / comparator.inputs.DATASET)
        exp['execution_plan'].update(planned_model_calls=2500,
            planned_speech_audio_seconds=math.ceil(4 * view['audio_seconds_total']))
    return base, exp


def write(repo, source_repo, data, comparator_complete, agreement_complete, retrieval_complete, out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    for tree in (repo, source_repo):
        if subprocess.check_output(['git', '-c', 'safe.directory=' + str(tree), '-C', str(tree), 'status', '--porcelain']).strip():
            raise ValueError('clean source and destination required')
    comp, agree, prep = read(comparator_complete), read(agreement_complete), read(retrieval_complete)
    if (comp['schema'] != 'saea-r01c-current-comparator-v1' or comp['rows'] != 625
            or comp['retrieval_completion_sha256'] != sha(retrieval_complete)
            or agree['comparator_completion_sha256'] != sha(comparator_complete)
            or agree['rows'] != 625 or agree['status'] != 'complete'
            or prep['rows'] != 625 or not prep['values_unchanged']):
        raise ValueError('complete linked R01C preparation required')
    for rel, digest in comp['output_hashes'].items():
        if sha(comparator.confined(repo, rel)) != digest: raise ValueError('comparator source changed')
    for rel, digest in prep['pins'].items():
        if sha(comparator.confined(data, rel)) != digest: raise ValueError('retrieval artifact changed')
    ids = [r['sample_id'] for r in read(repo / comparator.inputs.ROSTER)['items']]
    receipt_path = data / agreement.OUTPUT / 'receipt.json'; receipt = read(receipt_path)
    agreement.verify_artifact(data, receipt, ids)
    if agree['receipt_sha256'] != sha(receipt_path) or agree['artifact_sha256'] != receipt['artifact']['sha256']:
        raise ValueError('agreement binding changed')
    from speech_aware_evidence_acquisition.core.bundle import verify_bundle
    result = read(data / 'work/codex/real500-tp4-s96-verified-result-v1.json')
    mp = comparator.confined(data, result['selection_evidence']['manifest']['relpath'])
    if sha(mp) != comp['source_manifest_sha256'] or sha(mp) != result['selection_evidence']['manifest']['sha256']:
        raise ValueError('A0 source manifest differs')
    bundle = verify_bundle(mp)
    head = subprocess.check_output(['git', '-c', 'safe.directory=' + str(source_repo), '-C', str(source_repo), 'rev-parse', 'HEAD'], text=True).strip()
    if head != bundle.manifest['study_commit'] or bundle.attempt_state != 'FINALIZED':
        raise ValueError('completed A0 method source required')
    fragments = bundle.manifest['config']['fragments']
    from speech_aware_evidence_acquisition.core.config import compose
    if compose(source_repo, **fragments).config_hash != bundle.manifest['config']['config_hash']:
        raise ValueError('source config changed')
    b = read(source_repo / f"configs/baseline/{fragments['baseline']}.json")
    e = read(source_repo / f"configs/experiment/{fragments['experiment']}.json")
    base, exp = derive(repo, b, e, prep['pins'], agree['artifact_sha256'], agree['receipt_sha256'])
    outputs = {f'configs/baseline/{CELL}.json': raw(base), f'configs/experiment/{CELL}.json': raw(exp)}
    out = comparator.confined(data, str(out))
    if not out.is_relative_to(data.resolve() / 'work/codex') or out.exists() or any((repo / p).exists() for p in outputs):
        raise ValueError('fresh owned outputs required')
    record = {'schema': 'saea-r01c-current-baseline-v1', 'rows': 625, 'source_manifest_sha256': sha(mp),
        'source_commit': head, 'comparator_complete_sha256': sha(comparator_complete),
        'agreement_complete_sha256': sha(agreement_complete), 'retrieval_complete_sha256': sha(retrieval_complete),
        'model_contacts': 0, 'readiness': 'held', 'normal_generation_calls': 1250, 'fallback_inclusive_cap': 2500,
        'claims_full_a0_lock': False, 'output_hashes': {p: hashlib.sha256(v).hexdigest() for p, v in outputs.items()},
        'remaining': ['full A0 gain lock', 'selected resident binding', 'own phase supply', 'registered plans and admission']}
    out.mkdir(parents=True); publish_new(out / 'intent.json', raw(record))
    for p, v in outputs.items(): publish_new(repo / p, v); (repo / p).chmod(0o644)
    publish_new(out / 'complete.json', raw(record))
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('repo', 'source-repo', 'data', 'comparator-complete', 'agreement-complete', 'retrieval-complete', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    print(json.dumps(write(**vars(p.parse_args()))))
