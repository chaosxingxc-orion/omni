"""Derive the current A0 method on Fold A, held until independent admission.

The agreement-free comparator is a preparation input, never the evaluated arm.
Restore the actual source component and bind only new data and spend identities.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
import fold_a_current_comparator as comparator
import fold_a_current_agreement as agreement
from speech_aware_evidence_acquisition.rails import consumption_module as cm
from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks
from speech_aware_evidence_acquisition.core.atomic_file import publish_new

read, sha, raw = comparator.read, comparator.sha, comparator.raw
CELL = 'slurp-agent-current-a0-fold-a-npu-v1'
RUN = 'SAEA-SLURP-AGENT-CURRENT-A0-FOLD-A-NPU-V1'
FIT = RUN + '-FIT'
PACKAGE = 'docs/readiness/2026-10-31-current-a0-fold-a-npu-v1'
DECISION = 'slurp-agent-current-a0-fold-a-npu-admission-v1'
DECISION_PATH = 'coordination/slurp-agent-current-a0-fold-a-npu-v1/root-admission-v1.json'


def derive(repo, source_base, source_exp, retrieval_pins, agreement_sha, receipt_sha):
    for pin in (agreement_sha, receipt_sha):
        if not isinstance(pin, str) or len(pin) != 64 or set(pin) - set('0123456789abcdef') or pin == '0'*64:
            raise ValueError('real agreement and receipt pins required')
    baseline, experiment = comparator.derive(repo, source_base, source_exp, retrieval_pins)
    originals = [c for c in source_base['native_episode']['consumption_module']['components']
                 if c['component_id'] == 'agree-vote-seed']
    if len(originals) != 1 or originals[0]['state'] != 'on':
        raise ValueError('actual A0 agreement component must be enabled')
    component = copy.deepcopy(originals[0])
    pin = component['params']['agreement_source']
    pin.update(relpath=agreement.OUTPUT + '/agreement.jsonl', sha256=agreement_sha,
               tool_contact_ledger_row=agreement.ARTIFACT)
    pin['embedded_signal'] = pin['embedded_signal'].replace('Fold B', 'Fold A')
    block = baseline['native_episode']['consumption_module']
    for index, current in enumerate(block['components']):
        if current['component_id'] == 'agree-vote-seed': block['components'][index] = component
    compiled = cm.compile_module(block, carrier_module_hooks(repo_root=repo), repo_root=repo)
    target = baseline['knowledge_stores']['intent-boundaries']['phase_supply']['perception']
    compiled_phase = compiled.values_on_roster_of(baseline)['knowledge_stores']['intent-boundaries']['phase_supply']['perception']
    for key in ('k', 'rank_weights', 'agreement_source'):
        target[key] = copy.deepcopy(compiled_phase[key])
    baseline['preparation_source_receipts'].append(
        {'root': 'data', 'relpath': agreement.OUTPUT + '/receipt.json', 'sha256': receipt_sha})
    baseline['preparation_source_receipts'].sort(key=lambda pin: (pin['root'], pin['relpath']))
    baseline['native_contact_fit'].update(run_id=FIT, attempt_id='fit-v1')
    baseline['what_this_is'] = 'Current frozen A0 method on all TRAIN Fold A rows; held pending independent admission.'
    cm.verify_declared_module(block, baseline, repo_root=repo)
    # Equality after removing only the agreement input pin proves no component,
    # sampling column, module lock or review program was silently dropped.
    def without_input(module):
        result = copy.deepcopy(module)
        for c in result['components']:
            if c['component_id'] == 'agree-vote-seed': c['params'].pop('agreement_source')
        return result
    if without_input(block) != without_input(source_base['native_episode']['consumption_module']):
        raise ValueError('A0 module semantics changed')
    cell, = experiment['cells']
    cell.update(cell_id=CELL, baseline_fragment=CELL, run_id=RUN, attempt_id='generation-v1',
        allowed_execution_stages=['check'], readiness={'state': 'held',
            'reason': 'Derived only; register, mint, prepare supply and preflight before model contact.',
            'requirements': [
                {'kind': 'decision', 'expected_decision_id': DECISION, 'relpath': DECISION_PATH,
                 'sha256': '0'*64, 'reason': 'Independent Fold A evaluation admission is pending.'},
                {'kind': 'preparation', 'relpath': agreement.OUTPUT+'/agreement.jsonl',
                 'sha256': agreement_sha, 'reason': 'Current A0 self-excluded Fold A agreement.'},
                {'kind': 'preparation', 'relpath': f'prepared/{RUN}/phase-supply-v1.json',
                 'sha256': '0'*64, 'reason': 'Own current-A0 Fold A first-phase supply is pending.'}]})
    experiment.update(experiment_id=RUN,
        purpose='Evaluate the unchanged current A0 method on all 1080 TRAIN Fold A rows; no test or TFRL claim.',
        baseline_predecessor={'cell':source_exp['cells'][0]['cell_id'],
            'run_id':source_exp['cells'][0]['run_id'], 'attempt_id':source_exp['cells'][0]['attempt_id'],
            'use':'Actual completed A0 method and service source; new Fold A inputs, not equal supply outputs.'},
        core_read_design={'reading':'Complete official final-delivery score; no benefit claim before paired analysis.',
                          'this_cell':'Current A0 Fold A evaluation, held pending registration.'})
    seconds = experiment['execution_plan']['planned_speech_audio_seconds']
    experiment['generation_budget']['definition'] = (
        f'Inherited c17 contact ceilings; {seconds} Fold A repeated audio seconds. '
        'Held draft: independent registration must validate measured accelerator allowance before contact.')
    experiment['generation_budget']['accelerator_bound'] = {
        'status':'PENDING-FOLD-A-ADMISSION', 'inherited_card_hours':source_exp['execution_plan']['planned_gpu_hours']}
    experiment['failure_policy'] = 'Preserve partial evidence; no automatic replay. Reconcile failure before recovery.'
    return baseline, experiment


def write(repo, source_repo, data, binding, comparator_complete, agreement_complete, out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    repo, source_repo, data = repo.resolve(), source_repo.resolve(), data.resolve()
    for tree in (repo, source_repo):
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=tree).strip():
            raise ValueError('clean source and destination required')
    spec_path, source_complete, retrieval_complete = comparator.bound_parents(source_repo, data, binding)
    source_head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=source_repo,text=True).strip()
    complete = read(source_complete)
    if source_head != complete['commit'] or complete['rows'] != 1080 or complete['score_surface'] != 'sealed-final-delivery':
        raise ValueError('complete frozen A0 source required')
    comparator_complete = comparator.inputs.confined(data, str(comparator_complete))
    agreement_complete = comparator.inputs.confined(data, str(agreement_complete))
    comp, agree = read(comparator_complete), read(agreement_complete)
    if (comp['source_commit'] != source_head or comp['source_run'] != complete['run_id']
            or comp['agent_completion_sha256'] != sha(source_complete)
            or comp['retrieval_completion_sha256'] != sha(retrieval_complete)
            or agree['status'] != 'complete' or agree['rows'] != 1080
            or agree['comparator_completion_sha256'] != sha(comparator_complete)):
        raise ValueError('preparation ancestry differs from actual A0')
    retrieval = read(retrieval_complete)
    if retrieval['rows'] != 1080 or retrieval['values_unchanged'] is not True:
        raise ValueError('complete verified Fold A retrieval required')
    for name,digest in retrieval['pins'].items():
        if sha(comparator.inputs.confined(data,name)) != digest: raise ValueError('retrieval input changed')
    receipt_path = data/agreement.OUTPUT/'receipt.json'
    receipt = read(receipt_path)
    ids = [r['sample_id'] for r in read(repo/comparator.inputs.ROSTER)['items']]
    agreement.verify_artifact(data,receipt,ids)
    if agree['receipt_sha256'] != sha(receipt_path) or agree['artifact_sha256'] != receipt['artifact']['sha256']:
        raise ValueError('agreement evidence changed')
    spec = read(spec_path)
    exp_rel,base_rel = f"configs/experiment/{spec['cell']}.json",f"configs/baseline/{spec['baseline']}.json"
    source_exp,source_base = read(source_repo/exp_rel),read(source_repo/base_rel)
    model_rel = 'configs/model/'+source_exp['cells'][0]['model_fragment']+'.json'
    if sha(repo/model_rel) != sha(source_repo/model_rel): raise ValueError('actual A0 model bytes differ')
    base,exp = derive(repo,source_base,source_exp,retrieval['pins'],receipt['artifact']['sha256'],sha(receipt_path))
    outputs = {f'configs/baseline/{CELL}.json':raw(base),f'configs/experiment/{CELL}.json':raw(exp)}
    record = {'source_commit':source_head, 'source_run':complete['run_id'], 'parent_binding':binding,
        'source_files':{n:sha(source_repo/n) for n in (base_rel,exp_rel,model_rel)},
        'comparator_completion_sha256':sha(comparator_complete),'agreement_completion_sha256':sha(agreement_complete),
        'comparator_complete':str(comparator_complete.relative_to(data)),
        'agreement_complete':str(agreement_complete.relative_to(data)),
        'model_contacts':0,'launchable':False,'allowed_execution_stages':['check'],
        'outputs':{n:hashlib.sha256(b).hexdigest() for n,b in outputs.items()},
        'remaining':['independent registration and mint','own phase supply','clean-source preflight','guarded dispatch']}
    outputs[PACKAGE+'/derivation.json'] = raw(record)
    out = comparator.inputs.confined(data,str(out))
    if out.exists() or out.is_relative_to(repo) or any((repo/n).exists() for n in outputs):
        raise ValueError('fresh derivation paths required')
    out.mkdir(parents=True,exist_ok=False)
    publish_new(out/'intent.json',raw(record))
    for n,blob in outputs.items(): publish_new(repo/n,blob); (repo/n).chmod(0o644)
    publish_new(out/'complete.json',raw(record))
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo-root','source-repo','data-root','parent-binding','comparator-complete','agreement-complete','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(write(a.repo_root,a.source_repo,a.data_root,read(a.parent_binding),
                           a.comparator_complete,a.agreement_complete,a.out),sort_keys=True))


if __name__=='__main__': main()
