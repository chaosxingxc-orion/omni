"""Derive a never-flown Fold A agreement-builder input from the actual A0 source.

This is a model-free config transform, not a flight registration or an A0 lock.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
import fold_a_retrieval_inputs as inputs
from speech_aware_evidence_acquisition.rails import consumption_module as cm
from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks

CELL = 'slurp-agent-fold-a-current-comparator-naturalend-npu-v1'
RUN = 'SAEA-SLURP-AGENT-FOLD-A-CURRENT-COMPARATOR-NATURALEND-NPU-V1'
PKG = 'docs/readiness/2026-10-31-fold-a-current-comparator-npu-v1'
SPEC = 'docs/readiness/2026-10-30-slurp-agent-fold-b-naturalend-npu-v1/successor-draft-v1.json'
SOURCE_COMPLETE = 'work/codex/agent-after-recovered-direct-v1/campaign/complete.json'
RETRIEVAL_COMPLETE = 'work/codex/fold-a-retrieval-after-encoding-v1/preparation/complete.json'


def read(path): return json.loads(Path(path).read_bytes())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def raw(doc): return (json.dumps(doc, indent=2, sort_keys=True) + '\n').encode()


def derive(repo, baseline, experiment, pins):
    """Only roster/query bindings and the agreement-free bootstrap differ."""
    baseline, experiment = copy.deepcopy(baseline), copy.deepcopy(experiment)
    cm.verify_declared_module(baseline['native_episode']['consumption_module'], baseline, repo_root=repo)
    source_solver = copy.deepcopy(baseline['knowledge_stores']['train-exemplars']['phase_supply']['solver'])
    source_cell, = experiment['cells']
    model, session = source_cell['model_fragment'], experiment['service_session']
    protocol = read(repo / inputs.PROTOCOL)
    roster = read(repo / inputs.ROSTER)
    ids = [row['sample_id'] for row in roster['items']]
    if len(ids) != 1080 or len(set(ids)) != 1080 or any(not sid.startswith('slurp:train:') for sid in ids):
        raise ValueError('complete unique TRAIN Fold A roster required')
    source_protocol = inputs.PROTOCOL.replace('fold-a', 'fold-b')
    pairs = {
        read(repo / source_protocol)['query_rows_relpath']: protocol['query_rows_relpath'],
        inputs.RECEIPT.replace('FOLD-A', 'FOLD-B'): inputs.RECEIPT,
        inputs.ALIAS_RECEIPT.replace('FOLD-A', 'FOLD-B'): inputs.ALIAS_RECEIPT,
        inputs.SEGMENTS.replace('fold-a', 'fold-b'): inputs.SEGMENTS,
        inputs.CONTENT.replace('fold-a', 'fold-b'): inputs.CONTENT,
        inputs.PROVENANCE.replace('fold-a', 'fold-b'): inputs.PROVENANCE,
        inputs.ROSTER.replace('fold-a', 'fold-b'): inputs.ROSTER,
    }
    digests = {**pins, inputs.ROSTER: sha(repo / inputs.ROSTER)}
    for name in pairs.values():
        value = digests.get(name)
        if not isinstance(value, str) or len(value) != 64 or set(value) - set('0123456789abcdef') or value == '0' * 64:
            raise ValueError('missing real Fold A input pin: ' + name)
    agreement_b = 'prepared/SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-B-NPU-PORT-V1/receipt.json'
    baseline['preparation_source_receipts'] = [pin for pin in baseline['preparation_source_receipts']
                                              if pin['relpath'] != agreement_b]
    def rebind(node):
        if isinstance(node, list): return [rebind(value) for value in node]
        if not isinstance(node, dict): return node
        out = {key: rebind(value) for key, value in node.items()}
        if node.get('relpath') in pairs:
            target = pairs[node['relpath']]
            out.update(relpath=target, sha256=digests[target])
            if 'tool_contact_ledger_row' in out:
                out['tool_contact_ledger_row'] = out['tool_contact_ledger_row'].replace('FOLD-B', 'FOLD-A')
            if 'embedded_signal' in out:
                out['embedded_signal'] = out['embedded_signal'].replace('Fold B', 'Fold A')
        return out
    baseline = rebind(baseline)
    block = copy.deepcopy(baseline['native_episode']['consumption_module'])
    for i, component in enumerate(block['components']):
        if component['component_id'] == 'agree-vote-seed':
            block['components'][i] = {'component_id': 'agree-vote-seed', 'state': 'off'}
    compiled = cm.compile_module(block, carrier_module_hooks(repo_root=repo), repo_root=repo)
    # Preserve every shared port store pin. Only this component's compiled
    # perception fields change; the Solver recall used by agreement is identical.
    target = baseline['knowledge_stores']['intent-boundaries']['phase_supply']['perception']
    source = compiled.values_on_roster_of(baseline)['knowledge_stores']['intent-boundaries']['phase_supply']['perception']
    target.pop('agreement_source', None)
    target['k'] = source['k']
    target['rank_weights'] = source['rank_weights']
    baseline['native_episode']['consumption_module'] = block
    cm.verify_declared_module(block, baseline, repo_root=repo)
    if baseline['knowledge_stores']['train-exemplars']['phase_supply']['solver'] != rebind(source_solver):
        raise ValueError('A0 Solver recall semantics changed')
    baseline.pop('prepared_phase_supply', None)
    baseline['native_contact_fit'].update(run_id=RUN + '-FIT', attempt_id='never-flown-fit')
    baseline['what_this_is'] = 'Never-flown current-A0 Fold A comparator for self-excluded agreement arithmetic only.'
    experiment.pop('baseline_predecessor', None)
    experiment.update(experiment_id=RUN, date='2026-10-31', split_id=roster['roster_id'],
        purpose='Model-free agreement-builder input; no model contact, admission or system-gain claim.',
        core_read_design={'reading': 'none; never flown', 'this_cell': 'current-A0 agreement-free comparator'})
    experiment['population'] = {'count': 1080, 'dataset_fragment': Path(inputs.DATASET).stem,
        'roster_receipt': inputs.ROSTER, 'split_identity_hash': roster['split_identity_hash'],
        'decision_population': 'Previously exposed TRAIN Fold A development rows; no test claim.'}
    experiment['populations'] = [{'dataset_fragment': Path(inputs.DATASET).stem, 'ids': ids}]
    source_cell.update(cell_id=CELL, baseline_fragment=CELL, dataset_fragment=Path(inputs.DATASET).stem,
        run_id=RUN, attempt_id='never-flown', allowed_execution_stages=['check'],
        readiness={'state': 'held', 'requirements': [], 'reason': 'NEVER FLOWN: model-free builder input only; no flight admission.'})
    seconds = math.ceil(sum(row['audio_seconds'] for row in roster['items']) * 34 - 1e-9)
    experiment['execution_plan']['planned_speech_audio_seconds'] = seconds
    experiment['family_budget']['repeated_speech_audio_seconds_cap'] = seconds
    experiment['generation_budget']['definition'] = 'Inherited c17 structural ceilings only; this comparator is never flown and authorizes zero contacts.'
    if source_cell['model_fragment'] != model or experiment['service_session'] != session:
        raise ValueError('A0 serving identity changed')
    if 'agreement_source' in json.dumps(baseline):
        raise ValueError('comparator still contains agreement input')
    return baseline, experiment


def bound_parents(source_repo, data, binding):
    """Verify explicit queue custody without changing historical parent defaults."""
    if binding.get('schema') != 'saea-fold-a-comparator-parent-v1':
        raise ValueError('unknown comparator parent binding')
    paths = {key: inputs.confined(data, binding[key]) for key in
             ('source_complete', 'retrieval_complete', 'queue_state')}
    spec_path = inputs.confined(source_repo, binding['source_spec'])
    if sha(spec_path) != binding['source_spec_sha256']:
        raise ValueError('source specification changed')
    queue = read(paths['queue_state'])
    if (queue.get('schema') != 'saea-card-queue-state-v1'
            or queue.get('queue_id') != binding['queue_id']):
        raise ValueError('queue identity differs')
    for job in ('a0-campaign', 'a0-paired', 'fold-a-encoding', 'fold-a-retrieval'):
        item = queue.get('jobs', {}).get(job, {})
        if item.get('status') != 'done' or item.get('exit') != 0:
            raise ValueError('successful upstream queue job required: ' + job)
    complete, preparation = read(paths['source_complete']), read(paths['retrieval_complete'])
    if (complete.get('commit') != binding['source_commit']
            or complete.get('run_id') != binding['source_run']):
        raise ValueError('completed A0 differs from frozen parent')
    if (preparation.get('schema') != 'saea-fold-a-retrieval-inputs-v1'
            or preparation.get('source_commit') != binding['retrieval_commit']):
        raise ValueError('retrieval preparation differs from frozen parent')
    return spec_path, paths['source_complete'], paths['retrieval_complete']


def write(repo, source_repo, data, out, parent_binding=None):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    assert_execution_environment()
    for tree in (repo, source_repo):
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=tree).strip():
            raise ValueError('clean source and destination trees required')
    if parent_binding is None:
        source_complete, retrieval_complete = data / SOURCE_COMPLETE, data / RETRIEVAL_COMPLETE
        spec_path = source_repo / SPEC
        status = read(data / 'work/codex/agent-after-recovered-direct-v1/status.json')
        retrieval_status = read(data / 'work/codex/fold-a-retrieval-after-encoding-v1/status.json')
        if status['state'] != 'done' or retrieval_status['state'] != 'done':
            raise ValueError('complete current Agent source and retrieval required')
    else:
        spec_path, source_complete, retrieval_complete = bound_parents(source_repo, data, parent_binding)
    complete = read(source_complete)
    if complete['rows'] != 1080 or complete['score_surface'] != 'sealed-final-delivery':
        raise ValueError('complete current Agent source required')
    source_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source_repo, text=True).strip()
    if source_head != complete['commit']:
        raise ValueError('source checkout does not match actual A0 flight')
    spec = read(spec_path)
    if complete['run_id'] != spec['run_id']:
        raise ValueError('Agent completion identity differs')
    preparation = read(retrieval_complete)
    if preparation['rows'] != 1080 or preparation['values_unchanged'] is not True:
        raise ValueError('complete verified Fold A retrieval inputs required')
    for name, digest in preparation['pins'].items():
        if sha(inputs.confined(data, name)) != digest:
            raise ValueError('Fold A retrieval artifact changed')
    base_rel = f"configs/baseline/{spec['baseline']}.json"
    exp_rel = f"configs/experiment/{spec['cell']}.json"
    source_exp = read(source_repo / exp_rel)
    model_rel = 'configs/model/' + source_exp['cells'][0]['model_fragment'] + '.json'
    if sha(repo / model_rel) != sha(source_repo / model_rel):
        raise ValueError('destination model fragment differs from actual A0')
    baseline, experiment = derive(repo, read(source_repo / base_rel), source_exp, preparation['pins'])
    outputs = {f'configs/baseline/{CELL}.json': raw(baseline), f'configs/experiment/{CELL}.json': raw(experiment)}
    record = {'source_commit': source_head, 'source_run': spec['run_id'],
        'source_files': {name: sha(source_repo / name) for name in (base_rel, exp_rel, model_rel)},
        'agent_completion_sha256': sha(source_complete),
        'retrieval_completion_sha256': sha(retrieval_complete),
        'output_hashes': {name: hashlib.sha256(blob).hexdigest() for name, blob in outputs.items()},
        'allowed_execution_stages': ['check'], 'model_contacts': 0, 'claims_system_lock': False}
    if parent_binding is not None:
        record['parent_binding'] = copy.deepcopy(parent_binding)
        record['queue_state_sha256'] = sha(inputs.confined(data, parent_binding['queue_state']))
    outputs[PKG + '/derivation.json'] = raw(record)
    out = inputs.confined(data, str(out))
    if out.exists() or out.is_relative_to(repo.resolve()) or any((repo / name).exists() for name in outputs):
        raise ValueError('fresh derivation paths required')
    out.mkdir(parents=True)
    publish_new(out / 'intent.json', raw(record))
    for name, blob in outputs.items():
        target = repo / name; target.parent.mkdir(parents=True, exist_ok=True)
        publish_new(target, blob); target.chmod(0o644)
    publish_new(out / 'complete.json', raw(record))
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, default=REPO)
    for name in ('source-repo', 'data-root', 'out'): p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--parent-binding', type=Path)
    a = p.parse_args()
    binding = read(a.parent_binding) if a.parent_binding else None
    print(json.dumps(write(a.repo_root, a.source_repo, a.data_root, a.out, binding), sort_keys=True))


if __name__ == '__main__': main()
