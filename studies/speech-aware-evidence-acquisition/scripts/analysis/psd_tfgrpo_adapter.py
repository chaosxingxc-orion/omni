"""Offline, explicitly scoped TF-GRPO experience adaptation for PSD v2.

The source release retains its original reviewed consumer roster; a separate
mapping records the human semantic adaptation and is not a new learning run.
"""
import copy
import json
from pathlib import Path

from speech_aware_evidence_acquisition.rails.tfgrpo_consumption import checked_pin
from speech_aware_evidence_acquisition.rails.tfgrpo_external_release import read_pin
from tfrl_hardset_a0_workload import confined, sha

SCHEMA = 'saea-psd-tfgrpo-consumer-v1'
KINDS = ('external-tfgrpo-v1', 'external-tfgrpo-reference-v1')


def verify_and_render(repo, data, descriptor, consumer_ids, view, model_id, base):
    if not isinstance(descriptor, dict) or set(descriptor) not in ({'schema','target','source','mapping','template'}, {'schema','target','source','template'}, {'schema','target','source','mapping','template','deployment_binding'}, {'schema','target','source','template','deployment_binding'}) or descriptor['schema'] != SCHEMA or descriptor['target'] not in ('slot','decision'):
        raise ValueError('explicit PSD target and source required')
    source = read_pin(data, descriptor['source'])
    if source.get('source_kind') not in KINDS:
        raise ValueError('only reviewed external TF-GRPO sources supported')
    from speech_aware_evidence_acquisition.rails.tfgrpo_consumption import validate_params
    validate_params(source)
    candidate = read_pin(data, source['candidate'])
    if 'deployment_binding' in descriptor:
        from speech_aware_evidence_acquisition.rails.tfgrpo_external_release import verify_deployment_binding
        verify_deployment_binding({'model_id':source['model_id'], 'runtime_model_id':model_id, 'deployment_binding':descriptor['deployment_binding']}, repo=repo, data=data)
    elif model_id != source.get('runtime_model_id', source['model_id']):
        raise ValueError('different deployment requires pinned actor equivalence')
    if source['source_kind'] == KINDS[0]:
        from speech_aware_evidence_acquisition.rails.tfgrpo_external_release import verify_source, render
    else:
        from speech_aware_evidence_acquisition.rails.tfgrpo_reference_release import verify_source, render
    verify_source(source, repo=repo, data=data, candidate=candidate)
    # Also check the original reviewed renderer and template; no legacy fallback.
    old_base = checked_pin(Path(repo)/'configs', source.get('contract_control', source['base_template'])).read_text()
    if checked_pin(Path(repo)/'configs', source['template']).read_text() != render(candidate, old_base, source['consumer_ids']):
        raise ValueError('original reviewed template differs')
    if 'mapping' not in descriptor:
        if candidate.get('psd_target') != descriptor['target']:
            raise ValueError('direct consumption requires target-specific learned candidate')
        items = candidate['items']
    else:
        if candidate.get('psd_target'):
            raise ValueError('target-specific candidate must be consumed without rewriting')
        mapping = read_pin(data, descriptor['mapping'])
        if set(mapping) != {'schema','target','candidate_sha256','items','rationale'} or mapping['schema'] != 'saea-psd-tfgrpo-semantic-map-v1' or mapping['target'] != descriptor['target'] or mapping['candidate_sha256'] != candidate['sha256'] or not isinstance(mapping['rationale'], str) or not mapping['rationale'].strip():
            raise ValueError('explicit derived semantic adaptation required')
        items = mapping['items']
        if not isinstance(items, dict) or not items or not set(items) <= set(candidate['items']) or any(not isinstance(v,str) or not v.strip() for v in items.values()):
            raise ValueError('mapping must cite existing experience IDs')
    rows = view.get('items', [])
    if [r['sample_id'] for r in rows] != list(consumer_ids):
        raise ValueError('exact consumer audio view required')
    original = read_pin(repo, read_pin(data, source['candidate_release'])['learning_dataset'])
    index = {r['sample_id']: r for r in original['items']}
    learned = candidate['learned_query_ids']
    if any(s not in index for s in learned):
        raise ValueError('missing learning audio identity')
    identity = lambda s: ':'.join(s.split(':')[:3])
    import re
    audios = {index[s]['audio_sha256'] for s in learned}
    if not learned or any(not isinstance(h, str) or not re.fullmatch('[a-f0-9]{64}', h) for h in audios):
        raise ValueError('valid learning audio digests required')
    learned_utterances = {identity(s) for s in learned}
    if any(not r['sample_id'].startswith('slurp:train:') or not isinstance(r.get('audio_sha256'), str) or not re.fullmatch('[a-f0-9]{64}', r['audio_sha256']) or r['audio_sha256'] in audios or identity(r['sample_id']) in learned_utterances for r in rows):
        raise ValueError('learning and PSD consumers overlap in utterance or audio')
    if descriptor['target'] == 'slot':
        boundary = 'Use these conditional experiences only for Slot extraction from available evidence. Keep candidate IDs and produce entities only from observed input.'
    else:
        boundary = 'Use these conditional experiences only to compare existing candidate IDs. Select one existing candidate ID; do not generate entities. Entities are copied from Slot.'
    return base + '\n\n' + boundary + '\n' + '\n'.join(f'[{k}]. {v}' for k,v in sorted(items.items())) + '\n'


def apply(base, repo, data, descriptor, consumer_ids, view, model_id):
    role = 'slot' if descriptor.get('target') == 'slot' else 'solver'
    original = confined(Path(repo)/'configs', base['phase_material'][role]['instruction_template']).read_text()
    text = verify_and_render(repo, data, descriptor, consumer_ids, view, model_id, original)
    pin = descriptor['template']
    if not pin.get('relpath','').startswith('configs/probes/'):
        raise ValueError('PSD template must be under configs/probes')
    path = confined(repo, pin['relpath'])
    if set(pin) != {'relpath','sha256'} or sha(path) != pin['sha256'] or path.read_text() != text:
        raise ValueError('PSD template differs from source and semantic mapping')
    out = copy.deepcopy(base)
    out['phase_material'][role]['instruction_template'] = pin['relpath'][len('configs/'):]
    return out


def main():
    import argparse
    from intent_slot_workload import NodeWorkload
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('repo','data','workload','descriptor','output'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    workload = json.loads(a.workload.read_text()); descriptor = json.loads(a.descriptor.read_text())
    if 'tfgrpo_experience' in workload: raise ValueError('base workload must not already contain TF-GRPO')
    base, _, _, view, binding = NodeWorkload(workload).derive(a.repo, a.data, workload['material'])
    role = 'slot' if descriptor['target'] == 'slot' else 'solver'
    original = confined(a.repo/'configs', base['phase_material'][role]['instruction_template']).read_text()
    text = verify_and_render(a.repo, a.data, descriptor, workload['sample_ids'], view, binding['model_fragment'], original)
    path = confined(a.repo, descriptor['template']['relpath'])
    if not descriptor['template']['relpath'].startswith('configs/probes/'): raise ValueError('probe path required')
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive publication and rollback on any failed validation or output write.
    created_template = False; created_output = False
    try:
        with path.open('x') as f:
            created_template = True; f.write(text)
        descriptor['template']['sha256'] = sha(path)
        workload['tfgrpo_experience'] = descriptor
        NodeWorkload(workload).derive(a.repo, a.data, workload['material'])
        with a.output.open('x') as f:
            created_output = True; json.dump(workload, f, ensure_ascii=False, indent=2); f.write('\n')
    except BaseException:
        if created_output: a.output.unlink(missing_ok=True)
        if created_template: path.unlink(missing_ok=True)
        raise

if __name__ == '__main__': main()
