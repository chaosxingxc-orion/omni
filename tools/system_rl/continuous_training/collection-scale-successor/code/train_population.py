"""The complete TRAIN denominator belongs to the pinned input, not the sampler."""
import re


def official_domains(dataset):
    def need(ok):
        if not ok:
            raise ValueError('OFFICIAL_FULL_TRAIN_REQUIRED')
    need(dataset.get('schema') == 'automatic-dataset-manifest-v1' and
         dataset.get('split') == 'TRAIN' and dataset.get('synthetic') is False and
         not dataset.get('bounded_selection') and not dataset.get('reference_ids'))
    names = {'train_real': 'real', 'train_synth': 'synthetic'}
    inputs = {r['path']: r for r in dataset['inputs']}
    need(len(inputs) == len(dataset['inputs']))
    counts, paths = {}, set()
    for domain in dataset['domains']:
        name = domain['domain']
        meta = domain['metadata']
        n = meta['recording_refs']
        path = meta['path']
        need(name in names and names[name] not in counts and not domain.get('metadata_lines') and
             path and path not in paths and path in inputs and
             isinstance(meta['sha256'], str) and re.fullmatch('[0-9a-f]{64}', meta['sha256']) and
             inputs[path]['sha256'] == meta['sha256'] and
             inputs[path].get('recording_refs', n) == n and type(n) is int and n > 0)
        counts[names[name]] = n
        paths.add(path)
    need(len(counts) == len(names) and type(dataset['population']) is int and
         sum(counts.values()) == dataset['population'])
    return counts
