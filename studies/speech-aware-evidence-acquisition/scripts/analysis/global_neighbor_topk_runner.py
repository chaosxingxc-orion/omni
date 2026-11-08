#!/usr/bin/env python3
"""Rank pinned cosine matrices using externally prepared identities and source groups."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from speech_aware_evidence_acquisition.analysis.global_neighbor_topk import Config, aggregate


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_pinned(pin):
    path = Path(pin['path'])
    if digest(path) != pin['sha256']:
        raise ValueError(f'input SHA mismatch: {path}')
    return path


def run(config):
    members_path = read_pinned(config['members'])
    queries_path = read_pinned(config['queries'])
    members = json.loads(members_path.read_text())
    queries = json.loads(queries_path.read_text())
    ids = [q['sample_id'] for q in queries]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate query sample_id')
    routes = {}
    for name, pin in config['matrices'].items():
        matrix = np.load(read_pinned(pin), mmap_mode='r')
        if matrix.shape != (len(queries), len(members)):
            raise ValueError('matrix shape does not match queries and members')
        for distinct in config['source_distinct']:
            for k in config['k']:
                key = f"{name}/{'source_distinct' if distinct else 'raw'}/k{k}"
                routes[key] = [aggregate(matrix[i], members, Config(k=k, source_distinct=distinct),
                                         [query['source_group']])
                               for i, query in enumerate(queries)]
    return {'schema': 'global-neighbor-topk-rankings-v1', 'sample_ids': ids,
            'routes': routes, 'query_gold_used_for_ranking': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(config['output'])
    result = run(config)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'output': str(output), 'sha256': digest(output), 'queries': len(result['sample_ids'])}))


if __name__ == '__main__':
    main()
