"""Offline, source-bound slot-contract publication for PSD v2 workloads.

Compiler inputs include recorded external response custody; this module never
contacts a model. Each publication occupies a fresh directory under data_root.
"""
import argparse
import copy
import json
from pathlib import Path

from external_slot_candidate import read, checked, verify_response, compile_candidate, authored_release
from tfrl_hardset_a0_workload import confined, sha
from tfrl_hardset_a0_registration import pinned
from speech_aware_evidence_acquisition.rails.knowledge import KnowledgeSnapshot

SCHEMA = 'saea-psd-slot-knowledge-v1'


def reconstruct(compiler):
    cfg = read(checked(compiler))
    if set(cfg) != {'schema','original','base','contract','dynamic','teacher','candidate_index','forbidden_sample_ids','output'} or cfg['schema'] != 'saea-external-slot-candidate-config-v1':
        raise ValueError('exact external compiler configuration required')
    original, base, contract, dynamic, forbidden = (read(checked(cfg[k])) for k in ('original','base','contract','dynamic','forbidden_sample_ids'))
    proposals, custody = verify_response(cfg['teacher'], dynamic)
    index = cfg['candidate_index']
    if type(index) is not int or index < 0 or index >= len(proposals['candidates']):
        raise ValueError('explicit candidate index required')
    candidate = compile_candidate(original, base, contract, dynamic, proposals['candidates'][index], custody, forbidden)
    authored, snapshot = authored_release(candidate)
    return candidate, authored, snapshot


def apply(base, data, descriptor, consumer_ids):
    if set(descriptor) != {'schema','compiler','release','snapshot'} or descriptor['schema'] != SCHEMA:
        raise ValueError('exact PSD slot knowledge descriptor required')
    candidate, authored, rebuilt = reconstruct(descriptor['compiler'])
    cfg = read(checked(descriptor['compiler']))
    dynamic = read(checked(cfg['dynamic']))
    identity = lambda value: ':'.join(value.split(':')[:3]) if value.startswith('slurp:') else value
    evidence = {identity(item) for group in dynamic['edit_contract']['evidence_groups'].values() for item in group}
    if evidence & {identity(item) for item in consumer_ids}:
        raise ValueError('construction evidence overlaps workload consumers')
    store = base['knowledge_stores']['slot-contracts']
    if store['snapshot']['sha256'] != candidate['original_snapshot_sha256']:
        raise ValueError('candidate parent differs from workload slot snapshot')
    release = pinned(data, descriptor['release'])
    snapshot_path = confined(data, descriptor['snapshot']['relpath'])
    snapshot = KnowledgeSnapshot.from_frozen_json(read(snapshot_path))
    if snapshot.frozen_json() != rebuilt or snapshot.content_hash() != descriptor['snapshot']['sha256']:
        raise ValueError('published snapshot differs from reconstructed candidate')
    expected = {'schema':'saea-knowledge-store-release-v1', 'store_id':'slot-contracts',
                'store_role':'slot-contracts', 'snapshot_sha256':snapshot.content_hash(),
                'source_split_identity_hash':snapshot.source_split_identity_hash,
                'builder_policy_id':snapshot.builder_policy_id, 'source':release['source']}
    if release != expected or pinned(data, release['source']) != authored:
        raise ValueError('published release differs from authored reconstruction')
    if store['store_role'] != 'slot-contracts':
        raise ValueError('unexpected store role')
    out = copy.deepcopy(base)
    out['knowledge_stores']['slot-contracts']['release'] = copy.deepcopy(descriptor['release'])
    out['knowledge_stores']['slot-contracts']['snapshot'] = copy.deepcopy(descriptor['snapshot'])
    out.pop('prepared_phase_supply', None)
    return out


def publish(data, compiler, output):
    candidate, authored, snapshot = reconstruct(compiler)
    root = confined(data, output)
    root.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        path = root/name
        with path.open('x', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        return {'relpath':path.relative_to(Path(data).resolve()).as_posix(), 'sha256':sha(path)}
    source = write('authored.json', authored)
    snap_file = write('snapshot.json', snapshot)
    snap = {'relpath':snap_file['relpath'], 'sha256':snapshot['snapshot_sha256']}
    release = write('release.json', {'schema':'saea-knowledge-store-release-v1',
        'store_id':'slot-contracts','store_role':'slot-contracts',
        'snapshot_sha256':snapshot['snapshot_sha256'],
        'source_split_identity_hash':snapshot['source_split_identity_hash'],
        'builder_policy_id':snapshot['builder_policy_id'],'source':source})
    descriptor = {'schema':SCHEMA, 'compiler':copy.deepcopy(compiler), 'release':release, 'snapshot':snap}
    write('descriptor.json', descriptor)
    return descriptor, {'original_snapshot_sha256':candidate['original_snapshot_sha256'],
                        'snapshot_sha256':snapshot['snapshot_sha256'],
                        'changed_value_ids':candidate['changed_value_ids'],
                        'isolation':candidate['isolation'], 'new_model_calls':0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    args = p.parse_args()
    cfg = read(args.config)
    if set(cfg) != {'schema','data_root','compiler','output'} or cfg['schema'] != 'saea-psd-slot-publication-config-v1':
        raise ValueError('exact publication configuration required')
    descriptor, audit = publish(Path(cfg['data_root']), cfg['compiler'], cfg['output'])
    print(json.dumps({'descriptor':descriptor, 'audit':audit}, ensure_ascii=False))

if __name__ == '__main__':
    main()
