"""Pinned TRAIN-only diagnostic material for the native ACE learner.

No inference, tokenizer shortcuts, solver-input injection or publication lives
here. Runtime reloads the pins and reconstructs the material before native FIT.
"""
import copy
import hashlib
import json
from pathlib import Path
import re

from .grounded_synthesis_fusion import project_group_evidence
from .tfgrpo_learning import LearningError, sha

SCHEMA = 'saea-construction-development-spec-v1'
MATERIAL = 'saea-construction-development-material-v1'
BOUNDARY = 'TRAIN gold is diagnostic only; never evaluation Solver input or runtime trigger.'


def require(condition, message):
    if not condition:
        raise LearningError(message)


def pinned_bytes(pin, *, repo_root, data_root):
    require(isinstance(pin, dict) and set(pin) == {'root', 'relpath', 'sha256'}
            and pin['root'] in ('repository', 'data')
            and isinstance(pin['relpath'], str) and not Path(pin['relpath']).is_absolute()
            and '..' not in Path(pin['relpath']).parts, 'exact confined development pin required')
    root = Path(repo_root if pin['root'] == 'repository' else data_root).resolve()
    path = (root / pin['relpath']).resolve()
    require(path.is_relative_to(root), 'development pin escaped root')
    blob = path.read_bytes()
    require(hashlib.sha256(blob).hexdigest() == pin['sha256'], 'development source pin changed')
    return blob


def _entities(row):
    return sorted((e['type'], ' '.join(row['tokens'][i]['surface'] for i in e['span']).lower())
                  for e in row['entities'])


def load_material(spec, source, *, repo_root, data_root):
    require(isinstance(spec, dict) and set(spec) == {'schema', 'diagnostic', 'original_train', 'task_template'}
            and spec['schema'] == SCHEMA, 'exact construction development spec required')
    blobs = {key: pinned_bytes(spec[key], repo_root=repo_root, data_root=data_root)
             for key in ('diagnostic', 'original_train', 'task_template')}
    diagnostic = json.loads(blobs['diagnostic'])
    require(diagnostic.get('schema') == 'construction-diagnostic-evidence-private-v1'
            and diagnostic.get('original_train_sha256') == spec['original_train']['sha256'],
            'diagnostic original TRAIN pin differs')
    original = {}
    for line in blobs['original_train'].splitlines():
        row = json.loads(line)
        key = str(row['slurp_id'])
        require(key not in original, 'duplicate original TRAIN utterance')
        original[key] = row
    task = blobs['task_template'].decode('utf-8')
    require(bool(task.strip()) and len(task) <= 8192, 'bounded UTF-8 development task required')
    groups = source['batch']['groups']; rows = diagnostic.get('groups')
    require(isinstance(rows, list) and len(rows) == len(groups), 'complete diagnostic TRAIN roster required')
    projected = []
    for group, row in zip(groups, rows):
        qid = group['query_id']; match = re.fullmatch(r'slurp:train:([0-9]+):(.+)', qid)
        require(match is not None and row.get('query_id') == qid
                and row.get('original_group_sha256') == sha(group), 'diagnostic group identity differs')
        require(row.get('original_evidence') == project_group_evidence(group),
                'diagnostic original evidence differs from native source')
        train = original.get(str(int(match.group(1))))
        require(isinstance(train, dict) and match.group(2) + '.flac' in
                {r['file'] for r in train['recordings']}, 'TRAIN utterance/audio identity differs')
        reference = json.loads(group['answer']) if isinstance(group['answer'], str) else group['answer']
        require(train['scenario'] == reference['scenario'] and train['action'] == reference['action']
                and _entities(train) == sorted((e['type'], e['filler'].lower()) for e in reference['entities']),
                'original TRAIN annotation differs from native feedback')
        diagnosis = row.get('diagnostic_train_only', {})
        require(diagnosis.get('slurp_id') == train['slurp_id']
                and diagnosis.get('reference_transcript') == train['sentence']
                and diagnosis.get('reference_annotation') == train['sentence_annotation'],
                'diagnostic transcript or annotation changed')
        projected.append({'query_id': qid, 'group_sha256': sha(group),
            'diagnostic_train_only': {'slurp_id': train['slurp_id'],
                'reference_transcript': train['sentence'],
                'reference_annotation': train['sentence_annotation'], 'boundary': BOUNDARY},
            'evidence': project_group_evidence(group, evidence_view='full-query-v1')})
    body = {'schema': MATERIAL, 'source_batch_sha256': sha(source['batch']),
        'spec': copy.deepcopy(spec), 'task_template': task, 'boundary': BOUNDARY,
        'solver_input_permitted': False, 'audio_inspected': False, 'groups': projected}
    return {**body, 'sha256': sha(body)}


def validate_material(material, source):
    require(isinstance(material, dict) and material.get('schema') == MATERIAL
            and material.get('sha256') == sha({k: v for k, v in material.items() if k != 'sha256'})
            and material.get('source_batch_sha256') == sha(source['batch'])
            and material.get('boundary') == BOUNDARY and material.get('solver_input_permitted') is False
            and material.get('audio_inspected') is False, 'construction material identity or boundary differs')
    groups = source['batch']['groups']; rows = material.get('groups')
    require(isinstance(rows, list) and len(rows) == len(groups), 'complete development roster required')
    for group, row in zip(groups, rows):
        require(row['query_id'] == group['query_id'] and row['query_id'].startswith('slurp:train:')
                and row['group_sha256'] == sha(group)
                and row['evidence'] == project_group_evidence(group, evidence_view='full-query-v1')
                and row['diagnostic_train_only']['boundary'] == BOUNDARY,
                'development evidence is not derived from original source')
    return material
