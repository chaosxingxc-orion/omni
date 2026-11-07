"""Focused source-bound publication checks; real inputs are explicitly supplied."""
import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from intent_snapshot_adapter import publish, validate_published


@pytest.fixture
def inputs(tmp_path):
    root = os.environ.get('INTENT_RELEASE_TEST_INPUTS')
    if not root:
        pytest.skip('real pinned parent inputs not configured')
    cfg = json.loads((Path(root) / 'config-2.json').read_bytes())
    def change(name, edit):
        value = json.loads(Path(cfg[name]['path']).read_bytes())
        edit(value)
        path = tmp_path / (name + '.json')
        path.write_text(json.dumps(value, ensure_ascii=False))
        cfg[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return cfg, change


def test_duplicate_publication_never_overwrites(inputs):
    cfg, _ = inputs
    before = (Path(cfg['data_root']) / cfg['output'] / 'release.json').read_bytes()
    with pytest.raises(FileExistsError):
        publish(cfg)
    assert (Path(cfg['data_root']) / cfg['output'] / 'release.json').read_bytes() == before


def test_new_content_is_rejected(inputs):
    cfg, change = inputs
    change('candidate', lambda c: c['raw_candidate']['memory']['card_operations'][0].__setitem__('new', 'invented'))
    with pytest.raises(ValueError):
        publish(cfg)


def test_wrong_card_source_is_rejected(inputs):
    cfg, change = inputs
    def corrupt(c):
        next(iter(c['locked_intent_card_bodies'].values()))['source_uri'] = 'wrong'
    change('context', corrupt)
    with pytest.raises(ValueError):
        publish(cfg)


def test_unchanged_card_tampering_is_rejected(inputs):
    cfg, change = inputs
    target = json.loads(Path(cfg['candidate']['path']).read_bytes())['artifact']['patches'][0]['target_id']
    def corrupt(c):
        key = next(k for k in c['locked_intent_card_bodies'] if k != target)
        c['locked_intent_card_bodies'][key]['content'] += ' changed'
    change('context', corrupt)
    with pytest.raises(ValueError):
        publish(cfg)


def test_existing_publication_is_source_bound_and_read_only(inputs, tmp_path):
    cfg, change = inputs
    root = Path(cfg['data_root']) / cfg['output']
    def pin(name):
        path = root / name
        return {'relpath': path.relative_to(cfg['data_root']).as_posix(),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    parent = json.loads(Path(cfg['baseline']['path']).read_bytes())
    before = {name: (root / name).read_bytes() for name in ('snapshot.json', 'release.json', 'patch.json', 'audit.json')}
    updated = validate_published(cfg, parent, pin('patch.json'), pin('audit.json'))
    expected = copy.deepcopy(parent)
    expected['knowledge_stores']['intent-boundaries']['snapshot'] = updated['knowledge_stores']['intent-boundaries']['snapshot']
    expected['knowledge_stores']['intent-boundaries']['release'] = updated['knowledge_stores']['intent-boundaries']['release']
    expected.pop('prepared_phase_supply', None)
    assert updated == expected
    assert before == {name: (root / name).read_bytes() for name in before}
    wrong = copy.deepcopy(parent)
    wrong['knowledge_stores']['intent-boundaries']['snapshot']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='baseline differs'):
        validate_published(cfg, wrong, pin('patch.json'), pin('audit.json'))
    fake = json.loads(before['patch.json'])
    fake[2]['path'] = '/knowledge_stores/slot-contracts/snapshot'
    path = tmp_path / 'patch.json'; path.write_text(json.dumps(fake))
    # The data-root confinement is itself enforced for an outside asset.
    with pytest.raises(ValueError):
        validate_published(cfg, parent, {'relpath': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}, pin('audit.json'))
    change('candidate', lambda c: c['raw_candidate']['memory']['card_operations'][0].__setitem__('new', 'invented'))
    with pytest.raises(ValueError):
        validate_published(cfg, parent, pin('patch.json'), pin('audit.json'))
