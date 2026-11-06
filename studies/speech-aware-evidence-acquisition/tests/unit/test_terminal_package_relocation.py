"""Prepare moves pinned evidence by explicit nested mappings, without repinning."""
import copy
import json
import shutil
import sys
from pathlib import Path

import pytest

from test_producer_asset_bridge import layered, producer, write
from speech_aware_evidence_acquisition.core.flight.checkpoint_compatibility import (
    CheckpointCompatibility, evidence_pins,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/environment'))
from configured_terminal_custody_package import prepare, relocate_evidence


def relocation(value, destination):
    if isinstance(value, dict) and set(value) == {'path', 'sha256'}:
        target = destination / Path(value['path']).name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(value['path'], target)
        return dict(value, path=str(target))
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            if key == 'root':
                target = destination / 'root'; shutil.copytree(child, target)
                result[key] = str(target)
            elif isinstance(child, dict):
                result[key] = relocation(child, destination / key)
        return result
    raise AssertionError('unexpected fixture shape')


def prepare_config(tmp_path, bridge, mappings):
    asset = write(tmp_path / 'candidate.json', {'candidate': 'fixture'})
    plan = write(tmp_path / 'plan.json', {'plan': 'fixture'})
    manifest = write(tmp_path / 'input-assets.json', {'schema': 'saea-experiment-assets-v1',
        'files': {'docs/candidate.json': asset, 'docs/plan.json': plan}})
    compiler = write(tmp_path / 'compiler.json', {'fragments': bridge['predecessor_fragments'],
        'target': {'attempt_id': 'successor'}, 'repo': '/sealed/repo', 'data': '/data',
        'candidate': {'relpath': 'docs/candidate.json', 'sha256': asset['sha256']},
        'plan': {'relpath': 'docs/plan.json', 'sha256': plan['sha256']}})
    decision = write(tmp_path / 'decision.json', {'decision': 'preserved'})
    return {'schema': 'saea-terminal-package-prepare-v1', 'source_manifest': manifest,
        'source_paths': {'docs/candidate.json': asset, 'docs/plan.json': plan},
        'bridge': write(tmp_path / 'bridge-input.json', bridge), 'bridge_paths': mappings,
        'compiler': compiler, 'admission': {'predecessor_decision': {'sha256': decision['sha256']},
            'registration_relpath': 'docs/registration.json'},
        'predecessor_decision_file': decision, 'lock': write(tmp_path / 'lock.json', {'preserved': True}),
        'output': str(tmp_path / 'prepared'), 'package_output': str(tmp_path / 'package')}


@pytest.mark.parametrize('mode', ['git', 'workspace', 'layers'])
def test_prepare_nested_and_flat_mapping(layered, tmp_path, mode):
    doc, _, _, _ = layered
    if mode != 'layers': del doc['evidence']['producer_assets']
    if mode == 'git': del doc['evidence']['producer_workspace']
    original = copy.deepcopy(doc)
    paths = relocation(doc['evidence'], tmp_path / 'relocated')
    config = prepare_config(tmp_path, doc, paths)
    result = prepare(config)
    moved = json.loads((Path(result['output']) / 'bridge.json').read_bytes())
    assert sorted(p['sha256'] for p in evidence_pins(moved['evidence'])) == sorted(
        p['sha256'] for p in evidence_pins(original['evidence']))
    assert moved['evidence']['startup_data_root'] == original['evidence']['startup_data_root']
    assert doc == original
    assert moved['successor_attempt_id'] == 'successor'
    held = json.loads((Path(result['output']) / 'package-held.json').read_bytes())
    assert held['status'] == 'HELD_TERMINAL' and held['activation_authorized'] is False
    for name in ('producer_assets', 'producer_workspace'):
        if name in moved['evidence']:
            assert moved['evidence'][name]['root'] != original['evidence'][name]['root']
    old = CheckpointCompatibility.from_document(original).old_source()
    # Remove the old roots and pins: the prepared bridge must be self-sufficient.
    for name in ('producer_assets', 'producer_workspace'):
        if name in original['evidence']: shutil.rmtree(original['evidence'][name]['root'])
    for pin in evidence_pins(original['evidence']):
        Path(pin['path']).unlink(missing_ok=True)
    assert CheckpointCompatibility.from_document(moved).old_source() == old


@pytest.mark.parametrize('fault', ['pin-sha', 'pin-bytes', 'repinned-bytes', 'root-bytes', 'root-symlink',
                                 'root-traversal', 'missing-pin', 'extra-field', 'missing-root'])
def test_prepare_rejects_bad_nested_mapping(layered, tmp_path, fault):
    doc, _, _, _ = layered
    paths = relocation(doc['evidence'], tmp_path / 'relocated')
    external = paths['producer_assets']
    if fault == 'pin-sha': external['manifest']['sha256'] = '0' * 64
    elif fault == 'pin-bytes': Path(external['manifest']['path']).write_bytes(b'changed')
    elif fault == 'repinned-bytes':
        external['manifest'] = write(Path(external['manifest']['path']), {'new': 'bytes with new SHA'})
    elif fault == 'root-bytes':
        receipt = Path(external['root']) / 'producer/assets/docs/receipts/runtime.json'
        receipt.write_bytes(b'changed')
    elif fault == 'root-symlink':
        alias = tmp_path / 'alias'; alias.symlink_to(external['root']); external['root'] = str(alias)
    elif fault == 'root-traversal': external['root'] += '/../root'
    elif fault == 'missing-pin': del external['launch']
    elif fault == 'missing-root': del external['root']
    else: external['schema'] = 'relabel-is-not-relocation'
    config = prepare_config(tmp_path, doc, paths)
    with pytest.raises(ValueError): prepare(config)
    assert not Path(config['output']).exists()


def test_relocation_accepts_nested_lists_without_changing_metadata(tmp_path):
    pin = write(tmp_path / 'pin.json', {'frozen': True})
    value = {'schema': 'future', 'groups': [{'evidence': pin, 'label': 'kept'}], 'startup_data_root': '/old'}
    target = tmp_path / 'copy.json'; shutil.copyfile(pin['path'], target)
    mapped = {'groups': [{'evidence': dict(pin, path=str(target))}]}
    result = relocate_evidence(value, mapped)
    assert result == dict(value, groups=[{'evidence': dict(pin, path=str(target)), 'label': 'kept'}])
