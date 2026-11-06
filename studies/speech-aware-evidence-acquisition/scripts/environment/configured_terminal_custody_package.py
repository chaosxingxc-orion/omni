#!/usr/bin/env python3
"""Produce a validated terminal custody package from explicitly pinned evidence.

All paths are absolute paths as visible inside the fixed source image. Inputs are
read-only; output is a new directory on the same filesystem as its parent.
"""
# Host tools import pinned release modules; keep its signed file set immutable.
import sys
sys.dont_write_bytecode = True

import argparse
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

# This script may be mounted alone outside the release tree.  Its sibling
# helper and the package implementation belong to the immutable image.
_release_study = Path('/opt/saea-release/studies/speech-aware-evidence-acquisition')
for _path in (_release_study / 'src', _release_study / 'scripts' / 'environment'):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.append(str(_path))

from checkpoint_terminal_custody import _pinned, inventory_snapshot, validate_terminal_custody
from speech_aware_evidence_acquisition.contracts import ExecutionPlan
from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal
from speech_aware_evidence_acquisition.core.attempts import AttemptStore


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def strict(blob):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError('duplicate JSON key: ' + k)
            result[k] = v
        return result
    def constant(v):
        raise ValueError('invalid JSON constant: ' + v)
    return json.loads(blob, object_pairs_hook=pairs, parse_constant=constant)


def pin(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def scan(roots):
    inventory = []
    for root in sorted(roots):
        p = Path(root)
        if not p.is_absolute() or str(p) != str(p.resolve()) or not p.is_dir():
            raise ValueError('invalid or redirected inventory root: ' + root)
        files = {}
        def onerror(e):
            raise e
        for parent, dirs, names in os.walk(p, followlinks=False, onerror=onerror):
            for name in dirs:
                d = Path(parent) / name
                if d.is_symlink() or not d.is_dir():
                    raise ValueError('inventory directory symlink or non-directory')
            for name in names:
                f = Path(parent) / name
                if f.is_symlink() or not f.is_file():
                    raise ValueError('inventory non-regular file')
                files[f.relative_to(p).as_posix()] = hashlib.sha256(f.read_bytes()).hexdigest()
        inventory.append({'root': root, 'files': files})
    inventory_snapshot(inventory)
    return inventory


def write(path, blob):
    with path.open('xb') as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish(src, dst):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.renameat2(-100, os.fsencode(src), -100, os.fsencode(dst), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(dst))
    fsync_dir(dst.parent)


def _has_evidence_location(value, field=None):
    if field == 'root':
        return True
    if isinstance(value, dict):
        return set(value) == {'path', 'sha256'} or any(
            _has_evidence_location(child, key) for key, child in value.items())
    if isinstance(value, list):
        return any(_has_evidence_location(child) for child in value)
    return False


def relocate_evidence(value, mapping, field=None):
    """Relocate only pin leaves/archive roots; preserve digests and metadata.

    Mapping mirrors just the relocatable branches (no schema/startup root).
    Original evidence may be offline; new pin bytes must match original SHA.
    """
    if field == 'root':
        if not isinstance(mapping, str):
            raise ValueError('archive root mapping must be an absolute directory')
        root = Path(mapping)
        if not root.is_absolute() or root.resolve() != root or root.is_symlink() or not root.is_dir():
            raise ValueError('archive root mapping must be a real absolute directory')
        return mapping
    if isinstance(value, dict) and set(value) == {'path', 'sha256'}:
        _pinned(mapping)
        if Path(mapping['path']).resolve() != Path(mapping['path']):
            raise ValueError('bridge evidence path must not traverse symlinks')
        if mapping['sha256'] != value['sha256']:
            raise ValueError('bridge evidence pin differs')
        return copy.deepcopy(mapping)
    if isinstance(value, dict):
        keys = {key for key, child in value.items() if _has_evidence_location(child, key)}
        if not isinstance(mapping, dict) or set(mapping) != keys:
            raise ValueError('bridge mapping must cover exactly every evidence pin and archive root')
        return {key: relocate_evidence(child, mapping[key], key) if key in keys else copy.deepcopy(child)
                for key, child in value.items()}
    if isinstance(value, list):
        if not isinstance(mapping, list) or len(mapping) != len(value):
            raise ValueError('bridge list mapping shape differs')
        return [relocate_evidence(child, mapped) if _has_evidence_location(child)
                else copy.deepcopy(child) for child, mapped in zip(value, mapping)]
    raise ValueError('unrecognized evidence relocation')


def prepare(config):
    """Resolve a pinned source manifest and native bridge into held templates."""
    keys = {'schema', 'source_manifest', 'source_paths', 'bridge', 'bridge_paths',
            'compiler', 'admission', 'predecessor_decision_file', 'lock', 'output', 'package_output'}
    if set(config) - {'lineage_paths'} != keys or config['schema'] != 'saea-terminal-package-prepare-v1':
        raise ValueError('invalid prepare configuration')
    manifest = strict(_pinned(config['source_manifest']))
    bridge = strict(_pinned(config['bridge']))
    compiler = strict(_pinned(config['compiler']))
    if manifest.get('schema') != 'saea-experiment-assets-v1' or set(manifest) != {'schema', 'files'}:
        raise ValueError('invalid source manifest')
    if set(config['source_paths']) != set(manifest['files']):
        raise ValueError('source mapping must cover exactly every asset')
    for name, old in manifest['files'].items():
        new = config['source_paths'][name]
        _pinned(new)
        if old['sha256'] != new['sha256']:
            raise ValueError('source asset pin differs: ' + name)
        manifest['files'][name] = copy.deepcopy(new)
    lineage_manifest = None
    if 'lineage_paths' in config and 'lineage_assets_manifest' not in compiler:
        raise ValueError('lineage mapping requires compiler catalog pin')
    if 'lineage_assets_manifest' in compiler:
        import configured_checkpoint_successor as native
        lineage_original = native.read_pin(compiler['lineage_assets_manifest'])
        lineage_manifest = native.strict_json(lineage_original)
        native.require(isinstance(lineage_manifest, dict) and set(lineage_manifest) == {'schema', 'files'}
                       and lineage_manifest['schema'] == 'saea-experiment-assets-v1'
                       and isinstance(lineage_manifest['files'], dict), 'invalid lineage assets manifest')
        if 'lineage_paths' in config:
            paths = config['lineage_paths']
            native.require(isinstance(paths, dict) and set(paths) == set(lineage_manifest['files']),
                           'lineage mapping must cover exactly every catalog asset')
            for name, old in lineage_manifest['files'].items():
                native.logical(name)
                native.read_pin(paths[name])
                native.require(paths[name]['sha256'] == old['sha256'], 'lineage asset pin differs: ' + name)
            lineage_manifest['files'] = copy.deepcopy(paths)
    from speech_aware_evidence_acquisition.core.flight.checkpoint_compatibility import MULTI_SCHEMA
    if bridge.get('schema') == MULTI_SCHEMA:
        if not isinstance(config['bridge_paths'], dict) or set(config['bridge_paths']) != set(bridge['bridges']):
            raise ValueError('bridge mapping must cover every ancestor')
        bridges = [(item, config['bridge_paths'][key]) for key, item in bridge['bridges'].items()]
    else:
        bridges = [(bridge, config['bridge_paths'])]
    for item, paths in bridges:
        item['evidence'] = relocate_evidence(item['evidence'], paths)
        if any(name in item['evidence'] for name in ('producer_workspace', 'producer_assets')):
            from speech_aware_evidence_acquisition.core.flight.checkpoint_compatibility import ImageReleaseCompatibility
            ImageReleaseCompatibility.from_document(item).old_source()
        if item['predecessor_fragments'] != compiler['fragments']:
            raise ValueError('bridge and compiler fragments differ')
        item['successor_attempt_id'] = compiler['target']['attempt_id']
    _pinned(config['lock'])
    _pinned(config['predecessor_decision_file'])
    if config['admission']['predecessor_decision']['sha256'] != config['predecessor_decision_file']['sha256']:
        raise ValueError('predecessor decision pin differs')
    bridge['successor_attempt_id'] = compiler['target']['attempt_id']
    out = Path(config['output'])
    if not out.is_absolute() or not out.parent.is_dir() or out.parent.resolve() != out.parent or os.path.lexists(out):
        raise ValueError('fresh absolute output with existing real parent required')
    stage = Path(tempfile.mkdtemp(prefix='.' + out.name + '.', dir=out.parent))
    try:
        assets_bytes, bridge_bytes = encode(manifest), encode(bridge)
        write(stage / 'source-manifest.json', assets_bytes)
        write(stage / 'bridge.json', bridge_bytes)
        compiler['assets_manifest'] = {'path': str(out / 'source-manifest.json'), 'sha256': hashlib.sha256(assets_bytes).hexdigest()}
        if lineage_manifest is not None:
            lineage_bytes = encode(lineage_manifest)
            write(stage / 'lineage-source-manifest.json', lineage_original)
            write(stage / 'lineage-assets.json', lineage_bytes)
            lineage_pin = {'path': str(out / 'lineage-assets.json'),
                           'sha256': hashlib.sha256(lineage_bytes).hexdigest()}
            write(stage / 'lineage-relocation.json', encode({
                'schema': 'saea-lineage-relocation-v1', 'source_compiler': config['compiler'],
                'source_manifest': compiler['lineage_assets_manifest'],
                'source_manifest_copy': {'path': str(out / 'lineage-source-manifest.json'),
                                         'sha256': hashlib.sha256(lineage_original).hexdigest()},
                'relocated_manifest': lineage_pin, 'paths': lineage_manifest['files']}))
            compiler['lineage_assets_manifest'] = lineage_pin
            # Validate the consumed closure against staged bytes before publication.
            staged = dict(compiler, lineage_assets_manifest=dict(lineage_pin, path=str(stage / 'lineage-assets.json')))
            assets = {native.logical(name): native.read_pin(pin) for name, pin in manifest['files'].items()}
            native.complete_lineage_inputs(staged, assets, native.strict_json(assets[compiler['candidate']['relpath']]))
        compiler['admission'] = copy.deepcopy(config['admission'])
        compiler['terminal'] = None
        for name in ('candidate', 'plan'):
            item = compiler[name]
            if manifest['files'][item['relpath']]['sha256'] != item['sha256']:
                raise ValueError(name + ' mismatch')
        compiler_bytes = encode(compiler)
        write(stage / 'compiler-template.json', compiler_bytes)
        candidate = manifest['files'][compiler['candidate']['relpath']]
        plan = manifest['files'][compiler['plan']['relpath']]
        package = {'schema': 'saea-terminal-custody-package-config-v1', 'repo': compiler['repo'],
                   'data': compiler['data'], 'candidate': candidate, 'plan': plan, 'target': compiler['target'],
                   'resume_from': None, 'bridge': {'path': str(out / 'bridge.json'), 'sha256': hashlib.sha256(bridge_bytes).hexdigest()},
                   'ancestors': None, 'evidence': None, 'output': config['package_output'],
                   'compiler': {'template': {'path': str(out / 'compiler-template.json'), 'sha256': hashlib.sha256(compiler_bytes).hexdigest()},
                                'registration_relpath': config['admission']['registration_relpath'], 'lock': config['lock']}}
        # registration_relpath belongs to the package, not native admission.
        del compiler['admission']['registration_relpath']
        compiler_bytes = encode(compiler)
        (stage / 'compiler-template.json').unlink()
        write(stage / 'compiler-template.json', compiler_bytes)
        package['compiler']['template']['sha256'] = hashlib.sha256(compiler_bytes).hexdigest()
        write(stage / 'package-held.json', encode({'status': 'HELD_TERMINAL', 'config': package,
              'required_terminal_fields': ['resume_from', 'ancestors', 'evidence'], 'activation_authorized': False}))
        fsync_dir(stage)
        publish(stage, out)
        return {'output': str(out), 'held': pin(out / 'package-held.json')}
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def complete(config):
    if set(config) != {'schema', 'held', 'resume_from', 'ancestors', 'evidence'} or config['schema'] != 'saea-terminal-package-complete-v1':
        raise ValueError('invalid terminal completion configuration')
    held = strict(_pinned(config['held']))
    if held.get('status') != 'HELD_TERMINAL' or held.get('activation_authorized') is not False:
        raise ValueError('invalid held template')
    package = held['config']
    for name in ('resume_from', 'ancestors', 'evidence'):
        package[name] = config[name]
    return build(package)


def validate_compiler_template(cfg):
    """Check the native compiler's static schema and pinned source references."""
    import configured_checkpoint_successor as native
    from speech_aware_evidence_acquisition.rails.dispatch_slot_fill import parse_dispatch_slot_fill

    fields = {'schema', 'mode', 'repo', 'data', 'output', 'assets_manifest', 'candidate',
              'plan', 'fragments', 'target', 'dispatch', 'width', 'supply', 'terminal', 'admission'}
    native.require(isinstance(cfg, dict) and set(cfg) - {'lineage_assets_manifest', 'execution_mode'} == fields and
                   cfg['schema'] == 'saea-checkpoint-successor-compiler-v1' and
                   cfg['mode'] == 'finalize', 'exact compiler schema required')
    native.validate_execution_mode(cfg)
    repo, data, out = (Path(cfg[k]) for k in ('repo', 'data', 'output'))
    native.require(all(p.is_absolute() for p in (repo, data, out)) and repo.is_dir() and data.is_dir(), 'absolute existing roots required')
    native.require(out.parent.is_dir() and not out.parent.is_symlink() and not os.path.lexists(out), 'fresh output required')
    root = out.parent.resolve() / out.name
    native.require(not any(root.is_relative_to(p.resolve()) or p.resolve().is_relative_to(root)
                           for p in (repo, data)), 'output must be isolated from repo and data')
    fragments, target = cfg['fragments'], cfg['target']
    for value, keys in ((fragments, {'model', 'dataset', 'baseline', 'experiment'}),
                        (target, {'attempt_id', 'fit_run_id', 'fit_attempt_id'})):
        native.require(isinstance(value, dict) and set(value) == keys and
                       all(isinstance(v, str) and native.NAME.fullmatch(v) for v in value.values()),
                       'invalid compiler names')
    native.require(type(cfg['width']) is int and 1 <= cfg['width'] <= 128, 'bounded width required')
    parse_dispatch_slot_fill({'dispatch': cfg['dispatch']})
    manifest = native.strict_json(native.read_pin(cfg['assets_manifest']))
    native.require(isinstance(manifest, dict) and set(manifest) == {'schema', 'files'} and
                   manifest['schema'] == 'saea-experiment-assets-v1' and
                   isinstance(manifest['files'], dict), 'invalid assets manifest')
    assets = {native.logical(name): native.read_pin(pin) for name, pin in manifest['files'].items()}
    originals = []
    for name in ('candidate', 'plan'):
        item = cfg[name]
        native.require(isinstance(item, dict) and set(item) == {'relpath', 'sha256'}, 'logical source pin required')
        path = native.logical(item['relpath'])
        native.require(path in assets and native.digest(assets[path]) == item['sha256'], 'original source asset differs')
        originals.append(native.strict_json(assets[path]))
    candidate, plan = originals
    if 'lineage_assets_manifest' in cfg:
        native.complete_lineage_inputs(cfg, assets, candidate)
    for kind, name in fragments.items():
        native.require('configs/' + kind + '/' + name + '.json' in assets, 'fragment absent from pinned assets')
    native.require(candidate['experiment'] == fragments['experiment'] and candidate['baseline'] == fragments['baseline'] and
                   candidate['model_fragment'] == fragments['model'] and candidate['dataset']['relpath'] ==
                   'configs/dataset/' + fragments['dataset'] + '.json', 'registration fragment differs')
    source = plan['execution_plan']
    native.require(all(source[k] == candidate[k] for k in ('run_id', 'attempt_id', 'protocol_hash')) and
                   plan['composed_config_hash'] == candidate['composed_config_hash'], 'registered plan identity differs')
    native.require(all(target[k] != candidate[k] for k in target), 'successor identities must be fresh')
    supply = cfg['supply']
    native.require(isinstance(supply, dict) and set(supply) == {'old', 'new'}, 'both supply pins required')
    values = []
    for pin in supply.values():
        native.require(isinstance(pin, dict) and set(pin) == {'relpath', 'sha256'} and
                       isinstance(pin['relpath'], str) and not Path(pin['relpath']).is_absolute() and
                       '..' not in Path(pin['relpath']).parts, 'invalid supply pin')
        root = Path(cfg['data']).resolve()
        path = (root / pin['relpath']).resolve()
        native.require(path.is_relative_to(root) and path.is_file() and native.digest(path.read_bytes()) == pin['sha256'], 'supply pin differs')
        values.append(native.strict_json(path.read_bytes())['output'])
    native.require(values[0] == values[1] and native.strict_json(assets['configs/baseline/' + fragments['baseline'] + '.json'])['prepared_phase_supply'] == supply['old'], 'registered supply differs')
    admission = cfg['admission']
    native.require(isinstance(admission, dict) and set(admission) == {'workspace', 'decision_relpath', 'predecessor_decision'}, 'exact external admission inputs required')
    decision = admission['predecessor_decision']
    native.require(isinstance(decision, dict) and set(decision) == {'relpath', 'sha256'} and
                   decision['relpath'] == candidate['decision_relpath'] and isinstance(decision['sha256'], str) and
                   native.HEX.fullmatch(decision['sha256']), 'predecessor decision differs')
    native.require(isinstance(admission['workspace'], str) and Path(admission['workspace']).is_absolute() and
                   not os.path.lexists(admission['workspace']) and
                   all(not Path(admission['workspace']).resolve().is_relative_to(p.resolve()) and
                       not p.resolve().is_relative_to(Path(admission['workspace']).resolve()) for p in (repo, data, out)) and
                   isinstance(admission['decision_relpath'], str) and admission['decision_relpath'].startswith('coordination/') and
                   '..' not in Path(admission['decision_relpath']).parts and admission['decision_relpath'] != decision['relpath'], 'invalid admission paths')


def build(config):
    allowed = {'schema', 'repo', 'data', 'candidate', 'plan', 'target', 'resume_from',
               'bridge', 'ancestors', 'evidence', 'output', 'compiler'}
    if set(config) != allowed or config['schema'] != 'saea-terminal-custody-package-config-v1':
        raise ValueError('invalid package config fields or schema')
    output = Path(config['output'])
    if not output.is_absolute() or str(output.parent) != str(output.parent.resolve()) or not output.parent.is_dir():
        raise ValueError('output parent must be an existing absolute non-symlink directory')
    candidate = strict(_pinned(config['candidate']))
    plan = strict(_pinned(config['plan']))
    bridge = strict(_pinned(config['bridge']))
    compiler = strict(_pinned(config['compiler']['template']))
    if set(config['compiler']) != {'template', 'registration_relpath', 'lock'}:
        raise ValueError('invalid compiler binding')
    evidence = config['evidence']
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('evidence pins required')
    seen = set()
    for p in evidence:
        _pinned(p)
        if p['path'] in seen:
            raise ValueError('duplicate evidence path')
        seen.add(p['path'])
    roots = set()
    ancestors = config['ancestors']
    if not isinstance(ancestors, list) or not ancestors:
        raise ValueError('ancestors required')
    ids = set()
    for a in ancestors:
        if a['attempt_id'] in ids:
            raise ValueError('duplicate ancestor')
        ids.add(a['attempt_id'])
        raw = strict(_pinned(a['plan']))
        data = dict(raw['execution_plan'])
        if 'owner_allowance' in raw:
            if 'owner_allowance' in data and data['owner_allowance'] != raw['owner_allowance']:
                raise ValueError('plan allowance differs')
            data['owner_allowance'] = raw['owner_allowance']
        source = ExecutionPlan(**data).validate(source_root=config['repo'])
        journal = ContactJournal(Path(a['generation_root']), source, 'raw-speech-agent-native-tools', source_root=config['repo'])
        attempt = AttemptStore(Path(a['generation_root']) / 'attempts').handle(source.run_id, source.attempt_id).path
        roots.update(map(str, (journal.root, attempt.parent, Path(a['checkpoint']['path']).parent, Path(a['prepared_root']))))
    if any(output == Path(r) or Path(r) in output.parents for r in roots):
        raise ValueError('output must be outside custody roots')
    before = scan(roots)
    index = {'schema': 'saea-checkpoint-terminal-custody-v1', 'ancestors': ancestors, 'inventory': before}
    if compiler.get('mode') != 'finalize' or compiler.get('repo') != config['repo'] or compiler.get('data') != config['data'] or compiler.get('target') != config['target'] or compiler.get('candidate', {}).get('sha256') != config['candidate']['sha256'] or compiler.get('plan', {}).get('sha256') != config['plan']['sha256']:
        raise ValueError('compiler template identity differs')
    validate_compiler_template(compiler)
    import configured_checkpoint_successor as native
    native.logical(config['compiler']['registration_relpath'])
    manifest = native.strict_json(native.read_pin(compiler['assets_manifest']))
    native.require(config['compiler']['registration_relpath'] not in manifest['files'],
                   'predecessor registration path already exists')
    _pinned(config['compiler']['lock'])
    stage = Path(tempfile.mkdtemp(prefix='.' + output.name + '.staging-', dir=output.parent))
    try:
        index_bytes = encode(index)
        write(stage / 'index.json', index_bytes)
        final_index = {'path': str(output / 'index.json'), 'sha256': hashlib.sha256(index_bytes).hexdigest()}
        terminal = {'evidence': [final_index] + copy.deepcopy(evidence), 'resume_from': config['resume_from'],
                    'bridge': bridge, 'registration_relpath': config['compiler']['registration_relpath'],
                    'lock': config['compiler']['lock']}
        audit_terminal = copy.deepcopy(terminal)
        audit_terminal['evidence'][0]['path'] = str(stage / 'index.json')
        result = validate_terminal_custody(cfg={'repo': config['repo'], 'target': config['target'], 'terminal': audit_terminal},
                                           candidate=candidate, plan=plan, data=Path(config['data']))
        if scan(roots) != before:
            raise ValueError('inventory changed during package generation')
        for p in evidence + [config['candidate'], config['plan'], config['bridge'], config['compiler']['template']]:
            _pinned(p)
        validate_compiler_template(compiler)  # Recheck consumed pins after the custody audit.
        compiler['terminal'] = terminal
        write(stage / 'evidence-list.json', encode(terminal['evidence']))
        write(stage / 'terminal.json', encode(terminal))
        write(stage / 'validation.json', encode(result))
        write(stage / 'compiler-finalize.json', encode(compiler))
        write(stage / 'config.json', encode(config))
        fsync_dir(stage)
        try:
            publish(stage, output)
        except FileExistsError:
            if output.is_symlink() or not output.is_dir():
                raise ValueError('conflicting output')
            expected = {p.name: p.read_bytes() for p in stage.iterdir()}
            actual = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file() and not p.is_symlink()}
            if len(list(output.iterdir())) != len(actual) or actual != expected:
                raise ValueError('conflicting existing package')
            fsync_dir(output.parent)
        return {'output': str(output), 'terminal': pin(output / 'terminal.json'),
                'compiler': pin(output / 'compiler-finalize.json')}
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = strict(Path(args.config).read_bytes())
    action = {'saea-terminal-package-prepare-v1': prepare,
              'saea-terminal-package-complete-v1': complete}.get(config.get('schema'), build)
    print(json.dumps(action(config), sort_keys=True))


if __name__ == '__main__':
    main()
