#!/usr/bin/env python3
"""Compile held checkpoint successors after an independent terminal custody audit."""
import argparse
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / 'src'))
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint_terminal_custody import validate_terminal_custody

HEX = re.compile(r'[0-9a-f]{64}\Z')
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*\Z')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_pin(pin):
    require(isinstance(pin, dict) and set(pin) == {'path', 'sha256'}, 'exact file pin required')
    require(isinstance(pin['path'], str) and Path(pin['path']).is_absolute() and
            isinstance(pin['sha256'], str) and HEX.fullmatch(pin['sha256']), 'invalid file pin')
    path = Path(pin['path'])
    require(not path.is_symlink() and path.is_file(), 'pinned file missing or symlink')
    blob = path.read_bytes()
    require(digest(blob) == pin['sha256'], 'pinned file hash changed')
    return blob


def encode(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def strict_json(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result

    def constant(value):
        raise ValueError('non-finite JSON number: ' + value)

    return json.loads(blob, object_pairs_hook=pairs, parse_constant=constant)


def logical(value):
    require(isinstance(value, str) and value.startswith(('configs/', 'docs/')) and
            not Path(value).is_absolute() and '..' not in Path(value).parts and
            Path(value).suffix in ('.json', '.jsonl', '.md', '.txt'), 'invalid logical asset path')
    return value


def _publish_directory(source, destination):
    # Linux renameat2 provides atomic no-replace publication even if the
    # destination appears between validation and publication.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))
    fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def asset_environment(manifest, *, width=None, profile=None):
    keys = ('SAEA_EXPERIMENT_MANIFEST', 'SAEA_EXPERIMENT_MANIFEST_SHA256',
            'SAEA_EXPERIMENT_WORKSPACE', 'SAEA_EXPERIMENT_WORKSPACE_MANIFEST',
            'SAEA_EXPERIMENT_WORKSPACE_SHA256', 'SAEA_ADMITTED_LEVEL', 'SAEA_DEPLOYMENT_PROFILE')
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        os.environ['SAEA_EXPERIMENT_MANIFEST'] = str(manifest)
        os.environ['SAEA_EXPERIMENT_MANIFEST_SHA256'] = digest(manifest.read_bytes())
        if width is not None:
            require(type(width) is int and 1 <= width <= 128, 'invalid admitted width')
            os.environ['SAEA_ADMITTED_LEVEL'] = str(width)
        if profile is not None:
            require(isinstance(profile, str) and profile, 'invalid deployment profile')
            os.environ['SAEA_DEPLOYMENT_PROFILE'] = profile
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _successor_plan_documents(*, candidate, plan, baseline, experiment, target,
                              supply, repo):
    """Build the independent plans consumed by external admission.

    The predecessor generation plan remains byte-for-byte in the asset set. The
    two documents below are new registered plans with the same scientific and
    budget fields, fresh identities, and one shared recomputed protocol hash.
    """
    from dataclasses import asdict, fields, replace
    from speech_aware_evidence_acquisition.contracts import ExecutionPlan
    source = copy.deepcopy(plan['execution_plan'])
    require('owner_allowance' not in source or 'owner_allowance' not in plan,
            'duplicate owner allowance')
    old_allowance = plan.get('owner_allowance', source.pop('owner_allowance', None))
    allowed = {field.name for field in fields(ExecutionPlan)}
    require(not set(source) - allowed, 'unknown original execution plan fields')
    original = ExecutionPlan(**source, owner_allowance=old_allowance).validate(source_root=repo)
    expected = experiment['execution_plan']
    for key in ('execution_profile', 'split_role', 'planned_model_calls',
                'planned_gpu_hours', 'planned_speech_audio_seconds'):
        require(getattr(original, key) == expected[key], 'original budget differs: ' + key)
    require(original.split_identity_hash == candidate['split_identity_hash'], 'original split differs')
    generation = replace(original, attempt_id=target['attempt_id'],
                         protocol_hash=candidate['protocol_hash']).validate(source_root=repo)
    fit = replace(generation, run_id=target['fit_run_id'], attempt_id=target['fit_attempt_id'],
                  planned_model_calls=baseline['native_contact_fit']['contact_cap']).validate(source_root=repo)
    protocol_doc = {
        'schema': 'saea-successor-protocol-v1', 'cell': candidate['cell'],
        'experiment': candidate['experiment'], 'run_id': candidate['run_id'],
        'attempt_id': candidate['attempt_id'], 'fit_run_id': target['fit_run_id'],
        'fit_attempt_id': target['fit_attempt_id'],
        'composed_config_hash': candidate['composed_config_hash'],
        'protocol_hash': candidate['protocol_hash'],
        'split_identity_hash': candidate['split_identity_hash'],
        'scientific_baseline_sha256': candidate['scientific_baseline_sha256'],
        'prepared_phase_supply': copy.deepcopy(supply), 'status': 'held',
    }
    protocol_path = 'docs/readiness/' + candidate['cell'] + '/protocol-successor-v1.json'
    protocol_pin = {'relpath': protocol_path, 'sha256': digest(encode(protocol_doc))}
    common = {
        'cell': candidate['cell'], 'experiment': candidate['experiment'],
        'composed_config_hash': candidate['composed_config_hash'],
        'expected_session_id': candidate['service_session'],
        'prepared_phase_supply': copy.deepcopy(supply),
        'protocol_document': copy.deepcopy(protocol_pin),
        'status': 'HELD_BEFORE_EXTERNAL_ADMISSION',
    }
    generation_doc = {'schema': 'saea-registered-generation-plan-v1', 'plan_kind': 'generation',
                      **common, 'execution_plan': asdict(generation)}
    fit_doc = {'schema': 'saea-registered-fit-plan-v1', 'plan_kind': 'fit',
               **common, 'execution_plan': asdict(fit)}
    if old_allowance is not None:
        generation_doc['owner_allowance'] = copy.deepcopy(old_allowance)
        fit_doc['owner_allowance'] = copy.deepcopy(old_allowance)
    return generation_doc, fit_doc, protocol_doc


def _admission_documents(*, cfg, candidate, output_pins, plan_paths,
                         terminal_result):
    admission = cfg.get('admission')
    require(isinstance(admission, dict) and set(admission) == {
        'workspace', 'decision_relpath', 'predecessor_decision'},
        'exact external admission inputs required')
    workspace = Path(admission['workspace'])
    roots = [Path(cfg[k]).resolve() for k in ('repo', 'data', 'output')]
    require(workspace.is_absolute() and not os.path.lexists(workspace)
            and all(not workspace.resolve().is_relative_to(root)
                    and not root.is_relative_to(workspace.resolve()) for root in roots),
            'fresh isolated admission workspace required')
    decision = admission['decision_relpath']
    require(isinstance(decision, str) and decision.startswith('coordination/')
            and not Path(decision).is_absolute() and '..' not in Path(decision).parts,
            'invalid successor decision path')
    require(decision != candidate['decision_relpath'], 'successor decision must be fresh')
    for pin in (admission['predecessor_decision'],):
        require(isinstance(pin, dict) and set(pin) == {'relpath', 'sha256'}
                and isinstance(pin['relpath'], str) and HEX.fullmatch(pin['sha256']),
                'exact predecessor admission pin required')
    require(admission['predecessor_decision']['relpath'] == candidate['decision_relpath'],
            'predecessor decision differs from candidate')
    candidate_pin = output_pins['docs/readiness/' + candidate['cell'] + '/candidate.json']
    plans = {kind: {'relpath': output_pins[path]['relpath'],
                    'sha256': output_pins[path]['sha256']}
             for kind, path in plan_paths.items()}
    external = {
        'schema': 'saea-external-successor-admission-v1', 'candidate': candidate['cell'],
        'candidate_pin': {'relpath': candidate_pin['relpath'], 'sha256': candidate_pin['sha256']},
        'plans': plans, 'predecessor_decision': admission['predecessor_decision'],
        'decision_relpath': decision, 'workspace': str(workspace),
    }
    operation = {
        'schema': 'saea-successor-admission-operation-v1',
        'admission_config': 'docs/readiness/' + candidate['cell'] + '/external-successor-admission.json',
        'predecessor_registration': candidate['predecessor_registration'],
        'predecessor_decision': admission['predecessor_decision'],
        'successor_decision_relpath': decision,
        'workspace': str(workspace), 'terminal_custody_result': terminal_result,
        'status': 'HELD_BEFORE_ADMISSION', 'activation_authorized': False,
    }
    return external, operation


def finalize(cfg, assets, candidate, plan, repo, data, out):
    terminal = cfg['terminal']
    require(isinstance(terminal, dict) and set(terminal) == {
        'evidence', 'resume_from', 'bridge', 'registration_relpath', 'lock'},
        'exact terminal configuration required')
    require(isinstance(terminal['evidence'], list) and terminal['evidence'], 'terminal evidence pins required')
    for pin in terminal['evidence']:
        read_pin(pin)
    read_pin(terminal['lock'])
    registration = logical(terminal['registration_relpath'])
    require(registration not in assets, 'predecessor registration path already exists')
    # This independent gate must establish actual terminal custody; it is not a
    # count check or an admission decision.
    terminal_result = validate_terminal_custody(cfg=cfg, candidate=candidate, plan=plan, data=data)
    require(isinstance(terminal_result, dict)
            and terminal_result.get('schema') == 'saea-checkpoint-terminal-custody-result-v1'
            and terminal_result.get('activation_authorized') is False
            and isinstance(terminal_result.get('ancestors'), list)
            and terminal_result['ancestors'],
            'terminal custody verifier returned no durable result')
    names = cfg['fragments']
    fragments = {kind: strict_json(assets['configs/' + kind + '/' + name + '.json'])
                 for kind, name in names.items()}
    original_blob = assets[cfg['candidate']['relpath']]
    baseline, experiment, successor = build_successor_documents(
        original=candidate, plan=plan, fragments=fragments, target=cfg['target'],
        dispatch=cfg['dispatch'], width=cfg['width'], supply=cfg['supply']['new'],
        declarations=terminal['resume_from'], bridge=terminal['bridge'],
        registration_relpath=registration, registration_sha256=digest(original_blob), data=data)
    files = dict(assets)
    files[registration] = original_blob
    files[cfg['candidate']['relpath']] = encode(successor)
    files['configs/baseline/' + names['baseline'] + '.json'] = encode(baseline)
    files['configs/experiment/' + names['experiment'] + '.json'] = encode(experiment)
    require(files[cfg['plan']['relpath']] == assets[cfg['plan']['relpath']], 'original plan changed')
    cell_path = 'docs/readiness/' + successor['cell']
    protocol_path = cell_path + '/protocol-successor-v1.json'
    generation_path = cell_path + '/generation-plan-successor-v1.json'
    fit_path = cell_path + '/fit-plan-successor-v1.json'
    terminal_path = cell_path + '/terminal-custody-result.json'
    for name in (protocol_path, generation_path, fit_path, terminal_path):
        require(name not in files, 'successor output path already exists')
    generation_doc, fit_doc, protocol_doc = _successor_plan_documents(
        candidate=successor, plan=plan, baseline=baseline, experiment=experiment,
        target=cfg['target'], supply=cfg['supply']['new'], repo=repo)
    protocol_blob = encode(protocol_doc)
    files[generation_path] = encode(generation_doc)
    files[fit_path] = encode(fit_doc)
    files[protocol_path] = protocol_blob
    files[terminal_path] = encode(terminal_result)
    temp = Path(tempfile.mkdtemp(prefix='.' + out.name + '.', dir=out.parent))
    try:
        os.chmod(temp, 0o700)
        for name, blob in sorted(files.items()):
            path = temp / logical(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            publish_new(path, blob)
        def manifest(root):
            return {'schema': 'saea-experiment-assets-v1', 'files': {
                name: {'path': str(root / name), 'sha256': digest(blob)}
                for name, blob in sorted(files.items())}}
        temporary_manifest = temp / '.validation-manifest.json'
        publish_new(temporary_manifest, encode(manifest(temp)))
        sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
        from registered_workload import validate_workload, assert_execution_binding
        from speech_aware_evidence_acquisition.core.flight.campaign import load_campaign_cells
        from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
        with asset_environment(temporary_manifest, width=cfg['width'],
                               profile=successor.get('profile')):
            rows = validate_workload(repo, data, names['experiment'], successor)
            assert_execution_binding(successor, repo=repo)
            cell, = load_campaign_cells(repo, names['experiment'])
            with tempfile.TemporaryDirectory(prefix='saea-successor-runtime-') as runtime:
                runtime_obj = build_flight_runtime(repo, data, **names, run_id=cell.run_id, attempt_id=cell.attempt_id,
                    runtime_root=Path(runtime), lock_path=Path(terminal['lock']['path']),
                    resume_from=cell.resume_from, checkpoint_compatibility=cell.checkpoint_compatibility)
                require(getattr(runtime_obj, '_admitted_level', None) == cfg['width'],
                        'runtime admitted width was not applied')
                require(getattr(runtime_obj, '_deployment_profile', None) == successor['profile'],
                        'runtime deployment profile was not applied')
        temporary_manifest.unlink()
        output_pins = {
            name: {'relpath': name, 'path': str(out / name), 'sha256': digest(blob)}
            for name, blob in files.items()
        }
        plan_paths = {'generation': generation_path, 'fit': fit_path, 'protocol': protocol_path}
        external, operation = _admission_documents(
            cfg=cfg, candidate=successor, output_pins=output_pins,
            plan_paths=plan_paths,
            terminal_result={'path': str(out / terminal_path),
                             'sha256': digest(files[terminal_path]),
                             'result_schema': terminal_result['schema']})
        admission_path = cell_path + '/external-successor-admission.json'
        operation_path = cell_path + '/successor-admission-operation.json'
        files[admission_path] = encode(external)
        files[operation_path] = encode(operation)
        for name in (admission_path, operation_path):
            require(name not in assets, 'admission output path already exists')
            path = temp / name
            path.parent.mkdir(parents=True, exist_ok=True)
            publish_new(path, files[name])
        final_manifest = encode(manifest(out))
        publish_new(temp / 'assets.json', final_manifest)
        publish_new(temp / 'pin-index.json', encode({'schema':'saea-checkpoint-successor-pin-index-v1',
            'assets_manifest': {'path': str(out / 'assets.json'), 'sha256': digest(final_manifest)},
            'assets': manifest(out)['files'],
            'terminal_custody': {'path': str(out / terminal_path),
                                 'sha256': digest(files[terminal_path])},
            'admission_config': {'path': str(out / admission_path),
                                 'sha256': digest(files[admission_path])},
            'activation_authorized': False}))
        for directory in sorted((p for p in temp.rglob('*') if p.is_dir()), reverse=True):
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
        fd = os.open(temp, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
        _publish_directory(temp, out)
        return {'status':'HELD', 'output':str(out), 'rows': rows, 'activation_authorized':False,
                'new_model_calls':0, 'new_fit_calls':0,
                'terminal_custody': {'path': str(out / terminal_path),
                                     'sha256': digest(files[terminal_path])},
                'admission_config': {'path': str(out / admission_path),
                                     'sha256': digest(files[admission_path])}}
    finally:
        if temp.exists(): shutil.rmtree(temp)


def build_successor_documents(*, original, plan, fragments, target, dispatch, width,
                              supply, declarations, bridge, registration_relpath, registration_sha256, data):
    """Derive held successor documents from original registration and four fragments.

    This pure compiler does not itself certify any terminal state or authorize use.
    Callers must run native workload and checkpoint verification before publication.
    """
    from speech_aware_evidence_acquisition.core.config import _compose_fragments, protocol_hash
    from speech_aware_evidence_acquisition.core.flight.campaign import CampaignCell
    from speech_aware_evidence_acquisition.core.flight.checkpoint_compatibility import (
        CheckpointCompatibility, validate_bridge_declarations)
    from speech_aware_evidence_acquisition.core.flight.resume import resume_declarations_from_document
    from speech_aware_evidence_acquisition.rails.dispatch_slot_fill import parse_dispatch_slot_fill
    require(set(fragments) == {'model', 'dataset', 'baseline', 'experiment'}, 'four fragments required')
    require(type(width) is int and 1 <= width <= 128, 'invalid width')
    parse_dispatch_slot_fill({'dispatch': dispatch})
    baseline = copy.deepcopy(fragments['baseline'])
    experiment = copy.deepcopy(fragments['experiment'])
    candidate = copy.deepcopy(original)
    require(len(experiment['cells']) == 1, 'exactly one campaign cell required')
    cell = experiment['cells'][0]
    require(cell['run_id'] == original['run_id'] and cell['attempt_id'] == original['attempt_id'],
            'original registration and campaign identity differ')
    require(plan['execution_plan']['run_id'] == original['run_id'] and
            plan['execution_plan']['attempt_id'] == original['attempt_id'] and
            plan['execution_plan']['protocol_hash'] == original['protocol_hash'] and
            plan['composed_config_hash'] == original['composed_config_hash'], 'original plan differs')
    require(set(target) == {'attempt_id', 'fit_run_id', 'fit_attempt_id'} and
            all(isinstance(v, str) and NAME.fullmatch(v) for v in target.values()), 'invalid successor identity')
    require(target['attempt_id'] != original['attempt_id'] and
            target['fit_run_id'] != original['fit_run_id'] and
            target['fit_attempt_id'] != original['fit_attempt_id'], 'successor identity reused')
    require(baseline['native_contact_fit']['run_id'] == original['fit_run_id'] and
            baseline['native_contact_fit']['attempt_id'] == original['fit_attempt_id'], 'source FIT differs')
    baseline['native_contact_fit']['run_id'] = target['fit_run_id']
    baseline['native_contact_fit']['attempt_id'] = target['fit_attempt_id']
    baseline['prepared_phase_supply'] = copy.deepcopy(supply)
    if dispatch is None:
        baseline.pop('dispatch', None)
    else:
        baseline['dispatch'] = copy.deepcopy(dispatch)
    cell['attempt_id'] = target['attempt_id']
    cell['resume_from'] = copy.deepcopy(declarations)
    cell['checkpoint_compatibility'] = copy.deepcopy(bridge)
    cell['readiness'] = {'state':'held', 'requirements':[], 'reason':'checkpoint successor requires native admission'}
    campaign = CampaignCell.from_document(cell)
    parsed = CheckpointCompatibility.from_document(bridge, attempt_id=target['attempt_id'])
    validate_bridge_declarations(parsed, resume_declarations_from_document(campaign.resume_from))
    candidate.update(target)
    candidate['execution']['admitted_level'] = width
    candidate['job_config']['execution']['admitted_level'] = width
    require(isinstance(registration_sha256, str) and HEX.fullmatch(registration_sha256), 'original byte hash required')
    candidate['predecessor_registration'] = {'relpath': logical(registration_relpath),
        'sha256': registration_sha256}
    names = {'model': original['model_fragment'],
             'dataset': Path(original['dataset']['relpath']).stem,
             'baseline': original['baseline'], 'experiment': original['experiment']}
    docs = dict(fragments, baseline=baseline, experiment=experiment)
    composed = _compose_fragments(names, docs, data)
    candidate['composed_config_hash'] = composed.config_hash
    candidate['protocol_hash'] = protocol_hash(composed.config_hash, composed.values['arm'],
                                                original['split_identity_hash'])
    return baseline, experiment, candidate


def execute(config_path):
    cfg = strict_json(Path(config_path).read_bytes())
    fields = {'schema', 'mode', 'repo', 'data', 'output', 'assets_manifest', 'candidate',
              'plan', 'fragments', 'target', 'dispatch', 'width', 'supply'}
    require(isinstance(cfg, dict) and set(cfg) == (fields | ({'terminal', 'admission'} if cfg.get('mode') == 'finalize' else set())) and
            cfg['schema'] == 'saea-checkpoint-successor-compiler-v1', 'exact compiler schema required')
    require(cfg['mode'] in ('prepare', 'finalize'), 'invalid compiler mode')
    repo, data, out = (Path(cfg[k]) for k in ('repo', 'data', 'output'))
    require(all(p.is_absolute() for p in (repo, data, out)) and repo.is_dir() and data.is_dir(), 'absolute existing roots required')
    require(out.parent.is_dir() and not out.parent.is_symlink() and not os.path.lexists(out), 'fresh output required')
    root = out.parent.resolve() / out.name
    require(not any(root.is_relative_to(p.resolve()) or p.resolve().is_relative_to(root)
                    for p in (repo, data)), 'output must be isolated from repo and data')
    fragments = cfg['fragments']
    require(isinstance(fragments, dict) and set(fragments) == {'model', 'dataset', 'baseline', 'experiment'} and
            all(isinstance(x, str) and NAME.fullmatch(x) for x in fragments.values()), 'four named fragments required')
    target = cfg['target']
    require(isinstance(target, dict) and set(target) == {'attempt_id', 'fit_run_id', 'fit_attempt_id'} and
            all(isinstance(x, str) and NAME.fullmatch(x) for x in target.values()), 'fresh successor identities required')
    require(type(cfg['width']) is int and 1 <= cfg['width'] <= 128, 'bounded width required')
    from speech_aware_evidence_acquisition.rails.dispatch_slot_fill import parse_dispatch_slot_fill
    parse_dispatch_slot_fill({'dispatch': cfg['dispatch']})
    manifest = strict_json(read_pin(cfg['assets_manifest']))
    require(isinstance(manifest, dict) and set(manifest) == {'schema', 'files'} and
            manifest['schema'] == 'saea-experiment-assets-v1' and isinstance(manifest['files'], dict), 'invalid assets manifest')
    assets = {}
    for name, pin in manifest['files'].items():
        logical(name)
        assets[name] = read_pin(pin)
    def original(item):
        require(isinstance(item, dict) and set(item) == {'relpath', 'sha256'}, 'logical source pin required')
        name = logical(item['relpath'])
        require(name in assets and digest(assets[name]) == item['sha256'], 'original source asset differs')
        return strict_json(assets[name])
    candidate, plan = original(cfg['candidate']), original(cfg['plan'])
    for kind, name in fragments.items():
        require('configs/' + kind + '/' + name + '.json' in assets, 'fragment absent from pinned assets')
    require(candidate['experiment'] == fragments['experiment'] and candidate['baseline'] == fragments['baseline'] and
            candidate['model_fragment'] == fragments['model'] and
            candidate['dataset']['relpath'] == 'configs/dataset/' + fragments['dataset'] + '.json', 'registration fragment differs')
    source = plan['execution_plan']
    require(source['run_id'] == candidate['run_id'] and source['attempt_id'] == candidate['attempt_id'] and
            source['protocol_hash'] == candidate['protocol_hash'] and
            plan['composed_config_hash'] == candidate['composed_config_hash'], 'registered plan identity differs')
    require(target['attempt_id'] != candidate['attempt_id'] and target['fit_run_id'] != candidate['fit_run_id'] and
            target['fit_attempt_id'] != candidate['fit_attempt_id'], 'successor identities must be fresh')
    supply = cfg['supply']
    require(isinstance(supply, dict) and set(supply) == {'old', 'new'}, 'both supply pins required')
    for pin in supply.values():
        require(isinstance(pin, dict) and set(pin) == {'relpath', 'sha256'} and
                isinstance(pin['relpath'], str) and not Path(pin['relpath']).is_absolute() and
                '..' not in Path(pin['relpath']).parts, 'invalid supply pin')
        path = (data / pin['relpath']).resolve()
        require(path.is_relative_to(data.resolve()) and path.is_file() and
                digest(path.read_bytes()) == pin['sha256'], 'supply pin differs')
    require(strict_json((data / supply['old']['relpath']).read_bytes())['output'] ==
            strict_json((data / supply['new']['relpath']).read_bytes())['output'], 'supply output differs')
    base = strict_json(assets['configs/baseline/' + fragments['baseline'] + '.json'])
    require(base['prepared_phase_supply'] == supply['old'], 'registered supply differs')
    if cfg['mode'] == 'finalize':
        return finalize(cfg, assets, candidate, plan, repo, data, out)
    temp = Path(tempfile.mkdtemp(prefix='.' + out.name + '.', dir=out.parent))
    try:
        os.chmod(temp, 0o700)
        pins = {}
        for name in sorted(assets):
            path = temp / name
            path.parent.mkdir(parents=True, exist_ok=True)
            publish_new(path, assets[name])
            pins[name] = {'path': str(out / name), 'sha256': digest(assets[name])}
        template = {'schema': 'saea-checkpoint-successor-held-template-v1', 'status': 'HELD_STATIC_ONLY',
                    'config': cfg, 'future_requirements': ['terminal attempts', 'checkpoint pins',
                    'source review pins', 'generation and FIT contact custody'], 'activation_authorized': False}
        publish_new(temp / 'held-template.json', encode(template))
        publish_new(temp / 'pin-index.json', encode({'schema':'saea-checkpoint-successor-pin-index-v1',
                    'source_manifest': cfg['assets_manifest'], 'assets': pins,
                    'activation_authorized': False}))
        for path in sorted((p for p in temp.rglob('*') if p.is_dir()), reverse=True):
            fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
        fd = os.open(temp, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
        _publish_directory(temp, out)
    finally:
        if temp.exists(): shutil.rmtree(temp)
    return {'status':'HELD_STATIC_ONLY', 'output':str(out), 'activation_authorized':False,
            'new_model_calls':0, 'new_fit_calls':0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    print(json.dumps(execute(parser.parse_args().config), sort_keys=True))
