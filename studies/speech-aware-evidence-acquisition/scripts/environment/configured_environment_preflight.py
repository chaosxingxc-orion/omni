#!/usr/bin/env python3
"""CPU dependency checks; a successful receipt can be pinned by the launcher."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

SCHEMA = 'saea-environment-preflight-v1'
RUNTIME_PATH_KINDS = {'SPEECHRL_DATA_DIR': 'data-root', 'SAEA_STAGE_TIMING_PATH': 'timing-file', 'SAEA_ADMISSION_CONTROL_CONFIG': 'admission-control'}
PROBE = '''import importlib, importlib.metadata, json, sys
spec = json.loads(sys.argv[1])
checks = []
for module in spec['imports']:
    try:
        importlib.import_module(module)
        checks.append({'module': module, 'ok': True})
    except Exception as exc:
        checks.append({'module': module, 'ok': False, 'error_type': type(exc).__name__})
for package, expected in spec['distributions'].items():
    try:
        actual = importlib.metadata.version(package)
        checks.append({'distribution': package, 'expected': expected, 'actual': actual, 'ok': actual == expected})
    except Exception as exc:
        checks.append({'distribution': package, 'ok': False, 'error_type': type(exc).__name__})
print('SAEA_PREFLIGHT=' + json.dumps({'checks': checks, 'ok': all(c['ok'] for c in checks),
                                    'executable': sys.executable, 'python_version': sys.version.split()[0]}))
'''


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def canonical_runtime_path(value):
    require(isinstance(value, str) and value.startswith('/') and not value.startswith('//') and value != '/'
            and str(Path(value)) == value and '..' not in Path(value).parts
            and not any(c in value for c in ('\x00', '\n', '\r', ':', ',')),
            'runtime path must be canonical, absolute and non-root')
    return Path(value)


def runtime_paths(doc):
    declarations = doc.get('runtime_paths', {})
    require(isinstance(declarations, dict) and set(declarations) <= set(RUNTIME_PATH_KINDS),
            'only known runtime environment paths may be declared')
    for key, item in declarations.items():
        fields = {'kind', 'path', 'mount'} if key in ('SAEA_STAGE_TIMING_PATH', 'SAEA_ADMISSION_CONTROL_CONFIG') else {'kind', 'path'}
        require(isinstance(item, dict) and set(item) == fields
                and item.get('kind') == RUNTIME_PATH_KINDS[key], 'exact runtime path semantics required: ' + key)
        value = canonical_runtime_path(item['path'])
        require(doc['required_environment'].get(key) == str(value),
                'runtime path must match required environment: ' + key)
        if key in ('SAEA_STAGE_TIMING_PATH', 'SAEA_ADMISSION_CONTROL_CONFIG'):
            parent = canonical_runtime_path(item['mount'])
            require(value != parent and value.is_relative_to(parent),
                    'timing file must be strictly below its declared mount')
    require(not (set(doc['required_environment']) & set(RUNTIME_PATH_KINDS)) - set(declarations),
            'runtime environment paths must be explicitly declared')
    return declarations


def load_config(path):
    doc = json.loads(Path(path).read_bytes())
    require(isinstance(doc, dict) and set(doc) - {'runtime_paths', 'required_files', 'compositions'} == {'schema', 'required_environment', 'interpreters', 'timeout_seconds'}, 'exact preflight schema required')
    require(doc['schema'] == SCHEMA, 'unsupported preflight schema')
    require(type(doc['timeout_seconds']) is int and 1 <= doc['timeout_seconds'] <= 300, 'bounded timeout required')
    require(isinstance(doc['required_environment'], dict) and all(
        isinstance(k, str) and re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', k) and isinstance(v, str) and v
        for k, v in doc['required_environment'].items()), 'explicit environment values required')
    runtime_paths(doc)
    files = doc.get('required_files', [])
    require(isinstance(files, list), 'required_files must be a list')
    seen = set()
    for item in files:
        require(isinstance(item, dict) and set(item) == {'path', 'sha256'}, 'exact required file pin expected')
        canonical_runtime_path(item['path'])
        require(isinstance(item['sha256'], str) and re.fullmatch('[a-f0-9]{64}', item['sha256']),
                'required file SHA256 must be lowercase hex')
        require(item['path'] not in seen, 'duplicate required file')
        seen.add(item['path'])
    require(isinstance(doc['interpreters'], list) and doc['interpreters'], 'interpreter checks required')
    names = set()
    for item in doc['interpreters']:
        require(isinstance(item, dict) and set(item) == {'name', 'python', 'environment_key', 'imports', 'distributions'}, 'exact interpreter schema required')
        require(isinstance(item['name'], str) and re.fullmatch(r'[a-z][a-z0-9_-]*', item['name']) and item['name'] not in names, 'unique interpreter name required')
        names.add(item['name'])
        require(isinstance(item['python'], str) and Path(item['python']).is_absolute() and '\x00' not in item['python'], 'absolute interpreter required')
        require(item['environment_key'] is None or (isinstance(item['environment_key'], str)
                and doc['required_environment'].get(item['environment_key']) == item['python']),
                'interpreter selection must match required environment')
        require(isinstance(item['imports'], list) and item['imports'] and all(isinstance(x, str) and re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)*', x) for x in item['imports']), 'module names required')
        require(isinstance(item['distributions'], dict) and all(isinstance(k, str) and re.fullmatch(r'[A-Za-z0-9_.-]+', k) and isinstance(v, str) and v for k, v in item['distributions'].items()), 'exact distribution versions required')
    compositions=doc.get('compositions',[])
    require(isinstance(compositions,list),'composition checks must be a list')
    for item in compositions:
        require(isinstance(item,dict) and set(item)=={'id','repo_root','data_root','fragments','expected_rows','verify_media'},'exact composition probe required')
        require(isinstance(item['id'],str) and re.fullmatch('[a-z0-9-]+',item['id']),'composition identity required')
        canonical_runtime_path(item['repo_root']);canonical_runtime_path(item['data_root'])
        require(isinstance(item['fragments'],dict) and set(item['fragments'])=={'model','dataset','baseline','experiment'},'four real fragments required')
        require(type(item['expected_rows']) is int and item['expected_rows']>0 and type(item['verify_media']) is bool,'explicit population and media verification required')
    return doc


def persist(path, doc):
    with path.open('x') as handle:
        handle.write(json.dumps(doc, sort_keys=True, indent=2) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def composition_probe(spec):
    """Exercise production composition, graph parsing and native media binding only."""
    import sys
    root=Path(__file__).resolve().parents[2]
    sys.path[:0]=[str(root/'src'),str(root/'scripts/analysis')]
    from speech_aware_evidence_acquisition.core.config import compose
    from speech_aware_evidence_acquisition.rails.adapters.runtime_view import load_runtime_view, registered_population_ids
    from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
    from speech_aware_evidence_acquisition.rails.native_episode_config import load_native_episode_config
    from speech_aware_evidence_acquisition.rails.phase_material import load_phase_material_specs
    cfg=compose(Path(spec['repo_root']),**spec['fragments'],data_root=spec['data_root']);values=cfg.values
    ids=registered_population_ids(values,spec['fragments']['dataset'])
    selected=load_runtime_view(values,Path(spec['data_root']),sample_ids=ids,workers=4) if spec['verify_media'] else ()
    count=len(selected) if spec['verify_media'] else len(ids) if ids is not None else values['count']
    require(count==spec['expected_rows'],'native population denominator differs')
    program=default_phase_program(values['policy'])
    native=load_native_episode_config(values,program)
    materials=load_phase_material_specs(values,program)
    # All actual configured templates are opened through the same asset resolver.
    from speech_aware_evidence_acquisition.core.experiment_assets import resolve_asset
    templates=[]
    for material in materials:
        path=resolve_asset(Path(spec['repo_root']),'configs/'+material.instruction_template)
        require(path.is_file(),'phase template absent')
        templates.append({'phase':material.phase,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return {'id':spec['id'],'ok':True,'config_hash':cfg.config_hash,'rows':count,
            'media_hashes_verified':len(selected),'policy':values['policy'],'templates':templates,
            'new_model_calls':0,'fit_calls':0}


def execute(config, output):
    doc = load_config(config)
    started = time.monotonic()
    output = Path(output)
    # A unique directory preserves old failure receipts and concurrent invocations.
    output.mkdir(exist_ok=False)
    checks = [{'environment': key, 'ok': os.environ.get(key) == value}
              for key, value in sorted(doc['required_environment'].items())]
    for item in doc.get('required_files', []):
        check = {'required_file': item['path'], 'expected_sha256': item['sha256'], 'ok': False}
        try:
            require(Path(item['path']).is_file(), 'required dependency is not a regular file')
            checksum = hashlib.sha256()
            with Path(item['path']).open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    checksum.update(chunk)
            check['actual_sha256'] = checksum.hexdigest()
            check['ok'] = check['actual_sha256'] == item['sha256']
        except (OSError, ValueError) as exc:
            check['error_type'] = type(exc).__name__
        checks.append(check)
    if all(check['ok'] for check in checks):
        for spec in doc['interpreters']:
            try:
                result = subprocess.run([spec['python'], '-B', '-c', PROBE, json.dumps(spec)],
                                        text=True, capture_output=True, timeout=doc['timeout_seconds'],
                                        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
                lines = [line[len('SAEA_PREFLIGHT='):] for line in result.stdout.splitlines() if line.startswith('SAEA_PREFLIGHT=')]
                require(result.returncode == 0 and len(lines) == 1, 'interpreter probe failed')
                probe = json.loads(lines[0])
                require(type(probe['ok']) is bool and isinstance(probe['checks'], list) and probe['checks'], 'invalid probe response')
                checks.append({'interpreter': spec['name'], 'configured_python': spec['python'], **probe})
            except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError) as exc:
                checks.append({'interpreter': spec['name'], 'configured_python': spec['python'],
                               'ok': False, 'error_type': type(exc).__name__})
    if all(check['ok'] for check in checks):
        for spec in doc.get('compositions',[]):
            try:checks.append(composition_probe(spec))
            except Exception as exc:checks.append({'id':spec['id'],'ok':False,'error_type':type(exc).__name__,'error':str(exc)})
    ok = all(check['ok'] for check in checks)
    result = {'schema': SCHEMA, 'config_sha256': digest(doc), 'ok': ok,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'checks': checks, 'elapsed_seconds': time.monotonic() - started,
              'new_model_calls': 0, 'automatic_retry': False}
    persist(output / 'result.json', result)
    if ok:
        persist(output / 'ready.json', {'schema': SCHEMA, 'config_sha256': digest(doc),
                                      'result_sha256': hashlib.sha256((output / 'result.json').read_bytes()).hexdigest(),
                                      'ok': True, 'new_model_calls': 0})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.config, args.output)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result['ok'] else 1)
