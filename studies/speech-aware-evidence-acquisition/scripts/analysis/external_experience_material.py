"""Build a reviewed external experience workload from a pinned JSON configuration."""
import argparse
import importlib
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/analysis')]
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from external_experience_workload import measure, ExternalExperienceWorkload
from tfrl_hardset_a0_registration import pin

ADAPTERS = {'external-tfgrpo-v1': 'tfgrpo_external_release',
            'external-tfgrpo-reference-v1': 'tfgrpo_reference_release',
            'external-ace-v1': 'ace_external', 'external-reflexion-v1': 'reflexion_external'}


def build(config):
    fields = {'schema', 'source_kind', 'name', 'run_id', 'release', 'runtime_binding',
              'execution', 'params_path', 'measurement_path', 'material_relpath', 'workload_path'}
    if set(config) != fields or config['schema'] != 'saea-external-experience-material-job-v1':
        raise ValueError('exact material job configuration required')
    if config['source_kind'] not in ADAPTERS:
        raise ValueError('known external algorithm adapter required')
    if any(not isinstance(config[k], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', config[k])
           for k in ('name', 'run_id')):
        raise ValueError('safe name and run identity required')
    release = config['release']; repo = Path(release['repo']).resolve(); data = Path(release['data']).resolve()
    paths = {k: Path(config[k]).resolve() for k in ('params_path', 'measurement_path', 'workload_path')}
    paths['material'] = (repo / config['material_relpath']).resolve()
    if not paths['material'].is_relative_to(repo / 'configs') or any(
            not paths[k].is_relative_to(data) for k in ('params_path', 'measurement_path')):
        raise ValueError('owned repository material and data outputs required')
    if len(set(paths.values())) != len(paths) or any(p.exists() for p in paths.values()):
        raise ValueError('distinct fresh material outputs required; reconcile prior attempt')
    adapter = importlib.import_module('speech_aware_evidence_acquisition.rails.' + ADAPTERS[config['source_kind']])
    # The adapter replays original wire evidence and validates exact review/split pins.
    params = adapter.prepare(**release)
    if params['source_kind'] != config['source_kind']:
        raise ValueError('adapter source identity differs')
    measurement = measure(repo, data, params, config['runtime_binding'])
    def write(path, value):
        publish_new(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode())
    write(paths['params_path'], params); write(paths['measurement_path'], measurement)
    material = {'schema': 'saea-external-experience-material-v1',
                'params': pin(data, str(paths['params_path'].relative_to(data))),
                'measurement': pin(data, str(paths['measurement_path'].relative_to(data)))}
    write(paths['material'], material)
    workload = {'schema': 'saea-external-experience-workload-v1', 'name': config['name'],
                'run_id': config['run_id'], 'sample_ids': params['consumer_ids'],
                'runtime_binding': config['runtime_binding'], 'execution': config['execution'],
                'material': pin(repo, str(paths['material'].relative_to(repo)))}
    ExternalExperienceWorkload(workload)
    write(paths['workload_path'], workload)
    return {'schema': 'saea-external-experience-material-result-v1', 'source_kind': config['source_kind'],
            'name': config['name'], 'rows': len(params['consumer_ids']), 'new_model_calls': 0,
            'workload_path': str(paths['workload_path'])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(json.loads(args.config.read_bytes())), sort_keys=True))
