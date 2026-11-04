"""Build a reviewed external experience workload from a pinned JSON configuration."""
import argparse
import importlib
import json
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
    if not isinstance(config, dict) or not fields <= set(config) <= fields | {'evaluation_design', 'evaluation_partition'} or config['schema'] != 'saea-external-experience-material-job-v1':
        raise ValueError('exact material job configuration required')
    if config['source_kind'] not in ADAPTERS:
        raise ValueError('known external algorithm adapter required')
    release = config['release']; repo = Path(release['repo']).resolve(); data = Path(release['data']).resolve()
    paths = {k: Path(config[k]).resolve() for k in ('params_path', 'measurement_path', 'workload_path')}
    paths['material'] = (repo / config['material_relpath']).resolve()
    if not paths['material'].is_relative_to(repo / 'configs') or any(
            not paths[k].is_relative_to(data) for k in ('params_path', 'measurement_path', 'workload_path')):
        raise ValueError('owned repository material and data outputs required')
    if len(set(paths.values())) != len(paths) or any(p.exists() for p in paths.values()):
        raise ValueError('distinct fresh material outputs required; reconcile prior attempt')
    # Validate the exact downstream workload before the adapter publishes its
    # template. The material pin is intentionally absent until bytes exist;
    # constructor/runtime validation does not read material or dispatch work.
    workload = {'schema': 'saea-external-experience-workload-v1', 'name': config['name'],
                'run_id': config['run_id'], 'sample_ids': release['consumer_ids'],
                'runtime_binding': config['runtime_binding'], 'execution': config['execution'],
                'material': None}
    for key in ('evaluation_design', 'evaluation_partition'):
        if key in config:
            workload[key] = config[key]
    if (release.get('contract_control') is not None) != (config.get('evaluation_design') == 'contract-control-v1'):
        raise ValueError('contract control release and workload design differ')
    if release.get('contract_control') is not None and config['source_kind'] != 'external-tfgrpo-v1':
        raise ValueError('contract control preparation currently requires TFGRPO adapter')
    job = ExternalExperienceWorkload(workload)
    binding = job.runtime_binding(repo)
    if (release.get('runtime_model_id') or release['model_id']) != binding['model_fragment']:
        raise ValueError('release model differs from configured deployment')
    adapter = importlib.import_module('speech_aware_evidence_acquisition.rails.' + ADAPTERS[config['source_kind']])
    # The adapter replays original wire evidence and validates exact review/split pins.
    params = adapter.prepare(**release)
    if params['source_kind'] != config['source_kind'] or params['consumer_ids'] != workload['sample_ids']:
        raise ValueError('adapter source or consumer identity differs')
    measurement = measure(repo, data, params, binding)
    def write(path, value):
        publish_new(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode())
    write(paths['params_path'], params); write(paths['measurement_path'], measurement)
    material = {'schema': 'saea-external-experience-material-v1',
                'params': pin(data, str(paths['params_path'].relative_to(data))),
                'measurement': pin(data, str(paths['measurement_path'].relative_to(data)))}
    write(paths['material'], material)
    workload['material'] = pin(repo, str(paths['material'].relative_to(repo)))
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
