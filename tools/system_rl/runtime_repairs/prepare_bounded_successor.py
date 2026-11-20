"""Bind the next bounded TRAIN window to the verified live adapter and Clock."""
import argparse
from pathlib import Path

from dependencies import require
from run_package import pin, read, write


def prepare(config):
    from bounded_tp4 import validate_routes
    from next_input import bounded_capture_input, migrate
    from publication import promote, validate_publication

    root = Path(config['output'])
    prior_ref = config['prior_input']
    prior = read(prior_ref)
    window = read(config['parity_window'])
    observations = Path(window['evidence'])
    activated = read(observations/'activation-probe01/RESULT.json')['lanes']
    published_ref = promote(window['policy_ref'], config['parity_window'], None,
                            root/'promoted', bounded_tp4=True)
    policy = read(published_ref)
    publication = validate_publication(policy['publication_ref'])
    require(publication['step'] == prior['continuation']['successful_optimizer_steps'] + 1,
            'policy must follow the previous capture step')
    lanes = {}
    for lane, result in zip(window['lanes'], activated, strict=True):
        name = lane['name']
        state = observations/name
        refs = {key: pin(state/file) for key, file in (
            ('activation_ref', 'ACTIVATION.json'), ('cache_ref', 'CACHE-POLICY.json'),
            ('precision_ref', 'PRECISION.json'))}
        require(all(ref['sha256'] == result[key]['sha256'] for key, ref in refs.items()),
                'live lane evidence changed')
        refs['response_ref'] = pin(observations/'activation-probe01'/name/'RESPONSE.json')
        require(refs['response_ref']['sha256'] == result['receipt_ref']['sha256'],
                'live lane response changed')
        lanes[name] = write(root/f'{name}-witness.json', dict(lane=name, **refs))
    profile_ref = prior['capture']['profile_ref']
    witness = write(root/'ENDPOINTS.json', dict(
        schema='omni-bounded-tp4-endpoints-v1', fixture=False, passed=True,
        policy_version=policy['version'], model=policy['model'], thinking=False,
        asset_manifest_ref=policy['asset_manifest_ref'], profile_ref=profile_ref, lanes=lanes))
    validate_routes(publication, witness, read(profile_ref))
    migration = migrate(prior_ref, published_ref, window['checkpoint_ref'], root/'migration')
    compiled = bounded_capture_input(config['sequence'], migration['input_ref'], published_ref,
                                     witness, migration['learning_ref'], root/'prepared',
                                     cohort_sequence=config['cohort_sequence'])
    capture = read(compiled)
    offset = config['window_offset']
    require(type(offset) is int and offset >= prior['bounded_train_probe'].get('offset', 0)
            + prior['bounded_train_probe']['groups'], 'fresh TRAIN window must advance')
    capture['bounded_train_probe']['offset'] = offset
    capture['cohort_id'] = capture['native']['attempt_id'] = config['cohort_id']
    capture['output_root'] = config['capture_root']
    capture['lease_fence'].update(root=config['lease_root'], remote=read(config['remote_lease']))
    capture['authorization']['model_calls'] = True
    return write(root/'CAPTURE.json', capture)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    print(prepare(read(parser.parse_args().config))['path'])
