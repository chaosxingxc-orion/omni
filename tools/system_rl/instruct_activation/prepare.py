"""Pack the existing Instruct activation for a new local deployment profile."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from config_profile import resolve

HERE = Path(__file__).resolve().parent


def prepare(profile_path, output):
    profile = json.loads(profile_path.read_bytes())
    template = HERE / 'activation.template.json'
    config = resolve(json.loads(template.read_bytes()), profile)
    output.mkdir(mode=0o700)
    for source in (template, HERE / 'launch.py', HERE / 'service.py'):
        shutil.copyfile(source, output / source.name)
    for name, value in [('PROFILE.local.json', profile), ('config.json', config)]:
        target = output / name
        target.write_text(json.dumps(value, indent=2) + '\n')
        target.chmod(0o600)
    pins = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    (output / 'PINS.json').write_text(json.dumps(pins, indent=2) + '\n')
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = prepare(args.profile, args.output)
    print(json.dumps({k: cfg[k] for k in ('name', 'cards', 'port', 'image')}))
