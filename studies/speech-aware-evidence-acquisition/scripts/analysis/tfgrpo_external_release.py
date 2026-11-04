"""Build reviewed external TF-GRPO consumer parameters from a JSON configuration."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from speech_aware_evidence_acquisition.rails.tfgrpo_external_release import prepare
from speech_aware_evidence_acquisition.core.atomic_file import publish_new

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise ValueError('fresh output required')
    params = prepare(**json.loads(args.config.read_bytes()))
    publish_new(args.out, (json.dumps(params, indent=2, sort_keys=True)+'\n').encode())
    print('external-tfgrpo-v1 development consumer prepared')
