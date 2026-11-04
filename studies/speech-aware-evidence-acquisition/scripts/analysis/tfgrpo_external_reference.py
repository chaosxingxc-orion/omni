"""Prepare, run/resume, or collect a configured external reference TF-GRPO batch."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from speech_aware_evidence_acquisition.rails.tfgrpo_external_reference import prepare, run, collect, audit

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'collect', 'audit'])
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    result = {'prepare': prepare, 'run': run, 'collect': collect, 'audit': audit}[args.action](args.path)
    print(json.dumps(result if args.action == 'audit' else
                     {'schema': result['schema'], 'deployment_accepted': False,
                      'plan': result.get('plan')}))
