"""Compile or prepare reviewed multistage TF-GRPO reference artifacts without dispatch."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from speech_aware_evidence_acquisition.rails import tfgrpo_reference_release as reference
from speech_aware_evidence_acquisition.core.atomic_file import publish_new

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['compile', 'prepare'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise ValueError('fresh output required')
    config = json.loads(args.config.read_bytes())
    result = reference.compile_preview(config) if args.action == 'compile' else reference.prepare(**config)
    publish_new(args.out, (json.dumps(result, indent=2, sort_keys=True)+'\n').encode())
    print(json.dumps({'action':args.action, 'new_model_calls':0, 'deployment_accepted':False}))
