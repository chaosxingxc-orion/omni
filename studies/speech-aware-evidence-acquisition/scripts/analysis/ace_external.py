"""Collect, compile, or prepare reviewed external ACE additions without dispatch."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from speech_aware_evidence_acquisition.rails import ace_external as ace
from speech_aware_evidence_acquisition.rails.external_exploration import collect
from speech_aware_evidence_acquisition.core.atomic_file import publish_new

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['collect','compile','prepare'])
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.action=='collect':
        result=collect(args.config)
    else:
        if args.out is None or args.out.exists():raise ValueError('fresh output required')
        config=json.loads(args.config.read_bytes())
        result=ace.compile_preview(config) if args.action=='compile' else ace.prepare(**config)
        publish_new(args.out,(json.dumps(result,indent=2,sort_keys=True)+'\n').encode())
    print(json.dumps({'action':args.action,'new_model_calls':0,'deployment_accepted':False}))
