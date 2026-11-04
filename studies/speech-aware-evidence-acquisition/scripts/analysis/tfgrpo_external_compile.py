"""CLI for external TF-GRPO preview compilation."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from speech_aware_evidence_acquisition.rails.tfgrpo_external_compile import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    context_parser = sub.add_parser('context')
    context_parser.add_argument('--source', type=Path, required=True)
    for name, default in [('max-operations', 2), ('max-items', 48), ('max-characters', 10000)]:
        context_parser.add_argument('--' + name, type=int, default=default)
    compile_parser = sub.add_parser('compile')
    compile_parser.add_argument('--spec', type=Path, required=True)
    for p in (context_parser, compile_parser):
        p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('fresh output required; do not overwrite evidence')
    result = (build_context(args.source, max_operations=args.max_operations,
              max_items=args.max_items, max_characters=args.max_characters)
              if args.action == 'context' else compile_preview(external.read(args.spec)))
    external.save(args.out, result)
    print(result['schema'])
