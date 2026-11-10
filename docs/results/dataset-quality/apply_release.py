"""Reproduce clean SLURP metadata from public exclusions and original JSONL files."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--exclusions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Output exists; use a new directory')
    decisions=read(args.exclusions)
    by_split={s:set() for s in ['train','devel','test','train_synthetic']}
    source={s:read(args.source/(s+'.jsonl')) for s in by_split}
    index={s:{str(r['slurp_id']):r for r in rows} for s,rows in source.items()}
    for decision in decisions:
        split,uid=decision['split'],str(decision['slurp_id'])
        row=index[split][uid]
        assert row['sentence']==decision['sentence'] and row['sentence_annotation']==decision['sentence_annotation']
        assert row['scenario']+'_'+row['action']==decision['original_intent']
        assert hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest()==decision['annotation_sha256']
        by_split[split].add(uid)
        if uid in index['train_synthetic']:
            other=index['train_synthetic'][uid]
            if all(row[k]==other[k] for k in ['sentence','sentence_annotation','scenario','action','entities']):
                by_split['train_synthetic'].add(uid)
    args.output.mkdir(parents=True)
    counts={}
    for split,rows in source.items():
        kept=[row for row in rows if str(row['slurp_id']) not in by_split[split]]
        out=args.output/(split+'.jsonl')
        out.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in kept))
        counts[split]=dict(original=len(rows),retained=len(kept),removed=len(rows)-len(kept),
                           sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    (args.output/'APPLY-RECEIPT.json').write_text(json.dumps(counts,indent=2)+'\n')
    print(json.dumps(counts))


if __name__=='__main__':main()
