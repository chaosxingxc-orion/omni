"""Local-only closure: copy existing stable files; never prepare data or contact services."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
ROOT=Path(__file__).resolve().parents[3]
OWNER=ROOT/'delivery/s-terminal-handoff-repair-v1'

def pin(p):
    return dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,
                sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest())

def make():
    files=set()
    def add(path):
        p=Path(path)
        if not p.is_absolute():p=ROOT/p
        assert p.is_file(),str(p)
        files.add(p)
    def tree(path,pattern='*'):
        for p in (ROOT/path).rglob(pattern):
            if p.is_file() and '__pycache__' not in p.parts:add(p)
    add('delivery/s-terminal-handoff-repair-v1/DEPLOYMENT.md')
    add('delivery/s-terminal-handoff-repair-v1/FIXED-CPU-DEPENDENCIES.json')
    add('delivery/s-terminal-handoff-repair-v1/evidence/MAIN-FIXED-IMAGE-IMPORT-PROBE.json')
    tree('delivery/s-terminal-handoff-repair-v1/code','*.py')
    tree('delivery/s-terminal-handoff-repair-v1/code','*.sh')
    for name in ('config','fixtures','handoff-execution-code'):tree('delivery/s-terminal-handoff-repair-v1/'+name)
    for name in ('delivery/s-single-intent-fewshot-v1/code','delivery/s-native-throughput-v1/code',
                 'delivery/s-correct-p-data-v1/code'):
        tree(name,'*.py')
        for sub in ('config','fixtures'):tree(str(Path(name).parent/sub))
    tree('delivery/s-native-throughput-v1/resources')
    tree('delivery/union-fewshot-implementation-v1/source/studies/speech-aware-evidence-acquisition/src')
    # Original immutable native used in the inverse-diff fixture, not deployed.
    for rel in ('train_runtime.py','bind_train.py','train-runtime-v1/run_s_native.py','train-runtime-v1/native_resources.py'):
        add('delivery/s-single-intent-execution-v1/rolling-code/'+rel)
    cfgpath=ROOT/'delivery/s-tfrl-rolling-v1/train-paired-fewshot/train-remainder-part1-fewshot-v1/single_intent_fewshot/native-UNBOUND.json'
    cfg=json.loads(cfgpath.read_bytes());add(cfgpath)
    for field in ('manifest','snapshot','parent_config','samples'):add(cfg[field])
    for field in ('budget_parent','budget_parent_script'):add(cfg[field]['path'])
    add(Path(cfg['prepared'])/'source-proof.private.jsonl')
    proofs=[json.loads(l) for l in (ROOT/cfg['prepared']/'source-proof.private.jsonl').read_bytes().splitlines()]
    for proof in proofs:add(proof['original_packet'])
    for row in (ROOT/cfg['manifest']).read_bytes().splitlines():add(json.loads(row)['payload'])
    parent=json.loads((ROOT/cfg['parent_config']).read_bytes())
    for p in parent['pins']:add(p)
    for p in ('delivery/s-tfrl-rolling-v1/evidence/train-remainder-part1-fewshot-v1/budget.json',
              'delivery/s-tfrl-rolling-v1/evidence/train-remainder-part1-fewshot-v1/CPU-VERIFIED.json'):
        add(p)
    doc=dict(schema='s-handoff-transport-v2',workspace='preserve relative tree; ROOT derived from __file__',
             image='sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026',
             python='/opt/saea-venv/bin/python',network='none',devices=[],
             real_samples=933,real_branches=2799,new_source=False,journal_files_included=False,
             files=[pin(p) for p in sorted(files)])
    (OWNER/'TRANSPORT-MANIFEST.json').write_text(json.dumps(doc,indent=2)+'\n')
    return doc

def verify(doc):
    for item in doc['files']:
        p=ROOT/item['path'];assert pin(p)==item,'transport drift: '+str(p)
    return len(doc['files'])

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('build','verify','pack'))
    p.add_argument('--output');a=p.parse_args()
    doc=make() if a.mode=='build' else json.loads((OWNER/'TRANSPORT-MANIFEST.json').read_bytes())
    count=verify(doc)
    if a.mode=='pack':
        assert a.output,'--output required'
        output=Path(a.output);assert not output.exists(),'fresh transport output required'
        with tarfile.open(output,'x:gz') as tar:
            for item in doc['files']:tar.add(ROOT/item['path'],arcname=item['path'],recursive=False)
            packed={item['path'] for item in doc['files']}
            for file in sorted(OWNER.rglob('*')):
                if file.is_file() and '__pycache__' not in file.parts:
                    rel=str(file.relative_to(ROOT))
                    if rel not in packed:tar.add(file,arcname=rel,recursive=False)
    print(json.dumps(dict(status='PASS',files=count,bytes=sum(x['bytes'] for x in doc['files']))))
if __name__=='__main__':main()
