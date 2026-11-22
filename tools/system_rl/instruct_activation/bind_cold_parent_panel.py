"""Bind a new parent instance to the exact warm and measurement wires of a candidate.

The other, long-lived control is not used to measure the parent effect.
Read-only links to immutable wires preserve audio, prompts, sample seeds and labels.
"""
import json
import sys
from pathlib import Path
from io_utils import doc, pin, require, save, storage


def bind(c):
    base = Path(c['root']).resolve(strict=True)
    src, dst = [base / c[k] for k in ('source_panel', 'output_panel')]
    require(src.parent == dst.parent == base and not dst.exists(), 'unexpected panel location')
    storage(dst, 0)
    rows = doc(pin(src / 'PANEL.json'))
    labels = doc(pin(src / 'gold.private.json'))
    require(len(rows) == len(labels) == len({r['sha256'] for r in rows})
            and set(r['wire'] for r in labels) == {r['sha256'] for r in rows}, 'panel/label mismatch')
    checked = []
    for r in rows:
        srcfile = Path(r['path'])
        require(srcfile.parent == src / 'wires' and pin(srcfile) == r, 'foreign or changed wire')
        checked.append(srcfile)
    # The paired-study loader verifies path containment. Use hard links, not symlinks.
    (dst / 'wires').mkdir(parents=True, exist_ok=False)
    refs=[]
    for r,file in zip(rows,checked,strict=True):
        target=dst/'wires'/file.name
        target.hardlink_to(file)
        refs.append(dict(r,path=str(target)))
    save(dst/'PANEL.json',refs)
    (dst/'gold.private.json').hardlink_to(src/'gold.private.json')
    save(dst/'SOURCE.json',dict(inputs=pin(src/'PANEL.json'),labels=pin(src/'gold.private.json'),
                                identical_wires=len(rows),independent_instance_required=True))


if __name__ == '__main__':
    bind(json.loads(Path(sys.argv[1]).read_bytes()))
