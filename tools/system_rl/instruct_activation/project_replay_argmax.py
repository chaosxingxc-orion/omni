"""Observe full-vocabulary argmax on a specified fixed history before forcing the replay action."""
from pathlib import Path
import sys
from io_utils import doc, pin, require, save, digest
from decode_lineage import snapshot, changes
from project_prefill_replay import build
from project_prefill_timing import once


def project(c):
    build(c)
    root=Path(c['projection']['target'])
    original=snapshot(Path(c['original_replay']))
    require(snapshot(root)==original,'changed replay arithmetic')
    rel='parents/jev-system-rl/implementation/decode_action_processor.py'
    mod=root/'bundle'/rel
    src=mod.read_text()
    observations=doc(c['observe'])
    src=once(src,'class ReplayActions(LogitsProcessor):',
             'from io_utils import digest, save\nfrom pathlib import Path\n\n'
             +f'TARGETS = {observations!r}\n\nclass ReplayActions(LogitsProcessor):')
    src=once(src,'return tuple(params.extra_args[\'decode_action_ids\']), output',
             """path = tuple(params.extra_args['decode_action_ids'])
            return path, output, digest([prompt, path])""")
    src=once(src,'self.rows = {}', 'self.rows = {}\n        self.observer_steps = {}')
    src=once(src,'path, output = self.rows[i]','path, output, conditioning = self.rows[i]')
    src=once(src,'            chosen.append(path[len(output)])',"""            position = len(output)
            if position in TARGETS.get(conditioning, []):
                row = logits[i]
                top = row.argmax()
                value = row[top]
                action = path[position]
                # Same argmax operator and full-vocabulary index order as the HF scorer.
                rank = torch.distributed.get_rank()
                key = (conditioning, position)
                step = self.observer_steps.get(key, 0)
                self.observer_steps[key] = step + 1
                save(Path('/runtime/state')/f'argmax-{conditioning}-{position}-r{rank}-s{step}.json',
                     dict(conditioning=conditioning, position=position, rank=rank, step=step, vocab=row.numel(),
                          top=int(top.item()), maxima=int((row==value).sum().item()),
                          action=action, raw_action_logprob=float(row.log_softmax(-1)[action].item())))
            chosen.append(path[len(output)])""")
    compile(src,rel,'exec')
    mod.write_text(src)
    delta=changes(original,snapshot(root))
    require(not delta['engine'] and not delta['rows'] and set(delta['files'])=={rel},'unexpected observer delta')
    save(root/'ARGMAX-DELTA.json',dict(parent=digest(original),changes=delta,observe=c['observe']))


if __name__=='__main__':
    project(doc(pin(sys.argv[1])))
