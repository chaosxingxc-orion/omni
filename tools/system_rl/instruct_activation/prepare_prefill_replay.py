"""Bind fixed-action wires to a completed natural batch and its original prompts."""
from pathlib import Path
import sys

from decode_action_replay import prepare
from io_utils import digest, doc, pin, require, save


def prepare_panel(c):
    cfg = doc(pin(c['natural_config']))
    lane = c['lane']
    rounds = Path(cfg['output'])
    require(doc(pin(rounds/'DRAINED.json'))['guards_releasable'], 'open natural window')
    result = doc(pin(rounds/'RESULT.json'))
    require(result['passed'], 'natural run failed')
    ref, = result['rounds']
    raw = doc(ref)
    width, = cfg['grouped_concurrency']
    require(raw['admission']=='group_drained' and raw['concurrency_per_service'][lane]==width,
            'requires a fixed natural batch')
    root = Path(c['panel'])
    salt = digest([cfg['measurement_id'], raw['block']])
    entries = sorted((r for r in raw['requests'] if r['service']==lane),
                     key=lambda r:(r['group'],r['replicate']))
    wires = raw['input_refs']
    require(len(entries)==len(wires) and len(wires)%width==0,'incomplete panel')
    out = []
    for i,(row,wire) in enumerate(zip(entries,wires,strict=True)):
        require((row['group'],row['replicate'])==divmod(i,width),'wrong batch position')
        response = row['receipt_ref']
        intent = pin(Path(response['path']).with_name('INTENT.json'))
        replay = prepare(dict(wire=wire, intent=intent, response=response, source_salt=salt))
        out.append(dict(natural=wire, intent=intent, response=response,
                        replay=save(root/'wires'/f'{i}.json',replay)))
    save(root/'SOURCE.json',dict(config=pin(c['natural_config']),round=ref,
                                 width=width,salt=salt,rows=out))


if __name__=='__main__':
    prepare_panel(doc(pin(sys.argv[1])))
