"""Exercise release boundaries with an adjacent, unlisted checkpoint and source."""
from pathlib import Path
import json,tempfile
from build import source_members,copy_base,pin
from source.release_members import rebind_manifests
from stage_sampled_actor_release import committed_files

def rejected(fn):
    try:fn()
    except (ValueError,FileNotFoundError):return
    raise AssertionError('invalid release accepted')

def check(root):
    src=root/'source';src.mkdir()
    for n in ('selected.py','unrelated.py'):(src/n).touch()
    assert source_members(src,['selected.py'])==[src/'selected.py']
    for names in (['missing.py'],['*.py'],['../escape.py'],['selected.py']*2):
        rejected(lambda:source_members(src,names))
    c=root/'custody';d=c/'step-4';d.mkdir(parents=True)
    (c/'unrelated').write_text('not a checkpoint member')
    (d/'unused').write_text('not a checkpoint member')
    actor=d/'actor.pt';actor.write_bytes(b'closed member')
    receipt=d/'COMMITTED.json'
    receipt.write_text(json.dumps(dict(recoverable=True,world_size=1,ranks=[dict(rank=0,files={'actor.pt':pin(actor)['sha256']})])))
    ref=pin(receipt);assert set(committed_files(c,receipt,ref))=={receipt,actor}
    actor.write_bytes(b'changed');rejected(lambda:committed_files(c,receipt,ref))
    base=root/'base';rel=Path('parents/worker/SOURCE-PINS.json')
    (base/'tree'/rel).parent.mkdir(parents=True);(base/'client').mkdir()
    (base/'tree'/rel).write_text('[]');(base/'tree'/rel.with_name('HISTORICAL.json')).write_text('not JSON')
    out=root/'release';manifests=copy_base(base,out);assert manifests=={rel}
    rebind_manifests(out/'tree',root/'logical',members=manifests)
    print('PASS: explicit sources, checkpoint members, release binding manifests; changed and missing members rejected')

if __name__=='__main__':
    with tempfile.TemporaryDirectory() as tmp:check(Path(tmp))
