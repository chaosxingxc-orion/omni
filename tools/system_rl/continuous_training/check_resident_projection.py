"""Exercise a process-lifetime mount with real consecutive rollout control bytes."""
import argparse,json,tempfile,sys,hashlib
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    for k in ('run','scratch'):p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--captures',nargs='+',required=True)
    a=p.parse_args();sys.path.insert(0,str(Path(__file__).parent/'source'))
    from resident_inputs import publish,digest
    with tempfile.TemporaryDirectory(dir=a.scratch) as tmp:
        root=Path(tmp);inbox=root/'inbox';rosters=[]
        def window(name,items):
            table=root/name;table.mkdir();rows=[]
            for path,data in items:
                source=table/path;source.parent.mkdir(parents=True,exist_ok=True);source.write_bytes(data)
                rows.append(dict(path=path,sha256=digest(source)))
            marker=dict(files=len(rows),identity=dict(host_root=str(table)),archive_sha256=hashlib.sha256(
                json.dumps(rows,sort_keys=True).encode()).hexdigest())
            (table/'INPUT-CLOSED.json').write_text(json.dumps(marker));return table,rows,marker
        for name in a.captures:
            path=a.run/'rollout'/name/'cohort'
            items=[(f'run/{name}/'+k,(path/k).read_bytes()) for k in ('SELECTED.json','GENERATION-CLOSED.json')]
            role='run/shared/SAMPLING.json'
            # Actual policy-invariant sampler material reused across these windows.
            items.append((role,(a.run.parent/'formal-inputs/SAMPLING.json').read_bytes()))
            table,rows,marker=window(name,items)
            publish(rows,table,inbox,['run'],marker);first=(inbox/role).stat().st_ino
            publish(rows,table,inbox,['run'],marker)
            assert first==(inbox/role).stat().st_ino
            rosters.append(first)
        assert len(set(rosters))==1
        bad,rows,marker=window('bad',[(role,b'changed contents')])
        try:publish(rows,bad,inbox,['run'],marker)
        except ValueError as e:assert 'conflicting' in str(e)
        else:raise AssertionError('overwrote data of a previous window')
        print(json.dumps(dict(windows=len(rosters),immutable_replay=True,reused_inode=True,conflict_rejected=True)))
if __name__=='__main__':main()
