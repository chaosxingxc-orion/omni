"""Explicit CPU recovery modules; never replace the frozen numerical tree."""
import argparse,hashlib,importlib.util,json,os,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--release',required=True);a=p.parse_args()
    release=Path(a.release);delta=json.loads((release/'DELTA.json').read_text())
    allowed={'parents/jev-system-rl/implementation/'+name for name in ('iteration.py','transport.py','pull_closed.py','resume_closed.py','cold_assets.py')}
    changed={r['target'] for r in delta['changes']}
    assert changed and changed<=allowed
    for row in delta['changes']:
        path=release/'tree'/row['target'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['result_ref']['sha256']
    # dependencies.HERE/ROOT and numerical SOURCE-PINS remain the original tree.
    sys.path.pop(0)
    import dependencies
    original=dependencies.HERE
    assert (Path(delta['base_assembly_ref']['path']).parent/'tree/parents/jev-system-rl/implementation').resolve()==original.resolve()
    reader=release/'tree/parents/jev-system-rl/implementation/pull_closed.py'
    os.environ['OMNI_CLOSED_READER_PATH']=str(reader)
    os.environ['OMNI_CLOSED_READER_SHA256']=hashlib.sha256(reader.read_bytes()).hexdigest()
    for name in ('cold_assets','transport','iteration'):
        if 'parents/jev-system-rl/implementation/'+name+'.py' not in changed:continue
        path=release/'tree/parents/jev-system-rl/implementation'/(name+'.py')
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    assert dependencies.HERE==original
    import iteration
    print(json.dumps(iteration.run(a.config)))

if __name__=='__main__':main()
