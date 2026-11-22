"""Keep the measured graph engine, moving only lane identity and batch-invariance mode."""
import ast
import shutil
import sys
from pathlib import Path
from io_utils import pin, doc, save, storage, require


def argv_ast(source):
    return next(f for f in ast.parse(source).body if isinstance(f,ast.FunctionDef) and f.name=='argv')


def project(spec):
    src=Path(spec['parent']); dest=Path(spec['output']); storage(dest,2*1024**3)
    original=doc(pin(src/'PINS.json'))
    require(all(pin(src/k)['sha256']==v for k,v in original.items()),'parent drift')
    parent=doc(pin(src/'config.json'));layout=doc(pin(Path(spec['lane'])/'config.json'))
    require(type(spec['batch_invariant']) is bool and spec['batch_invariant'] and
            '--compilation-config' in parent['serve_args'] and 'batch_invariant' not in parent,'parent or mode invalid')
    fields={'name','cards','port','vllm_port','hccl_if_base_port','root'}
    require(all(layout.get(k)==v for k,v in parent.items() if k not in fields|{'serve_args'}),'lane not equivalent')
    new_source=Path(spec['launcher']); new_body=argv_ast(new_source.read_bytes())
    change,=[n for n in new_body.body if isinstance(n,ast.If) and ast.unparse(n.test)=="'batch_invariant' in C"]
    expected=ast.parse("""if 'batch_invariant' in C:
        assert type(C['batch_invariant']) is bool, 'batch_invariant must be a boolean'
        env['VLLM_BATCH_INVARIANT'] = '1' if C['batch_invariant'] else '0'
""").body[0]
    require(ast.dump(change)==ast.dump(expected),'unexpected environment override')
    new_body.body.remove(change)
    require(ast.dump(new_body)==ast.dump(argv_ast((src/'launch.py').read_bytes())),'container invocation differs')
    dest.mkdir(mode=0o700)
    for name in original:
        if name=='config.json':continue
        target=dest/name;target.parent.mkdir(exist_ok=True)
        shutil.copyfile(new_source if name=='launch.py' else src/name,target)
    cfg=parent|{k:layout[k] for k in fields}|{'root':str(dest),'batch_invariant':True}
    save(dest/'config.json',cfg)
    save(dest/'PINS.json',{k:pin(dest/k)['sha256'] for k in original})
    save(dest/'CANDIDATE.json',dict(parent=pin(src/'PINS.json'),launch=pin(dest/'launch.py'),
                                    runtime_changes={'VLLM_BATCH_INVARIANT':'1'}))

if __name__=='__main__':project(doc(pin(sys.argv[1])))
