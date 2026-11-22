"""Project the pinned native engine onto a sibling lane with bounded graph capture sizes."""
import json,shutil,sys
from pathlib import Path
from io_utils import pin,save,storage,require


def project(spec):
    root=Path(spec['output'])
    storage(root,2*1024**3)
    stock=Path(spec['stock']); lane=Path(spec['lane'])
    original=json.loads((stock/'PINS.json').read_bytes())
    require(all(pin(stock/name)['sha256']==h for name,h in original.items()),'stock drift')
    cfg=json.loads((stock/'config.json').read_bytes());layout=json.loads((lane/'config.json').read_bytes())
    fields={'root','name','cards','port','vllm_port','hccl_if_base_port'}
    require(all(layout[k]==v for k,v in cfg.items() if k not in fields|{'serve_args'}),'unexpected lane difference')
    sizes=spec['capture_sizes'];args=list(cfg['serve_args'])
    require('--enforce-eager' in args and '--compilation-config' not in args and
            sizes==sorted(set(sizes)) and sizes[0]==1 and sizes[-1]==int(args[args.index('--max-num-seqs')+1]),
            'bad candidate geometry')
    args.remove('--enforce-eager')
    args+=['--compilation-config',json.dumps(dict(cudagraph_mode=spec['graph_mode'],cudagraph_capture_sizes=sizes))]
    root.mkdir(mode=0o700)
    for name in original:
        if name=='config.json':continue
        f=root/name;f.parent.mkdir(exist_ok=True);shutil.copyfile(stock/name,f)
    cfg.update({k:layout[k] for k in fields}); cfg.update(root=str(root),serve_args=args)
    save(root/'config.json',cfg)
    save(root/'PINS.json',{name:pin(root/name)['sha256'] for name in original})
    save(root/'CANDIDATE.json',dict(parent=pin(stock/'PINS.json'),deployment=pin(lane/'config.json'),
        graph=spec['graph_mode'],capture_sizes=sizes,changed_engine_flags=['--enforce-eager','--compilation-config']))

if __name__=='__main__':project(json.loads(Path(sys.argv[1]).read_bytes()))
