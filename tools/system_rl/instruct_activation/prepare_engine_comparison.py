"""Prepare isolated stock/overlay services for a device-level engine comparison."""
import json,shutil,sys
from pathlib import Path
from decode_lineage import digest
from io_utils import save,storage

def prepare(c):
    parent=Path(c['parent']);root=Path(c['output']);storage(root,2*1024**3)
    assert not (root/'SWITCH.json').exists()
    if c.get("stock_lora_dtype") not in ("auto","bfloat16","float16"):
        raise ValueError("stock_lora_dtype must be supported by the native vLLM CLI")
    pairs=[]
    for lane,kind in c['arms'].items():
        assert kind in ('stock','overlay')
        src=parent/('lane-'+lane);dst=root/src.name;dst.mkdir()
        for k,v in json.loads((src/'PINS.json').read_bytes()).items():assert digest(src/k)==v
        cfg=json.loads((src/'config.json').read_bytes())
        cfg.update(root=str(dst))
        # Stock reads the exact published PEFT files from the already read-only activation mount.
        if kind=='stock':
            overlay=cfg.pop('policy_overlay'); adapter=Path(overlay['bundle'])/c['adapter_relative']
            assert digest(adapter/'adapter_model.safetensors')==c['adapter_hash']
            (dst/'adapter').mkdir()
            for f in ('adapter_model.safetensors','adapter_config.json'):shutil.copyfile(adapter/f,dst/'adapter'/f)
            args=cfg['serve_args'];i=args.index('--worker-extension-cls');del args[i:i+2]
            args[args.index('--lora-dtype')+1]=c['stock_lora_dtype']
            args.extend(['--lora-modules',json.dumps(dict(name=c['adapter_alias'],path='/activation/adapter',base_model_name=cfg['served_model_name']))])
        cfg['serve_args']+=['--profiler-config',json.dumps(c['profiler'])]
        for f in ('launch.py','service.py'):shutil.copyfile(src/f,dst/f)
        if kind=='overlay':
            # launcher requires server under bundle; reuse original identical service name via a fresh
            # copied activation bundle whose immutable assets are verified from the frozen packet.
            from project_screen_lanes import project
            p=c['projection']
            project(p,Path(c['lock']))
            prep=Path(p['target'])/('lane-'+lane)
            cfg['policy_overlay']=json.loads((prep/'config.json').read_bytes())['policy_overlay']
        save(dst/'config.json',cfg)
        pins={str(f.relative_to(dst)):digest(f) for f in dst.rglob('*') if f.is_file()}
        save(dst/'PINS.json',pins)
        pairs.append(dict(old_code=str(src),new_code=str(dst),old_attempt=c['old_attempt'],new_attempt=c['attempt'],preflight=c['check_attempt']))
    save(root/'SWITCH.json',dict(output=str(root/'cutover'),guard=cfg['guard'],terminal=c['terminal'],lanes=pairs))
if __name__=='__main__':prepare(json.loads(Path(sys.argv[1]).read_bytes()))
