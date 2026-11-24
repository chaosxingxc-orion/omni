"""Exercise equal-size mutations under indistinguishable filesystem timestamps."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import sys
import transition_controller as ctl


def run():
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def read(ref):
        if sha(ref['path'])!=ref['sha256']:raise ValueError('bad commit')
        return json.loads(Path(ref['path']).read_bytes())
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);a=p/'lora_adapter';a.mkdir()
        names=['adapter_model.safetensors','adapter_config.json','PUBLICATION-DTYPE.json']
        for n in names:(a/n).write_bytes(b'orig')
        manifest=p/'commit';manifest.write_text(json.dumps({'ranks':[{'files':{'lora_adapter/'+n:sha(a/n) for n in names}}]}))
        ref={'path':str(manifest),'sha256':sha(manifest)};calls=[]
        def original(r,*,verify_shards):
            calls.append(verify_shards)
            for name,digest in read(r)['ranks'][0]['files'].items():
                if sha(p/name)!=digest:raise ValueError('bad export')
            return {'valid':True}
        mod=SimpleNamespace(checkpoint=original)
        with patch.dict(sys.modules,dependencies=SimpleNamespace(local_path=Path,sha=sha),run_package=SimpleNamespace(read=read)),patch.object(ctl,'signature',lambda path:(str(path),Path(path).stat().st_size)):
            ctl.install_checkpoint_cache(mod)
            mod.checkpoint(ref)['valid']=False
            assert mod.checkpoint(ref)['valid'] and calls==[False]
            mod.checkpoint(ref,verify_shards=True)
            assert calls==[False,True]
            for n in names:
                (a/n).write_bytes(b'evil')
                try:mod.checkpoint(ref)
                except ValueError:pass
                else:raise AssertionError('accepted changed '+n)
                (a/n).write_bytes(b'orig')
            assert mod.checkpoint(ref)['valid']
    return dict(equal_size_mutations_rejected=names,verify_shards_always_full=True,defensive_copy=True)


if __name__=='__main__':print(json.dumps(run()))
