import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

p=Path(__file__).parents[2]/'scripts/environment/configured_asset_stage.py'
s=importlib.util.spec_from_file_location('asset_stage',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def test_readable_copy_retains_exact_hash_and_refuses_repeat(tmp_path):
    src=tmp_path/'source';src.write_bytes(b'original scientific bytes');src.chmod(0o600)
    pin={'source':str(src),'relative':'knowledge/store.json','sha256':hashlib.sha256(src.read_bytes()).hexdigest()}
    cfg=tmp_path/'config';out=tmp_path/'state';cfg.write_text(json.dumps({'schema':'saea-asset-stage-v1','output_root':str(out),'files':[pin]}))
    result=m.execute(cfg)
    assert result['new_model_calls']==0 and (out/pin['relative']).read_bytes()==src.read_bytes()
    assert (out/pin['relative']).stat().st_mode & 0o777 == 0o644
    with pytest.raises(FileExistsError):m.execute(cfg)


@pytest.mark.parametrize('fault',['hash','escape','duplicate'])
def test_invalid_copy_leaves_no_output(tmp_path,fault):
    src=tmp_path/'source';src.write_bytes(b'bytes');out=tmp_path/'state'
    pin={'source':str(src),'relative':'knowledge/store.json','sha256':hashlib.sha256(src.read_bytes()).hexdigest()}
    if fault=='hash':pin['sha256']='a'*64
    if fault=='escape':pin['relative']='../escaped'
    cfg=tmp_path/'config';cfg.write_text(json.dumps({'schema':'saea-asset-stage-v1','output_root':str(out),'files':[pin,pin] if fault=='duplicate' else [pin]}))
    with pytest.raises(ValueError):m.execute(cfg)
    assert not out.exists()
