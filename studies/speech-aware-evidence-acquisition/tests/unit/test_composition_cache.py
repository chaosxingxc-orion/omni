"""Repeated preparation reuses validation without hiding input changes."""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from speech_aware_evidence_acquisition.core import config


def inputs(root):
    for kind, name, value in [('model','m',{'model_id':'x'}),('dataset','d',{'source':'${SAEA_DATA_ROOT}/data'}),('baseline','b',{'nested':{'items':[1]}}),('experiment','e',{'experiment_id':'e'})]:
        path=root/'configs'/kind/(name+'.json');path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
    return (root,'m','d','b','e')


def test_identical_bytes_validate_once_and_return_isolated_values(tmp_path,monkeypatch):
    args=inputs(tmp_path);expected=config.compose(*args);cache=config.CompositionCache()
    original=config.assert_reasoning_knobs_are_well_formed;calls=[]
    def checked(v):calls.append(1);return original(v)
    monkeypatch.setattr(config,'assert_reasoning_knobs_are_well_formed',checked)
    first=cache.compose(*args);first.values['nested']['items'].append(2)
    first.fragments['model']='changed'
    assert cache.compose(*args)==expected
    assert len(calls)==1


def test_same_size_same_mtime_edit_and_deleted_input_not_hidden(tmp_path):
    args=inputs(tmp_path);cache=config.CompositionCache();old=cache.compose(*args)
    path=tmp_path/'configs/model/m.json';st=path.stat()
    path.write_text(path.read_text().replace('"x"','"y"'));os.utime(path,ns=(st.st_atime_ns,st.st_mtime_ns))
    assert cache.compose(*args).config_hash!=old.config_hash
    path.unlink()
    with pytest.raises(config.ConfigError):cache.compose(*args)


@pytest.mark.parametrize('body',['{"nested":1,"nested":2}','{"model_id":"collision"}','[]'])
def test_invalid_changed_fragment_is_rejected_without_poisoning_cache(tmp_path,body):
    args=inputs(tmp_path);cache=config.CompositionCache();old=cache.compose(*args)
    path=tmp_path/'configs/baseline/b.json';before=path.read_text();path.write_text(body)
    with pytest.raises(config.ConfigError):cache.compose(*args)
    path.write_text(before)
    assert cache.compose(*args)==old


def test_environment_and_explicit_data_root_are_part_of_identity(tmp_path,monkeypatch):
    args=inputs(tmp_path);cache=config.CompositionCache()
    for root in ['/one','/two']:
        monkeypatch.setenv('SAEA_DATA_ROOT',root)
        assert cache.compose(*args)==config.compose(*args)
        assert cache.compose(*args).values['source']==root+'/data'
    assert cache.compose(*args,data_root='/explicit')==config.compose(*args,data_root='/explicit')


def test_concurrent_calls_have_no_shared_mutable_result(tmp_path):
    args=inputs(tmp_path);cache=config.CompositionCache()
    def run(i):
        result=cache.compose(*args);result.values['nested']['items'].append(i)
        return result.values['nested']['items']
    with ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(run,range(24)))
    assert rows==[[1,i] for i in range(24)]
    assert cache.compose(*args)==config.compose(*args)
