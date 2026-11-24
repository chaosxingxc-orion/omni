"""Run with the assembled release on PYTHONPATH; no services, devices or network needed."""
import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch
from source.config_profile import compile_profile, resolve
from run_package import read, write


def check(root):
    template={'host':'${TRAIN_HOST}','endpoint':'http://${TRAIN_HOST}:${PORT}',
              'devices':'${CARDS}','bundle':'${BUNDLE}','microbatch':8,'rank':128,'dropout':0.1}
    profile={'schema':'omni-local-profile-v2','environment':dict(TRAIN_HOST='host',PORT='port',CARDS='devices',HOT='path',BUNDLE='path'),
             'values':dict(TRAIN_HOST='trainer-a',PORT=8200,CARDS=[0,1],HOT='/local/a',BUNDLE='${HOT}/run')}
    t=write(root/'template.json',template);p=write(root/'profile.json',profile)
    env=dict(OMNI_DEPLOY_HOT='/local/b',OMNI_DEPLOY_TRAIN_HOST='trainer-b',OMNI_DEPLOY_PORT='8080',
             OMNI_DEPLOY_CARDS='[4,5]',OMNI_DEPLOY_RANK='1',UNDECLARED_VARIABLE='omit-this-value')
    def run(directory):
        receipt=compile_profile(t,p,root/directory)
        return receipt,read(read(receipt)['resolved_ref'])
    with patch.dict(os.environ,env):
        first,config=run('first')
        assert config==dict(host='trainer-b',endpoint='http://trainer-b:8080',devices=[4,5],bundle='/local/b/run',
                            microbatch=8,rank=128,dropout=0.1)
        assert 'omit-this-value' not in (root/'first/ENVIRONMENT.json').read_text()
    with patch.dict(os.environ,dict(OMNI_DEPLOY_PORT='9000',OMNI_DEPLOY_HOT='/local/c')):
        assert run('first')==(first,config)
        _,second=run('second')
        assert second['endpoint']=='http://trainer-a:9000' and second['bundle']=='/local/c/run' and second['devices']==[0,1]
    for i,(key,v) in enumerate([('PORT','65536'),('CARDS','[1,1]'),('PORT','true'),('HOT','relative'),('TRAIN_HOST','-bad')]):
        with patch.dict(os.environ,{'OMNI_DEPLOY_'+key:v}):
            try:run('invalid-'+str(i))
            except ValueError:pass
            else:raise AssertionError('invalid deployment was accepted: '+key)
    with patch.dict(os.environ,{'OMNI_DEPLOY_PORT':'1'}):
        legacy={'schema':'omni-local-profile-v1','values':{'PORT':8181}}
        assert resolve('${PORT}',legacy)==8181
        ref=compile_profile(write(root/'legacy.template.json','${PORT}'),write(root/'legacy.profile.json',legacy),root/'legacy')
        assert read(read(ref)['resolved_ref'])==8181
    changed=write(root/'changed.json',dict(profile,values=dict(profile['values'],PORT=10)))
    try:compile_profile(t,changed,root/'first')
    except ValueError:pass
    else:raise AssertionError('changed input reused the run')
    missing=dict(profile,values={})
    try:compile_profile(t,write(root/'missing.json',missing),root/'absent')
    except ValueError:pass
    else:raise AssertionError('undefined binding accepted')
    return write(root/'CHECK.json',dict(checks=['typed_environment','file_fallback','derived_path','unlisted_ignored',
                'frozen_resume','second_deployment','invalid_values','legacy_v1','changed_input_rejected','undefined_binding'],
                dispatch=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    print(json.dumps(check(p.parse_args().output)))
