import json,subprocess
from pathlib import Path
import pytest
from speech_aware_evidence_acquisition.core.bundle import BundleError
from speech_aware_evidence_acquisition.rails import gepa_mutation_source as m


def historical(tmp_path):
    data=tmp_path/'data';repo=data/'history/study';repo.mkdir(parents=True)
    (repo/'source.json').write_text('{"historical": true}')
    subprocess.run(['git','init','-q',str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'add','.'],check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','source'],check=True)
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    return data,repo,{'root':'data','relpath':'history/study','commit':commit}


@pytest.mark.parametrize('fault',[None,'commit','dirty','escape','absolute','symlink','root','extra'])
def test_pinned_clean_historical_repository_is_separate_from_live(tmp_path,fault):
    data,old,pin=historical(tmp_path);live=tmp_path/'live';live.mkdir();(live/'source.json').write_text('{"current": true}')
    if fault=='commit':pin['commit']='a'*40
    elif fault=='dirty':(old/'source.json').write_text('changed')
    elif fault=='escape':pin['relpath']='../live'
    elif fault=='absolute':pin['relpath']=str(old)
    elif fault=='symlink':
        (data/'alias').symlink_to(old,target_is_directory=True);pin['relpath']='alias'
    elif fault=='root':pin['root']='repository'
    elif fault=='extra':pin['unbound']=True
    if fault:
        with pytest.raises((ValueError,BundleError)):m.source_repository({'historical_repository':pin},repo=live,data=data)
    else:
        assert m.source_repository({'historical_repository':pin},repo=live,data=data)==old
        assert m.source_repository({},repo=live,data=data)==live
        assert json.loads((old/'source.json').read_text())=={'historical':True}
