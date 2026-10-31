import pytest
from tfrl_current_queue import run_lanes


@pytest.mark.parametrize('failure',[None,'gen1','read1','gen2'])
def test_completion_drives_overlap_without_replay(failure):
    events=[]
    class Child:
        def __init__(self,name):self.name=name
        def wait(self):
            events.append('wait-'+self.name)
            return int(self.name==failure)
    def start(prefix,job):
        name=prefix+job;events.append('start-'+name);return Child(name)
    call=lambda:run_lanes(['1','2'],lambda j:start('gen',j),lambda j:start('read',j),lambda *a,**k:None)
    if failure:
        with pytest.raises(RuntimeError):call()
    else:call()
    assert events.count('start-gen1')==1
    if failure=='gen1':assert events==['start-gen1','wait-gen1']
    else:
        assert events.index('start-read1')<events.index('start-gen2')<events.index('wait-read1')
        assert events.count('start-gen2')==1
        assert ('start-read2' in events)==(failure!='gen2')
