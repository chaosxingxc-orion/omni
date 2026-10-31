import pytest
from tfrl_successor_prepare_queue import decision_state


def decision(passed):
    return {'schema':'saea-tfrl-first-construction-decision-v1','rows':625,
        'generation_authorized':False,'system_accepted':False,'preparation_gate_passed':passed,
        'decision':'prepare-held-successor' if passed else 'stop-family'}


def test_stop_is_a_named_terminal_and_pass_is_only_preparation():
    assert decision_state(decision(False))=='stopped-family'
    assert decision_state(decision(True))=='prepare'
    for key,value in [('rows',624),('generation_authorized',True),('system_accepted',True),('preparation_gate_passed',False)]:
        d=decision(True);d[key]=value
        with pytest.raises(ValueError):decision_state(d)


@pytest.mark.parametrize('passed',[False,True])
def test_queue_stops_or_stages_without_model_work(tmp_path,monkeypatch,passed):
    import json
    import tfrl_successor_prepare_queue as m
    monkeypatch.setattr(m,'D',tmp_path)
    tree=tmp_path/'tree';study=tree/'studies/speech-aware-evidence-acquisition'
    recipe=study/'docs/readiness/tfrl-current-recovered-consumers/analysis-recipe-v1.json'
    recipe.parent.mkdir(parents=True);recipe.write_text(json.dumps({'sources':{
        'h1':{'repo_relpath':'h1'},'h2':{'repo_relpath':'h2'}}}))
    for rid in (m.POOL,m.BANK):
        p=tmp_path/'analysis'/rid/'complete.json';p.parent.mkdir(parents=True);p.write_text('{}')
    monkeypatch.setattr(m.subprocess,'check_output',lambda args,**kw:
        '' if args[-1]=='--porcelain' else str(study) if args[-1]=='--show-toplevel' else 'a'*40)
    calls=[]
    class Child:
        pid=321
        def __init__(self,command):self.command=command
        def wait(self):
            from pathlib import Path
            target=Path(self.command[self.command.index('--out')+1])
            if '--decision-only' in self.command:
                target.write_text(json.dumps({**decision(passed),'construction_gain_pp':1.1 if passed else .5}))
            else:
                target.mkdir();(target/'complete.json').write_text(json.dumps({'rows':625,'iteration':2,
                    'readiness':'held','generation_authorized_by_this_file':False}))
            return 0
    def launch(command,**kwargs):
        assert kwargs['start_new_session']
        assert '--network none' in kwargs['env']['PORT_DOCKER_FLAGS']
        calls.append(command);return Child(command)
    monkeypatch.setattr(m.subprocess,'Popen',launch)
    out=tmp_path/'work/codex/queue';m.run(tree,tmp_path/'upstream',out)
    assert len(calls)==(2 if passed else 1)
    state=json.loads((out/'status.json').read_bytes())
    assert state['state']==('held-successor-prepared' if passed else 'stopped-family')
    assert state['generation_authorized'] is False
    assert (out/'held-successor').exists()==passed
