"""Read the deployed policy, compare the actual CPU request, and reject policy drift."""
import argparse, copy, json
from pathlib import Path
from repair import pm, canonical, sha
from export import Renderer


def check(settings,example,output):
    output.mkdir(parents=True,exist_ok=False)
    renderer=Renderer(settings)
    row=json.loads(example.read_bytes())
    audio=renderer.seed['wire']['messages'][1]['content'][0]
    packet=renderer.render(row,audio)
    assert [x['label'] for x in packet['ranking']]==row['supply']['dynamic_labels']+renderer.permanent
    rejected=[]
    for name,roster in [('missing',renderer.permanent[:-1]),('reordered',list(reversed(renderer.permanent))),('duplicate',renderer.permanent*2)]:
        cfg=copy.deepcopy(renderer.cfg); cfg['policy']['permanent_fixed_only']=roster
        path=output/(name+'.json');path.write_text(json.dumps(cfg))
        try:Renderer(path)
        except ValueError as exc:rejected.append(name)
        else:raise AssertionError(name)
    result=dict(permanent=renderer.permanent,wire_sha256=packet['wire_sha256'],rejected=rejected,model_calls=0)
    (output/'RESULT.json').write_text(json.dumps(result))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('settings',type=Path);p.add_argument('example',type=Path);p.add_argument('output',type=Path)
    print(check(**vars(p.parse_args())))
