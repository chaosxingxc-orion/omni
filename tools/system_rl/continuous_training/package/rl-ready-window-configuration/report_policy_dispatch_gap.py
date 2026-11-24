"""Report controller-clock wall time between a closed update and its successor dispatch."""
import argparse
import json
from pathlib import Path


def measure(run, step, cohort):
    def load(path):
        return json.loads(path.read_bytes())
    def at(path):
        return path.stat().st_mtime_ns
    policy = run / f'policy-{step:06}'
    update = run / f'step-{step:06}' / 'update/returned/CLOSED-COMMIT.json'
    config = load(cohort/'INPUT.json')
    actual = Path(config['policy']['current_ref']['path'])
    if (not load(update)['immutable'] or actual.resolve() != (policy/'POLICY.json').resolve()
        or load(actual)['step'] != step or not load(policy/'ACTIVATIONS-CLOSED.json')['terminal_verified']
        or not (cohort/'GENERATION-CLOSED.json').is_file()):
        raise ValueError('not a closed successor of the requested update')
    marks = [('checkpoint_return', at(update)), ('adapter_marker', at(policy/'MARKER.json')),
             ('activated', at(policy/'ACTIVATIONS-CLOSED.json')), ('selected', at(cohort/'SELECTED.json')),
             ('reserved', at(cohort/'RESERVED-CLOCK.json')), ('lease',at(cohort/'CAPTURE-LEASE.json')),
             ('first_annotation',min(map(at,cohort.glob('annotations/*.json')))),
             ('first_supply',min(map(at,cohort.glob('groups/*/episode-*/SUPPLY.json')))),
             ('first_dispatch',min(map(at,cohort.glob('groups/*/episode-*/native-journal/*.dispatch.json'))))]
    if any(b<a for (_,a),(_,b) in zip(marks,marks[1:])):
        raise ValueError('milestone order changed; examine evidence directly')
    return dict(step=step,cohort=cohort.parent.name,elapsed_s=(marks[-1][1]-marks[0][1])/1e9,
                clock='controller filesystem milestones; intervals include all intervening work',
                marks_ns=dict(marks),seconds={f'{a}_to_{b}':(t2-t1)/1e9
                for (a,t1),(b,t2) in zip(marks,marks[1:])})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('cohorts',nargs='+',type=Path)
    args=p.parse_args()
    results=[]
    for cohort in args.cohorts:
        spec=json.loads((cohort/'INPUT.json').read_bytes())
        policy=json.loads(Path(spec['policy']['current_ref']['path']).read_bytes())
        results.append(measure(args.run,policy['step'],cohort))
    print(json.dumps(results,indent=2))
