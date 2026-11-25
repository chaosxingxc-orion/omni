"""Check every already closed training selection against the new static recording producer."""
import argparse,copy,json,sys
from pathlib import Path

def check(spec_path):
    spec=json.loads(spec_path.read_bytes());sys.path.insert(0,str(spec_path.parent))
    from transition_controller import main
    sys.argv=[sys.argv[0],str(spec_path),'--check'];main()
    from fresh_adapter import scope
    root=Path(spec['iteration_ref']['path']).parent.parent/'run'
    with scope():
        from transition_static_inputs import for_window
        from cohort_io import read
        macro=read(spec['recording_macros_ref']);n=0;pages=0
        for i,ref in enumerate(macro['pages']):
            page=read(ref);assert page['start']==i*macro['width'] and len(page['rows'])==min(macro['width'],macro['count']-page['start'])
            pages+=len(page['rows'])
        assert pages==macro['count']
        for batch in sorted((root/'rollout').glob('*-capture/cohort')):
            if not (batch/'GENERATION-CLOSED.json').is_file():continue
            cfg=json.loads((batch/'INPUT.json').read_bytes());plan=json.loads((batch/'SELECTED.json').read_bytes())
            cfg['capture'].update({k:spec[k] for k in ('static_supply_ref','prompt_length_index_ref','recording_macros_ref')})
            rows=for_window(cfg,plan)
            assert [r['annotation'] for r in rows]==json.loads((batch/'ANNOTATION-BATCH.json').read_bytes())['rows']
            n+=len(rows)
        assert n>0
        wrong=copy.deepcopy(plan);wrong['draws'][0]['dataset_draw_index']+=1
        try:for_window(cfg,wrong)
        except ValueError:pass
        else:raise ValueError('wrong offset accepted')
    return dict(recordings=pages,replayed=n,width=macro['width'],wrong_offset_rejected=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config',type=Path);a=p.parse_args();print(json.dumps(check(a.config)))
