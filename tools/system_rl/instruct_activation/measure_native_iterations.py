"""Measure native engine iterations on a cold copy using the existing cutover and panel."""
from pathlib import Path
from statistics import median
import re
import shutil
import subprocess
import sys

from cutover import switch
from io_utils import doc, pin, require, save, storage
from run_engine_comparison import main as measure
from window_iterations import report


def run(spec):
    stock = Path(spec['stock'])
    study = Path(spec['output'])
    predecessor = Path(spec['predecessor'])
    previous = doc(pin(predecessor.with_name(predecessor.name + '-restored-switch.local.json')))
    old, = previous['lanes']
    retained, = previous['retain_lanes']
    require(Path(retained['code']) == stock and Path(old['new_code']).parent == predecessor,
            'native or candidate identity differs')
    decision = doc(pin(predecessor.with_name(predecessor.name + '-native-DECISION.json')))
    require(decision['action'] == 'no_promotion', 'candidate still in use')
    require(study.parent == predecessor.parent and re.fullmatch(r'[a-z0-9-]+', study.name),
            'unexpected study directory')
    parent_pins = doc(pin(stock/'PINS.json'))
    cfg = doc(pin(stock/'config.json'))
    previous_lane = doc(pin(Path(old['new_code'])/'config.json'))
    require(all(pin(stock/name)['sha256'] == h for name,h in parent_pins.items()), 'native drift')
    require('--compilation-config' in cfg['serve_args'] and '--enable-logging-iteration-details' not in cfg['serve_args'],
            'expected original graph runtime')
    code = study/('lane-' + spec['lane'])
    storage(study, cfg['runtime_budget_bytes'])
    study.mkdir(mode=0o700)
    code.mkdir(mode=0o700)
    for name in parent_pins:
        if name != 'config.json':
            dst = code/name
            dst.parent.mkdir(exist_ok=True)
            shutil.copyfile(stock/name, dst)
    for key in ('name','cards','port','vllm_port','hccl_if_base_port'):
        cfg[key] = previous_lane[key]
    cfg.update(root=str(code), serve_args=[*cfg['serve_args'], '--enable-logging-iteration-details'])
    save(code/'config.json', cfg)
    pins = {name:pin(code/name)['sha256'] for name in parent_pins}
    require(all(pins[k] == parent_pins[k] for k in pins if k != 'config.json'), 'changed native code or adapter')
    save(code/'PINS.json', pins)
    cut_path = study/'cutover'
    score = predecessor.with_name(predecessor.name + '-score')
    cut = dict(previous, lanes=[dict(old_code=old['new_code'], old_attempt=old['new_attempt'],
               new_code=str(code), new_attempt=spec['attempt'], preflight=spec['preflight'])],
               terminal=str(score/'DRAINED.json'), after_score=str(score/'RESULT.json'), output=str(cut_path))
    prior_run = doc(pin(predecessor.with_name(predecessor.name + '-restored-run.local.json')))
    receipt = cut_path/('STARTED-' + str(cfg['port']) + '.json')
    run_cfg = dict(prior_run, phases=[], phase_specs={}, start_receipts=dict(prior_run['start_receipts']))
    run_cfg.pop('activate_on_first_warmup')
    run_cfg['start_receipts'][spec['lane']] = str(receipt)
    for phase in prior_run['phases']:
        require(phase.startswith(predecessor.name+'-'), 'unexpected panel owner')
        name = study.name + phase[len(predecessor.name):]
        config = doc(prior_run['phase_specs'][phase])
        lane, = [arm for arm in config['services'] if arm['name'] == spec['lane']]
        lane.update(stock_adapter_alias=run_cfg['adapter_alias'], container=cfg['name']+'-'+spec['attempt'])
        config.update(output=str(study.parent/name/'rounds'), measurement_id=name)
        run_cfg['phases'].append(name)
        run_cfg['phase_specs'][name] = save(study/(name+'.json'), config)
    require(run_cfg['phases'] == [study.name+'-warm', study.name+'-latency'], 'wrong phase order')
    save(study/'CUTOVER.json', cut)
    save(study/'MEASUREMENT.json', run_cfg)
    subprocess.run([sys.executable,'-B',str(code/'launch.py'),'check','--attempt',spec['preflight']],check=True)
    switch(cut)
    measure(run_cfg)
    window = study.with_name(run_cfg['phases'][-1])
    evidence = report(window, spec['lane'], receipt)
    save(study/'ITERATIONS.json', evidence)
    widths = {}
    for row in evidence['rows']:
        if row['kind'] == 'decode':
            widths.setdefault(row['generation_tokens'], []).append(row['elapsed_ms'])
    save(study/'WIDTHS.json', dict(round=evidence['round_sha256'], intervals=evidence['intervals'],
         widths={str(width):dict(steps=len(times), median_ms=median(times)) for width,times in sorted(widths.items())},
         scope='All native pure decode steps. Natural generation and original iteration timer.'))


if __name__ == '__main__':
    run(doc(pin(sys.argv[1])))
