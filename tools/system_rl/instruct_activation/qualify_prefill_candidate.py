"""Connect a measured candidate to the existing TP4 replay and restored FSDP8 scorer."""
from pathlib import Path
import subprocess
import sys
import time

from io_utils import doc, pin, require, save
from prepare_prefill_replay import prepare_panel
from run_prefill_replay import run as replay
from score_after_prefill_replay import run as score
from stage_prefill_score import stage


def qualify(c):
    deadline = time.monotonic() + c['wait_seconds']
    while not Path(c['decision']).is_file():
        require(time.monotonic()<deadline,'assessment has not finished')
        time.sleep(5)
    d = doc(pin(c['decision']))
    if d['action'] != 'retain_for_full_precision':
        require(d['action'] in ('measure_sibling','retain_for_divergence_analysis','no_promotion'), 'unexpected decision')
        return
    perf = doc(d['performance_lock'])
    packet = Path(c['packet'])
    diagnostic = Path(c['diagnostic'])
    root = packet.parent
    require(diagnostic.parent == root and doc(pin(diagnostic/'REPLAY-DELTA.json'))['parent_sha256'] ==
            perf['runtime_sha256'] and not perf['production_rollout'], 'wrong diagnostic runtime')
    natural = root/(packet.name+'-latency')/'CONFIG.local.json'
    cfg = doc(pin(natural))
    old = doc(pin(c['old_switch']))
    prior, = old['lanes']
    require(Path(prior['new_code']).parent==packet and Path(old['output']) ==
            Path(cfg['start_receipts'][c['lane']]).parent, 'wrong currently measured lane')
    lane = c['lane']
    panel = root/(diagnostic.name+'-panel')
    prepare_panel(dict(natural_config=str(natural),lane=lane,panel=str(panel)))
    source = panel/'SOURCE.json'
    actions = doc(pin(source))
    require(actions['round']['sha256']==perf['paired']['candidate_round'], 'wrong natural responses')
    stage_spec = doc(c['train_template'])
    stage_spec.update(panel_source=str(source),target=c['train_target'])
    stage(stage_spec)
    train = pin(Path(c['train_target'])/'TRAIN.local.json')
    engine = doc(pin(diagnostic/('lane-'+lane)/'config.json'))
    cp = root/('cutover-'+diagnostic.name)
    terminal = root/(packet.name+'-score')
    cut = dict(guard=old['guard'],output=str(cp),terminal=str(terminal/'DRAINED.json'),
               after_score=str(terminal/'RESULT.json'),retain_lanes=old['retain_lanes'],
               wait_seconds=old['wait_seconds'],lanes=[dict(old_code=prior['new_code'],old_attempt=prior['new_attempt'],
               new_code=str(diagnostic/('lane-'+lane)),new_attempt=c['attempt'],preflight=c['preflight'])])
    cutp = save(root/(diagnostic.name+'-switch.local.json'),cut)
    start = str(cp/('STARTED-'+str(engine['port'])+'.json'))
    rr = doc(c['measurement_template'])
    s, = rr['services']
    require(s['name']==lane and cfg['guards']==rr['guards'] and rr['grouped_concurrency']==[actions['width']],
            'unexpected replay protocol')
    s['container'] = engine['name']+'-'+c['attempt']
    rr.update(distinct_panels=[[r['replay'] for r in actions['rows']]], measurement_id=diagnostic.name,
              output=str(diagnostic/'rounds'))
    save(root/(diagnostic.name+'.json'),rr)
    run = doc(c['replay_run_template'])
    run.update(phases=[diagnostic.name],activate_on_first_warmup=dict(lane=lane,phase=diagnostic.name),
               start_receipts={lane:start})
    runp = save(root/(diagnostic.name+'-run.local.json'),run)
    replay(dict(cutover=cutp['path'],lock_config=c['lock_config'],lock_output=d['performance_lock']['path'],
                measurement=runp['path']))
    ctrl, = old['retain_lanes']
    b = c['control_lane']
    handoff=dict(rounds=str(diagnostic/'rounds'),train_config=train,wait_seconds=c['wait_seconds'],
                 restore_control=dict(attempt=c['restore_attempt'],code=ctrl['code'],preflight=c['restore_preflight']),
                 lanes=[dict(attempt=c['attempt'],code=cut['lanes'][0]['new_code'],started=start),
                        dict(ctrl,started=cfg['start_receipts'][b])])
    save(root/(diagnostic.name+'-hf-handoff.local.json'),handoff)
    score(handoff)
    out = save(root/(diagnostic.name+'-review.local.json'),dict(hf=train,source=str(source),
               natural_packet=str(packet),replay=str(diagnostic/'CONFIG.local.json'),lane=lane,limits=doc(pin(terminal/'RESULT.json'))['limits'],output=str(root/(diagnostic.name+'-full.json')),
               aggregate_output=str(root/(diagnostic.name+'-forward-public.json'))))
    subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('review_prefill_panel.py')),out['path']],check=True)


if __name__ == '__main__':
    qualify(doc(pin(sys.argv[1])))
