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
    resume = c.get('resume_before_first_request')
    if resume is None:
        stage(stage_spec)
    else:
        sealed = doc(pin(Path(c['train_target'])/'PANEL-SOURCE.json'))
        require(sealed['source']==pin(source) and sealed['reference']==stage_spec['template_config']
                and not (Path(c['train_target'])/'STOP-INTENT.json').exists(),
                'different staged training panel or handoff already started')
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
    # Exercise the actual imported planner before any expensive service cutover.
    from run_engine_comparison import main as measure
    from measure_existing_services import distinct_rounds
    plan = list(distinct_rounds(rr,rr['services']))
    require(len(plan)==1 and rr['decode_replay'] is True, 'isolated replay planner required')
    if resume is None:
        replay(dict(cutover=cutp['path'],lock_config=c['lock_config'],lock_output=d['performance_lock']['path'],
                    measurement=runp['path']))
    else:
        cfg_ref = pin(diagnostic/'CONFIG.local.json')
        active = doc(cfg_ref)
        require(resume['config']==cfg_ref and resume['start']==pin(start)
                and active['start_receipts']==run['start_receipts']
                and active['services'][0]['container']==s['container']
                and not Path(rr['output']).exists(), 'cannot resume after request intent or identity change')
        # The normal driver checks the live container ID, PID/start ticks, idle
        # metrics and guard. Immutable CONFIG must still match the resolved seed.
        measure(run)
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


def close_tied_wave(c):
    """Reuse the completed eight-rank forward; observe only the original tied wave."""
    from cutover import load
    from decode_action_replay import choice
    from io_utils import digest
    from project_replay_argmax import project
    from review_prefill_panel import raw_winner
    from run_engine_comparison import main as measure
    handoff=doc(c['handoff'])
    cfg=doc(handoff['train_config'])
    rounds=Path(handoff['rounds'])
    require(doc(pin(rounds/'DRAINED.json'))['guards_releasable'] and
            not list(rounds.rglob('UNKNOWN.json')), 'replay not drained')
    rr,=doc(pin(rounds/'RESULT.json'))['rounds']
    run=doc(rr)
    seed=doc(c['replay_seed'])
    lane,=seed['services']
    width,=seed['grouped_concurrency']
    require(run['concurrency_per_service']=={lane['name']:width}
            and seed['distinct_panels']==[run['input_refs']], 'not the original replay')
    targets,groups={},set()
    for r in run['requests']:
        ids,items=choice(r['receipt_ref'])
        pos=[j for j,item in enumerate(items) if raw_winner(item) is None]
        if pos:
            import json
            prompt=json.loads(doc(r['receipt_ref'])['raw'])['prompt_token_ids']
            key=digest([prompt,ids])
            require(key not in targets,'duplicate tied history')
            targets[key]=pos
            groups.add(r['group'])
    group,=groups  # Each observer spec covers one entire original wave.
    root=rounds.parent.parent
    proj=doc(c['replay_project'])
    packet=root/c['packet_name']
    require(packet.parent==root and Path(proj['projection']['target'])==rounds.parent,'wrong replay project')
    proj.update(original_replay=str(rounds.parent),observe=save(root/(packet.name+'-targets.json'),targets))
    proj['projection'].update(target=str(packet),attempt=c['attempt'])
    if not packet.exists():
        project(proj)
    else:
        from decode_lineage import changes,snapshot
        prior=snapshot(rounds.parent)
        delta=doc(pin(packet/'ARGMAX-DELTA.json'))
        require(delta['parent']==digest(prior) and delta['observe']==proj['observe']
                and changes(prior,snapshot(packet))==delta['changes']
                and not (packet/'rounds').exists() and not (packet/'STARTED-A.json').exists(),
                'existing observer requires reconciliation')
    lane['container']=doc(pin(packet/('lane-'+lane['name'])/'config.json'))['name']+'-'+c['attempt']
    seed.update(distinct_panels=[run['input_refs'][group*width:(group+1)*width]],
                measurement_id=packet.name,output=str(packet/'rounds'))
    save(root/(packet.name+'.json'),seed)
    end=time.monotonic()+c['wait_seconds']
    trainroot=Path(cfg['hot']).parent
    initial=Path(c['initial_review'])
    while not initial.is_file():
        require(time.monotonic()<end,'initial review not sealed')
        time.sleep(5)
    review=doc(pin(initial))
    require(review['hf']==handoff['train_config'] and Path(review['replay']).parent==rounds.parent,'wrong review')
    public=Path(review['aggregate_output'])
    while not public.is_file():
        require(time.monotonic()<end,'initial audit not sealed')
        time.sleep(3)
    audit=doc(pin(public))
    require(audit['replay_round']==rr['sha256'] and audit['replay_changed']==0
            and audit['unresolved_top1']==sum(map(len,targets.values())) and all(
            x['over_token']==x['over_sequence']==x['top1_flips']==0 for x in audit['modes'].values()),
            'probability or top-1 mismatch needs independent localization')
    stopped=doc(pin(trainroot/'INFERENCE-DRAINED.json'))
    require({r['cid'] for r in stopped['stopped']}=={doc(pin(l['started']))['container_id'] for l in handoff['lanes']}
            and doc(pin(trainroot/'STOP-INTENT.json'))['source']==pin(rounds/'DRAINED.json'),'unrelated handoff')
    worker=doc(pin(Path(cfg['hot'])/'EXIT.json'))
    require(worker['complete_ranks']==8 and worker['exit_code']==0,'training scoring incomplete')
    service=load(packet/('lane-'+lane['name']),'observer')
    ws,=json.loads(subprocess.check_output(['docker','inspect',worker['container_id']]))
    require(service.C['host_boot_id']==cfg['boot_id'] and service.C['guard']==cfg['existing_guard']
            and ws['Id']==worker['container_id'] and ws['Image']==cfg['image']
            and ws['State']['Status']=='exited','training container still active or identity changed')
    ctrl=handoff['restore_control']
    control=load(ctrl['code'],'control')
    state=control.inspect(control.name(ctrl['attempt']))
    require(state['State']['Running'] and state['Id']==doc(pin(trainroot/'CONTROL-STARTED.json'))['container_id']
            and sorted(service.C['cards']+control.C['cards'])==cfg['devices'],'unverified control restoration')
    launch=[sys.executable,'-B',str(service.CODE/'launch.py')]
    subprocess.run(launch+['check','--attempt',c['preflight']],check=True)
    subprocess.run(launch+['start','--attempt',c['attempt'],'--preflight-attempt',c['preflight']],check=True)
    state=service.inspect(lane['container'])
    receipt=save(packet/'STARTED-A.json',dict(container_id=state['Id'],physical_devices=service.C['cards']))
    m=doc(c['replay_run'])
    m.update(phases=[packet.name],activate_on_first_warmup=dict(lane=lane['name'],phase=packet.name),
             start_receipts={lane['name']:receipt['path']})
    measure(m)
    review.update(argmax_observer=dict(packet=str(packet),group_index=group,vocab_size=c['vocab_size']),
                  output=str(root/(packet.name+'-full.json')),aggregate_output=str(root/(packet.name+'-forward-public.json')))
    spec=save(root/(packet.name+'-review.local.json'),review)
    subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('review_prefill_panel.py')),spec['path']],check=True)


if __name__ == '__main__':
    config=doc(pin(sys.argv[1]))
    (close_tied_wave if config.get('operation')=='close_tied_wave' else qualify)(config)
