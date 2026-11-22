"""Wire an isolated successor to an existing matched C8 panel and its retained control."""
from pathlib import Path
import sys
from bind_cold_parent_panel import bind
from derive_paired_study import prepare
from io_utils import doc, pin, require, save


def stage(c):
    r=Path(c['root']).resolve(strict=True)
    old,new=c['predecessor'],c['study']
    template=c.get('assessment_template', old)
    require(all(Path(n).name==n and n not in ('.','..','') for n in (old,new,template)) and old!=new, 'bad study names')
    def path(s): return str(r/(new+s))
    def get(n,s): return doc(pin(r/(n+s)))
    def put(s,v): return save(path(s),v)
    src=get(new,'-project.local.json')
    require(src['projection']['target']==path('') and src['projection']['attempt']==c['attempt'],'unexpected packet')
    bind(dict(root=str(r),source_panel=old+'-panel',output_panel=new+'-panel'))
    prepare(dict(root=str(r),predecessor=old,study=new,attempt=c['attempt'],
                 lane=c['lane'],preflight=c['preflight'],batch_size=c['width']))
    prior=get(old,'-restored-switch.local.json')
    cut=get(new,'-switch.local.json');cut['retain_lanes']=prior['retain_lanes']
    if 'native_diagnostic' in c:
        diagnostic = Path(c['native_diagnostic'])
        owner = doc(pin(diagnostic/'CUTOVER.json'))
        require(owner['guard'] == cut['guard'] and owner['retain_lanes'] == cut['retain_lanes'],
                'different diagnostic guard or control')
        active, = owner['lanes']
        slot, = cut['lanes']
        require(active['old_code'] == slot['old_code'] and active['old_attempt'] == slot['old_attempt'],
                'diagnostic did not replace predecessor')
        slot.update(old_code=active['new_code'], old_attempt=active['new_attempt'])
        measured = doc(pin(diagnostic/'MEASUREMENT.json'))
        latency = r/measured['phases'][-1]
        cut.pop('after_score')
        cut.update(terminal=str(latency/'rounds/DRAINED.json'), after_iterations=pin(diagnostic/'ITERATIONS.json'),
                   diagnostic_lane=c['lane'], diagnostic_receipt=pin(measured['start_receipts'][c['lane']]))
    switch=put('-restored-switch.local.json',cut)
    control=c['control_lane']
    run=get(new,'-run.local.json')
    for phase in run['phases']:
        cfg=doc(pin(r/(phase+'.json')))
        ctrl,=[s for s in cfg['services'] if s['name']==control]
        prev,=[s for s in get(old,phase[len(new):]+'-restored.local.json')['services'] if s['name']==control]
        require(ctrl['endpoint']==prev['endpoint'],'control endpoint drift')
        ctrl['container']=prev['container']
        run.setdefault('phase_specs',{})[phase]=save(r/(phase+'-restored.local.json'),cfg)
    run['start_receipts'][control]=get(old,'-restored-run.local.json')['start_receipts'][control]
    meas=put('-restored-run.local.json',run)
    if c.get('diagnostic_only'):
        require('native_diagnostic' not in c, 'diagnostic successor expects a scored candidate')
        flow=get(new,'-flow.local.json')
        flow.update(cutover=switch['path'],measure=meas['path'],
                    after_no_promotion=str(r/(old+'-native-DECISION.json')))
        put('-conditional-flow.local.json',flow)
        return
    spec=get(template,'-parent-trained-report.local.json')
    ls=get(template,'-native-lock.local.json')
    if template != old:
        from decode_lineage import snapshot
        from io_utils import digest
        timing=get(old,'-project.local.json')
        proof=doc(spec['training_proof'])
        require(timing['timing_parent']==ls['parent_packet'] and
                spec['parent_config']==ls['parent'] and proof['passed'] and
                digest(snapshot(Path(ls['parent_packet'])))==proof['runtime'] and
                doc(pin(timing['graph_lock']))['runtime_sha256']==proof['runtime'],
                'diagnostic and assessment template must share the verified runtime')
    spec.update(candidate_config=path('-latency/CONFIG.local.json'),historical_score=path('-score'),output=path('-cold-train.json'))
    report=put('-parent-trained-report.local.json',spec)
    flow=get(new,'-flow.local.json')
    flow.update(cutover=switch['path'],measure=meas['path'],parent_forward_report=report)
    put('-restored-flow.local.json',flow)
    if 'native_diagnostic' not in c and template == old:
        flow['after_no_promotion']=str(r/(old+'-native-DECISION.json'))
        put('-conditional-flow.local.json',flow)
    ls.pop('change_scope',None)
    ls.pop('components',None)
    if 'prefill_page_component' in src:
        ls['components']=pin(r/new/'COMPONENTS.json')
    ls.update(candidate=spec['candidate_config'],candidate_packet=path(''),version=c['version'],
              expected_engine_delta={},changed_file=c['changed_file'],changed_row=c['changed_row'])
    lock=put('-native-lock.local.json',ls)
    ac=get(template,'-native-assessment.local.json')
    ac.update(lock=lock,report=report,lock_output=path('-native.lock.json'),decision=path('-native-DECISION.json'))
    put('-native-assessment.local.json',ac)
    replay=get(template,'-replay-project.local.json')
    for k in ('verified_replay_packet','replay_source'): src[k]=replay[k]
    src.update(replay_runtime=c['replay_runtime'],measured_candidate=path(''))
    src['projection'].update(target=path('-replay'),attempt=c['replay_attempt'])
    q=get(template,'-native-qualify.local.json')
    q.update(packet=path(''),diagnostic=path('-replay'),old_switch=switch['path'],lock_config=lock['path'],
             decision=ac['decision'],attempt=c['replay_attempt'],preflight=c['replay_preflight'],
             restore_attempt=c['restore_attempt'],train_target=c['train_target'],
             replay_project=put('-replay-project.local.json',src))
    put('-native-qualify.local.json',q)


if __name__=='__main__':
    stage(doc(pin(sys.argv[1])))
