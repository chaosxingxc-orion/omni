"""Original fresh generation on175; same-action/reference finalization stays181."""
from pathlib import Path
import copy
import sys
import types
from dependencies import ROOT,require,load
from run_package import read,pin,write


def generate(request_ref,context):
    from fresh_adapter import scope
    source=read(read(request_ref)['input_ref'])
    if source.get('prepare_mode')=='domain_boundary_acquisition':
        from boundary_acquisition import generate as boundary_generate
        return boundary_generate(request_ref,context)
    with scope() as cohort:return _generate(request_ref,context,cohort)


def _generate(request_ref,context,cohort):
    request=read(request_ref);config=copy.deepcopy(read(request['input_ref']));root=Path(request['output_root'])/'cohort'
    if config.get('formal_startup'):
        require(config.get('diagnostic_evidence_only', {}).get('trainable') is False
                and config.get('prepare_mode') == 'finite_diagnostic_capture',
                'formal startup can only produce diagnostic evidence before full TRAIN screen')
    sys.path.insert(0,str(ROOT/'fresh-cohort-successor'))
    from fresh_adapter import policy as configure_policy
    configure_policy(load(ROOT/'fresh-cohort-successor/policy.py','policy'),allow_candidate_parity=request.get('extra',{}).get('purpose')=='updated_parity')
    config.update(output_root=str(root),cohort_id='dp8-'+request_ref['sha256'][:20])
    config['native']['attempt_id']=config['cohort_id']
    config['policy']['runtime_evidence_ref']=request['endpoint_witness_ref']
    config['capture']['endpoint_policy_ref']=request['endpoint_witness_ref']
    config['resources']['rank']=context['rank'];config['lease_fence']=context['lease_fence']
    config['authorization']=dict(config['authorization'],model_calls=True,native_training=False)
    # Source bytes remain pinned; index writes belong to this host's hot stage.
    from runtime_builder import bind_official_source
    require(config['capture'].get('official_source_config_ref'), 'prepared official source binding required')
    bind_official_source(config, root)
    # Original CaptureLease/DirectHTTP/ContactFence owns every actual request.
    from execution import CaptureLease
    cohort.prepare(config,root)
    capture=load(ROOT/'fresh-cohort-successor/cohort_capture.py','dp8_original_capture')
    with CaptureLease(config,root) as lease:
        from logits_scope import selected
        execution=read(context['execution_ref'])
        observation=selected(read(execution['owner_config_ref']),request)
        if observation is not None and 'selection' in observation:
            from logits_selection import bind_lease
            bind_lease(lease,observation,context)
        capture.capture(config,root,execution_enabled=True,lease=lease)
    return generation_result(root, request['endpoint_witness_ref'])


def generation_result(root, witness):
    pending=pin(root/'NATIVE.pending.disabled.json')
    closed=pin(root/'GENERATION-CLOSED.json')
    inventory=write(root/'GENERATION-INVENTORY.json',dict(closed=True,cohort_root=str(root),
        generation_closed_ref=closed,
        files={str(p.relative_to(root)):pin(p) for child in root.iterdir()
               if child.name not in ('environment','GENERATION-INVENTORY.json')
               for p in (child.rglob('*') if child.is_dir() else [child]) if p.is_file()}))
    return dict(endpoint_witness_ref=witness,input_ref=pin(root/'INPUT.json'),pending_capture_ref=pending,generation_inventory_ref=inventory,generation_closed_ref=closed)


def execute(request_ref):
    from rollout_owner import execute as owner
    return owner(request_ref=request_ref)


def check_lane_route(config, lane):
    """Check the same local tunnel the sampling client will use, before a POST intent."""
    from urllib.parse import urlsplit, urlunsplit
    from urllib.request import ProxyHandler, build_opener

    route = urlsplit(config['capture']['lane_transports'][lane]['local_endpoint'])
    require(route.scheme == 'http' and route.hostname in ('127.0.0.1', '::1')
            and route.port and not (route.username or route.password or route.query or route.fragment)
            and route.path == '/v1/chat/completions', 'invalid local TP4 route for ' + lane)
    health = urlunsplit((route.scheme, route.netloc, '/health', '', ''))
    with build_opener(ProxyHandler({})).open(health, timeout=5) as reply:
        require(reply.status == 200 and reply.geturl() == health, 'TP4 route not ready: ' + lane)


def bounded(config_ref, root, operation, draw_index=None, lane=None):
    """Run a configured window through the existing TP4 lanes."""
    from fresh_adapter import scope
    from runtime_builder import bind_official_source
    import fcntl
    root=Path(root)
    require(root.is_dir(),'preconfigured local hot root required')
    config=read(config_ref)
    formal=config['prepare_mode']=='full_train_screen'
    require((config['prepare_mode']=='bounded_train_probe' or
             (formal and config.get('sampled_window_ref')))
            and config['capture']['lanes']
            and config['output_root']==str(root),
            'explicit TP4 capture configuration required')
    with scope() as cohort:
        from cohort_io import environment, imports
        from execution import CaptureLease
        import cohort_capture as capture
        environment(root/'environment')
        lanes=config['capture']['lanes']
        if operation=='lane':
            if formal:
                require(draw_index is None and lane in lanes,'select a configured TP4 lane')
            else:
                require(lane is None and type(draw_index) is int and
                        0<=draw_index<len(lanes),'invalid bounded draw lane')
                lane=lanes[draw_index]
        else:
            require(lane is None and draw_index is None,'lane is only valid for collection')
        with (root/'COHORT.guard').open('a') as stream:
            shared=operation=='lane'
            fcntl.flock(stream,(fcntl.LOCK_SH if shared else fcntl.LOCK_EX)|fcntl.LOCK_NB)
            if operation=='prepare':
                from runtime_builder import model_padding_id
                if not formal:
                    with imports([ROOT/'collection-dispatch-integration/production-pipeline-next-version-v1']):
                        from bounded_train_probe import install
                        install(cohort)
                bind_official_source(config,root)
                consumer=read(config['consumer_config_ref'])
                consumer['pad_token_id']=model_padding_id(Path(config['native']['model_path']))
                config['consumer_config_ref']=write(root/'CAPTURE-CONSUMER.json',consumer)
                plan,_,_=cohort.prepare(config,root)
                from official import sources, annotation
                result=dict(prepared=len(plan['draws']),can_dispatch=True)
                static_rows=None
                if config['capture'].get('static_supply_ref'):
                    from transition_static_inputs import for_window
                    static_rows=for_window(config,plan)
                if static_rows is not None:
                    result['annotation_batch_ref']=write(root/'ANNOTATION-BATCH.json',dict(
                        selected_ref=pin(root/'SELECTED.json'),source_ref=config['capture']['official_source_config_ref'],
                        rows=[entry['annotation'] for entry in static_rows]))
                else:
                    with sources(config) as (official,_):
                        if formal and config['sampling'].get('node')=='P':
                            rows=[annotation(draw,official) for draw in plan['draws']]
                            result['annotation_batch_ref']=write(root/'ANNOTATION-BATCH.json', dict(
                                selected_ref=pin(root/'SELECTED.json'),
                                source_ref=config['capture']['official_source_config_ref'], rows=rows))
                        else:
                            for draw in plan['draws']:annotation(draw,official)
                write(root/'RUN-BOOT.json',dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
                return result
            config=read(root/'INPUT.json')
            plan=read(root/'SELECTED.json')
            if operation=='lane':
                # Keep the original numeric lane guard inode for both modes.
                with (root/f'lane-{lanes.index(lane)}.guard').open('a') as lane_lock:
                    fcntl.flock(lane_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    with CaptureLease(config,root) as lease:
                        check_lane_route(config,lane)
                        return capture.capture(config,root,execution_enabled=True,lease=lease,
                                               draw_index=draw_index,lane=lane if formal else None)
            require(operation=='close','unknown bounded capture operation')
            capture.close_generation(root,plan)
            return generation_result(root,config['policy']['runtime_evidence_ref'])


if __name__=='__main__':
    import argparse
    import json
    cli=argparse.ArgumentParser()
    cli.add_argument('--config',required=True)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--operation',choices=('prepare','lane','close'),required=True)
    route=cli.add_mutually_exclusive_group()
    route.add_argument('--draw-index',type=int)
    route.add_argument('--lane')
    args=cli.parse_args()
    print(json.dumps(bounded(pin(args.config),args.output,args.operation,args.draw_index,args.lane)))
