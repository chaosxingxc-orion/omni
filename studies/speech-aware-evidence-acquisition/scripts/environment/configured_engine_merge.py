#!/usr/bin/env python3
"""Merge idle resident lanes using pinned profiles and existing deployment checks.

This is a separate operation from the context-only engine transition. Every old
guard remains in place; retired lanes keep a pending marker to reject stale jobs.
"""
import argparse
import copy
from contextlib import ExitStack
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import configured_engine_transition as engine

require, read, pin = engine.require, engine.read, engine.pin
publish, inspect, deployment = engine.publish, engine.inspect, engine.deployment
SCHEMA = 'saea-configured-engine-merge-v1'


def geometry(old_profiles, target, protected_cards, protected_ports):
    cards, ports = set(), set()
    for old in old_profiles:
        require(len(old.instances) == 1, 'one instance per source required')
        require(not cards.intersection(old.cards), 'source cards overlap')
        require(old.instances[0].port not in ports, 'source ports overlap')
        cards.update(old.cards); ports.add(old.instances[0].port)
    require(len(old_profiles) >= 2 and len(target.instances) == 1, 'multiple sources and one target required')
    require(cards == set(target.cards), 'target must exactly cover source cards')
    require(not cards.intersection(protected_cards) and not ports.intersection(protected_ports)
            and target.instances[0].port not in protected_ports, 'protected resource overlap')
    require(target.instances[0].port in ports, 'target must reuse an explicitly retired port')
    require(all(p.image == target.image and p.model_path == target.model_path
                and p.effective_namespace_policy == target.effective_namespace_policy == 'shared-pid-host'
                for p in old_profiles), 'same pinned image, weights and host PID required')
    require(target.image.startswith('sha256:') and len(target.image) == 71, 'fixed digest required')


def load(path):
    d = read(path)
    require(d['schema'] == SCHEMA, 'merge schema differs')
    sources = []
    for s in d['sources']:
        profile = deployment.load_deployment_profile(pin(s['profile']))
        model, receipt = read(pin(s['model'])), read(pin(s['runtime_receipt']))
        launch = engine.launch.load_config(pin(s['producer_launch']))
        require(s['locks'] == str(engine.settlement.mapped(launch, Path(d['data_root'])/'locks', writable=True)),
                'source guard must be actual producer guard')
        sources.append(dict(s, profile_value=profile, model_value=model, receipt_value=receipt, producer=launch))
    target = deployment.load_deployment_profile(pin(d['new_profile']))
    model = read(pin(d['new_model']))
    geometry([s['profile_value'] for s in sources], target, d['protected_cards'], d['protected_ports'])
    native = read(pin(d['native_model_config']))['thinker_config']['text_config']
    require(engine.context(model['server_args']) <= native['max_position_embeddings']
            and native.get('rope_scaling', {}).get('rope_type', 'default') == 'default', 'native context exceeded')
    for s in sources:
        old = s['model_value']
        require(old['server_args'] == model['server_args'] and old['model_lock_key'] == model['model_lock_key']
                and old['model_materialization_root'] == model['model_materialization_root'], 'model protocol changed')
        require(s['receipt_value']['protocol']['server_args'] == model['server_args'], 'source receipt protocol differs')
        i, = s['receipt_value']['deployment']['instances']
        require(i['device_ids'] == list(s['profile_value'].cards) and i['port'] == s['profile_value'].instances[0].port,
                'source receipt geometry differs')
    require(len({s['container_id'] for s in sources}) == len(sources), 'duplicate old container')
    require(d['new_session'] not in [s['session'] for s in sources], 'fresh target session required')
    return d, sources, target, model


def terminal_producer(s):
    p = s['producer']; root = Path(p['receipt_dir'])
    intent, result = read(root/'launch-intent.json'), read(root/'launch-result.json')
    require(intent['config_sha256'] == engine.queue.canonical_hash(p)
            and intent['expected_commit'] == p['expected_commit'] and result['returncode'] == 0,
            'producer launch custody differs')
    c = inspect(result['stdout'].strip())
    require(c is not None and c['Name'] == '/'+p['container_name'] and c['Image'] == p['image']
            and c['Config']['Labels']['saea.launch.config'] == intent['config_sha256']
            and c['Config']['Labels']['saea.launch.runner'] == intent['runner_sha256'], 'producer identity differs')
    require(c['State']['Status'] == 'exited' and not c['State']['Running'] and c['State']['Pid'] == 0,
            'producer still active or unknown')
    return {'container_id': c['Id'], 'state': c['State'], 'launch_intent_sha256': engine.launch.digest(root/'launch-intent.json'),
            'launch_result_sha256': engine.launch.digest(root/'launch-result.json')}


def custody(s, logical):
    """Validate all writable source journals, including FIT and failed ancestors."""
    from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal
    p = s['producer']; results = []
    for mt in p['mounts']:
        rel = Path(mt['target'])
        if mt['readonly'] or not rel.is_relative_to(logical/'contact-journals'):
            continue
        root = Path(mt['source'])
        require(root.exists() and not root.is_symlink(), 'source journal directory missing')
        for attempt_dir in sorted(root.iterdir()):
            require(attempt_dir.is_dir() and not attempt_dir.is_symlink(), 'unexpected journal entry')
            run = root.name; aid = attempt_dir.name
            ap = engine.settlement.mapped(p, logical/'attempts'/run/(aid+'.json'))
            attempt = read(ap)
            require(attempt['state'] in ('ABORTED', 'FINALIZED') and attempt['run_id'] == run
                    and attempt['attempt_id'] == aid, 'nonterminal or mismatched attempt')
            files = sorted(attempt_dir.iterdir()); intents = sorted(attempt_dir.glob('*.intent.json'))
            require(len(files) == 2*len(intents) and len(intents) == attempt['actual']['calls'],
                    'unknown or orphan request count')
            contacts = []
            identity = None
            for f in intents:
                item = read(f)
                if identity is None: identity = item['identity']
                require(item['identity'] == identity and identity['run_id'] == run and identity['attempt_id'] == aid,
                        'contact attempt identity differs')
                journal = object.__new__(ContactJournal); journal.root = attempt_dir; journal.identity = identity
                rid = item['request_id']
                require(f == journal._path(rid, 'intent') and journal.inspect(rid)['state'] == 'observed',
                        'unknown source request')
                response = journal._path(rid, 'response')
                require(not response.is_symlink(), 'redirected response')
                contacts.append({'request_id': rid, 'intent_sha256': engine.launch.digest(f),
                                 'response_sha256': engine.launch.digest(response)})
            results.append({'run_id': run, 'attempt_id': aid, 'attempt_path': str(ap),
                            'attempt_sha256': engine.launch.digest(ap), 'state': attempt['state'],
                            'actual': attempt['actual'], 'contacts': contacts})
    require(results, 'source journal custody required')
    return results


def sessions(s):
    saved = {}
    for name in (s['session'], s['profile_value'].instances[0].name):
        for leaf in ('session.json', 'session-receipt.json'):
            f = Path(s['serving'])/name/leaf; saved[str(Path(name)/leaf)] = read(f)
        require(saved[str(Path(name)/'session.json')]['identity']['runtime_receipt_sha256']
                == s['runtime_receipt']['sha256'], 'old session runtime differs')
    return saved


def source_audit(s, logical, *, live=True):
    terminal = terminal_producer(s)
    contacts = custody(s, logical)
    old = engine.verify_container(inspect(s['container_id']), s['profile_value'], s['model_value'],
                                  exact_id=s['container_id'], require_running=live)
    saved = sessions(s)
    if live:
        engine.verify_live_old({'receipt':s['receipt_value'], 'expected':{'service_session':s['session']}}, old, saved)
    return {'producer':terminal, 'custody':contacts, 'old_container':old, 'old_sessions':saved}


def settle_lease(s, proof, out):
    lease = Path(s['locks'])/'flight.lock'
    if not os.path.lexists(lease): return {'lease':'absent'}
    blob = lease.read_bytes(); record = json.loads(blob)
    require(type(record.get('pid')) is int and not Path('/proc',str(record['pid'])).exists(), 'lease owner still exists')
    require(record['session_id'] == s['session'] and record['study_head'] == s['producer']['expected_commit'],
            'lease producer or session differs')
    matches = [x for x in proof['custody'] if x['run_id'] == record['run_id'] and x['attempt_id'] == record['attempt_id']]
    require(len(matches) == 1, 'lease must bind exact reconciled attempt')
    # Preserve the original bytes, including its private token, without printing them.
    preserved = out/('preserved-'+s['name']+'-flight.json')
    if not preserved.exists():
        from speech_aware_evidence_acquisition.core.atomic_file import publish_new
        publish_new(preserved, blob)
    require(preserved.read_bytes() == blob and lease.read_bytes() == blob, 'lease changed during reconciliation')
    lease.unlink(); engine.settlement.sync(lease.parent)
    return {'lease':'settled', 'preserved_path':str(preserved), 'sha256':engine.sha(blob),
            'pid_absent':record['pid'], 'observed_contacts':len(matches[0]['contacts'])}


def execute(path, mode='plan'):
    d, sources, target, model = load(path)
    out = Path(d['state_dir']); logical = Path(d['data_root'])
    ident = {'config_sha256':engine.launch.digest(path), 'tool_sha256':engine.launch.digest(__file__)}
    argv = deployment.docker_run_argv(target, target.instances[0], model, nic=d['nic'])
    if mode == 'plan':
        proofs = [source_audit(s,logical) for s in sources]
        quiet = engine.settlement.transition.quiescence([s['receipt_value']['deployment']['instances'][0] for s in sources])
        return {'schema':SCHEMA, 'identity':ident, 'argv':argv, 'sources':proofs, 'quiescence':quiet,
                'new_model_calls':0, 'mutates_services':False}
    require(mode == 'run', 'plan or run required'); out.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        guards = {}
        for s in sorted(sources,key=lambda x:x['locks']):
            check, gi = stack.enter_context(engine.settlement.guard(Path(s['locks'])/'generation.guard'))
            guards[s['name']] = (check,gi)
        marker = {'schema':SCHEMA,'state_dir':str(out),'identity':ident}
        donepath = out/'complete.json'
        if donepath.exists():
            done=read(donepath);require(done['identity']==ident,'completed merge identity differs')
            for item in [done['runtime_receipt'],*done['registered_files']]:pin(item)
            return done
        intentpath = out/'intent.json'; prepared = intentpath.exists()
        for s in sources:
            pending = Path(s['locks'])/'engine-transition.pending.json'
            if pending.exists():require(read(pending)==marker,'another transition pending')
        if not prepared:
            require(inspect(target.instances[0].name) is None, 'target already exists')
            proofs=[source_audit(s,logical) for s in sources]
            quiet=engine.settlement.transition.quiescence([s['receipt_value']['deployment']['instances'][0] for s in sources])
            for s in sources:publish(Path(s['locks'])/'engine-transition.pending.json',marker)
            settlements=[settle_lease(s,p,out) for s,p in zip(sources,proofs)]
            protected=[{'id':cid,'state':inspect(cid)['State']} for cid in d['protected_container_ids']]
            for s in sources:
                logs=engine.launch.run(['docker','logs',s['container_id']]);require(logs.returncode==0,'old logs unreadable')
                publish(out/('old-'+s['name']+'-logs.json'),{'stdout':logs.stdout,'stderr':logs.stderr})
            publish(intentpath,{'identity':ident,'guards':{k:v[1] for k,v in guards.items()},'sources':proofs,
                                'settlements':settlements,'quiescence':quiet,'protected':protected,'argv':argv})
        intent=read(intentpath)
        require(intent['identity']==ident and intent['argv']==argv
                and intent['guards']=={k:v[1] for k,v in guards.items()},'merge intent differs')
        for s in sources:
            require(not os.path.lexists(Path(s['locks'])/'flight.lock'),'unresolved source lease')
            guards[s['name']][0]()
            stopped=out/('old-'+s['name']+'-stopped.json');current=inspect(s['container_id'])
            if not stopped.exists():
                if current['State']['Running']:
                    stopintent=out/('old-'+s['name']+'-stop-intent.json')
                    require(not stopintent.exists(),'unresolved stop; inspect and explicit recovery required')
                    publish(stopintent,{'container_id':s['container_id']})
                    result=engine.launch.run(['docker','stop','--time','60',s['container_id']])
                    publish(out/('old-'+s['name']+'-stop-result.json'),{'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
                    require(result.returncode==0,'old stop failed');current=inspect(s['container_id'])
                require(current['State']['Status']=='exited' and current['State']['Pid']==0,'old service not terminal')
                publish(stopped,current)
            require(current['State']==read(stopped)['State'],'old service restarted')
            for name in (s['session'],s['profile_value'].instances[0].name):
                publish(Path(s['serving'])/name/'session-stop.json',{'schema':'saea-serving-session-stop-v1',
                    'session_id':name,'old_container_id':s['container_id'],'transition_state':str(out),
                    'stopped_at':current['State']['FinishedAt'],'stale':False,'new_model_calls':0})
        lane=Path(d['new_lane_root']);(lane/'locks').mkdir(parents=True,exist_ok=True);(lane/'serving').mkdir(exist_ok=True)
        target_guard=lane/'locks/generation.guard';old_guard=Path(sources[0]['locks'])/'generation.guard'
        if not target_guard.exists():os.link(old_guard,target_guard);engine.settlement.sync(target_guard.parent)
        require(os.path.samefile(target_guard,old_guard),'new guard must preserve original inode')
        launchintent=out/'launch-intent.json';new=inspect(target.instances[0].name)
        if not launchintent.exists():
            require(new is None,'unattributed new engine')
            table=engine.npu_inventory({'new':target});allocation=deployment.allocate_from_npu_smi(target,table)
            publish(out/'allocation.json',{'allocation':allocation,'npu_smi':table})
            publish(launchintent,{'identity':ident,'argv':argv,'automatic_retry':False})
            result=engine.launch.run(argv)
            publish(out/'launch-result.json',{'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
            require(result.returncode==0,'new launch failed; do not repeat');new=inspect(target.instances[0].name)
        require(new is not None,'unknown launch; no retry')
        engine.verify_container(new,target,model)
        c={'doc':d,'new':target,'new_model':model,'out':out,'logical':logical,'serving':lane/'serving'}
        new=engine.wait_ready(c,new['Id'])
        receipt=copy.deepcopy(sources[0]['receipt_value']);i=receipt['deployment']['instances'][0]
        ti=target.instances[0]
        i.update(instance_id=ti.name,device_ids=list(ti.cards),tensor_parallel_size=target.tensor_parallel_size,
                 port=ti.port,max_num_seqs=target.max_num_seqs,max_num_batched_tokens=target.max_num_batched_tokens,
                 gpu_memory_utilization=target.gpu_memory_utilization,mm_processor_cache_type=target.mm_processor_cache_type,
                 env={**dict(ti.env),**{k:d['nic'] for k in deployment.NIC_ENV_KEYS}})
        c['receipt']=receipt
        registered=engine.finish_registration(c,new)
        for p in intent['protected']:
            current=inspect(p['id'])['State']
            require(current['Pid']==p['state']['Pid'] and current['StartedAt']==p['state']['StartedAt']
                    and current['Running'],'protected engine changed')
        for s in sources:guards[s['name']][0]()
        done={'schema':SCHEMA,'identity':ident,'new_container_id':new['Id'], 'new_profile':d['new_profile'],
              'new_model':d['new_model'],'new_session':d['new_session'],'new_lane_root':str(lane),
              'cards':list(target.cards),'port':ti.port,'new_model_calls':0,'old_containers_preserved':True,
              'old_lanes_retired':True,'successor_requires_new_preflight':True,**registered}
        publish(donepath,done)
        return done


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('plan','run'))
    parser.add_argument('--config',required=True,type=Path);args=parser.parse_args()
    print(json.dumps(execute(args.config,args.mode),sort_keys=True))
