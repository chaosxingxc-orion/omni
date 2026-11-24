"""Stream CLOSED immutable results after controller exit; bounded by config.

Full FSDP shards remain authoritative on181 AND are streamed for custody. No
whole shard/tar is materialized in RAM. Never read active or Ceph run state.
"""
import hashlib,io,json,subprocess,sys,tarfile
from pathlib import Path


HASHES={}
def sha(path):
    path=Path(path);stat=path.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
    if key in HASHES:return HASHES[key]
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    HASHES[key]=h.hexdigest();return HASHES[key]


def projected_mount(path,mappings):
    logical=Path(path).resolve();candidates=[]
    for mapping in mappings or []:
        target=Path(mapping['Destination']).resolve()
        if logical==target or logical.is_relative_to(target):
            candidates.append((len(target.parts),Path(mapping['Source'])/logical.relative_to(target),mapping.get('RW',False)))
    if not candidates:return logical,True
    longest=max(n for n,_,_ in candidates)
    matches={(p,writable) for n,p,writable in candidates if n==longest}
    assert len(matches)==1,'ambiguous closed mount projection: '+str(logical)
    return matches.pop()


def physical_path(path,mappings):
    return projected_mount(path,mappings)[0]


def generated_group(value):
    """Large rollout data has a different contract from small control JSON."""
    if not isinstance(value,dict) or set(value)!={'group_id','query_draw_id','episodes','policy'}:
        return False
    if not all(isinstance(value[k],str) and value[k] for k in ('group_id','query_draw_id')):
        return False
    episodes=value['episodes']
    if not isinstance(episodes,list) or not episodes or not isinstance(value['policy'],dict):
        return False
    ids=set();replicates=set()
    for episode in episodes:
        if not isinstance(episode,dict) or set(episode)!={'episode_id','replicate','calls','synthetic'}:
            return False
        eid=episode['episode_id'];replica=episode['replicate']
        if (not isinstance(eid,str) or not eid.startswith(value['group_id']+':e') or eid in ids or
            type(replica) is not int or replica<0 or replica in replicates or episode['synthetic'] is not False):
            return False
        ids.add(eid);replicates.add(replica)
        if not isinstance(episode['calls'],list) or not episode['calls']:return False
        for call in episode['calls']:
            if not isinstance(call,dict) or not {'packet','reply','online','output','annotation',
                    'token_ids','behavior_logprobs','prompt_token_ids','raw_output','usage','envelope','annotation_ref'}<=set(call):
                return False
            if not isinstance(call['packet'],dict) or call['packet'].get('stage') not in ('P','S1','S2','D'):
                return False
            if not isinstance(call['token_ids'],list) or not isinstance(call['behavior_logprobs'],list) or len(call['token_ids'])!=len(call['behavior_logprobs']):
                return False
    return True


def collect(root,max_bytes,*,full_checkpoint=False,mappings=None):
    HASHES.clear()
    files={};seen=set();size=0
    normalized=[(Path(x['Destination']).resolve(),Path(x['Source']).resolve(),x.get('RW',False)) for x in mappings or []]
    projections={}
    def project(path):
        if path in projections:return projections[path]
        found=[(len(target.parts),source/path.relative_to(target),writable) for target,source,writable in normalized if path==target or path.is_relative_to(target)]
        if not found:result=(path,True)
        else:
            longest=max(x[0] for x in found)
            choices={(p,w) for n,p,w in found if n==longest}
            assert len(choices)==1,'ambiguous closed mount projection: '+str(path)
            result=choices.pop()
        projections[path]=result;return result
    def add(path,expand=True,expected=None):
        nonlocal size
        path=Path(path).resolve()
        if str(path).startswith('/mnt/cephfs'):return
        if path in seen:
            if expected is not None:assert files['files/'+str(path).lstrip('/')]['sha256']==expected,'duplicate closed ref hash differs'
            return
        physical,writable=project(path)
        if not physical.is_file():
            marker=physical.parent/'MODEL-RELOCATED.json'
            assert expected and marker.is_file(),'closed ref unavailable: '+str(path)
            ticket=json.loads(marker.read_text())
            assert ticket['model_files'].get(path.name)==expected and ticket['complete_ref']['sha256'],'missing shard lacks permanent custody'
            assert path.parent==Path(ticket['checkpoint_ref']['path']).parent,'relocated source path drift'
            count=ticket['model_sizes'][path.name]
            name='files/'+str(path).lstrip('/');files[name]=dict(path=str(path),sha256=expected,bytes=count)
            size+=count;assert size<=max_bytes,'closed return byte bound';seen.add(path)
            return
        if expected is not None:assert sha(physical)==expected,'closed ref changed: '+str(path)
        if path in seen:return
        seen.add(path);size+=physical.stat().st_size
        assert size<=max_bytes,'configured closed return byte budget exceeded'
        name='files/'+str(path).lstrip('/');files[name]=dict(path=str(path),sha256=sha(physical),bytes=physical.stat().st_size)
        # Readonly mounts are sealed task inputs, not newly produced outputs.
        # Retain their exact bytes/hash without recursively replaying historical
        # provenance that was intentionally not needed or staged for this task.
        if expand and writable and path.suffix=='.json':
            large=physical.stat().st_size>16*1024**2
            if large:
                assert (path.name=='GENERATED.json' and path.parent.name.isdigit() and
                        path.parent.parent.name=='groups'), 'control JSON byte budget: '+str(path)
                assert physical.stat().st_size<=min(max_bytes,128*1024**2), 'trajectory JSON byte budget: '+str(path)
            value=json.loads(physical.read_bytes())
            if large:assert generated_group(value), 'large JSON is not a closed generated rollout group: '+str(path)
            if path.name=='GENERATION-INVENTORY.json' and isinstance(value,dict) and set(value)=={'closed','cohort_root','generation_closed_ref','files'}:
                assert value['closed'] is True and Path(value['cohort_root']).resolve()==path.parent, 'generation inventory source differs'
                for name,ref in value['files'].items():
                    relative=Path(name)
                    assert not relative.is_absolute() and '..' not in relative.parts, 'generation inventory path escape'
                    assert Path(ref['path']).resolve()==path.parent/relative, 'generation inventory member differs'
                    add(ref['path'],False,ref['sha256'])
                refs(value['generation_closed_ref'])
                # Finalize reads these control dependencies outside the cohort;
                # source packet/history metadata stays verbatim in inventory.
                for name in ('INPUT.json','CONSUMER.json','NATIVE.pending.disabled.json','PENDING-CAPTURE.json'):
                    ref=value['files'].get(name)
                    if ref:
                        physical_control=physical_path(ref['path'],mappings)
                        assert physical_control.stat().st_size<=16*1024**2,'generation control byte bound'
                        refs(json.loads(physical_control.read_bytes()))
                return
            if (path.name=='SELECTED.json' and isinstance(value,dict)
                    and value.get('schema')=='selected-fresh-cohort-v1'
                    and value.get('prepare_mode') in ('bounded_train_probe','finite_diagnostic_capture')):
                # The finite selector read outcomes/records and merely cited
                # the full population stream from each frozen pool. It is not
                # an input to replay/finalize. Verify the citation in its
                # sealed source rather than requiring a new staged replica.
                for item in value['source_custody']:
                    assert set(item)=={'version','pool','streams'}
                    for key in ('version','pool'):
                        ref=item[key];add(ref['path'],False,ref['sha256'])
                    version,pool=(json.loads(project(Path(item[k]['path']).resolve())[0].read_bytes())
                                  for k in ('version','pool'))
                    assert version['training_pool']==item['pool'] and pool['sealed']
                    assert pool['streams']==item['streams'] and set(pool['streams'])=={'outcomes','records','population'}
                    for key in ('outcomes','records'):
                        refs(item['streams'][key])
                value={k:v for k,v in value.items() if k!='source_custody'}
            if path.name.endswith('.packet.json') and isinstance(value,dict) and 'fresh_capture_identity' in value and 'original_packet' in value:
                assert value['stage'] in ('P','S1','S2','D') and value['wire_sha256']==hashlib.sha256(json.dumps(value['wire'],ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(), 'captured packet wire drift'
                return
            if path.name=='GENERATED.json' and isinstance(value,dict) and set(value)=={'episode_id','replicate','calls','synthetic'}:
                group=dict(group_id=value['episode_id'].rsplit(':e',1)[0],query_draw_id='closed-episode',episodes=[value],policy={})
                assert generated_group(group),'closed episode schema differs'
                for call in value['calls']:
                    refs(call['envelope']);refs(call['online'].get('audio_assets',[]))
                return
            if path.name=='GENERATED.json' and generated_group(value):
                # Packet metadata includes historical provenance citations, not
                # live file dependencies. The full packet bytes remain intact;
                # follow only actual producer receipts and audio needed by finalize.
                for episode in value['episodes']:
                    for call in episode['calls']:
                        refs(call['envelope'])
                        refs(call['online'].get('audio_assets',[]))
                return
            if isinstance(value,dict) and value.get('schema')=='omni-dp8-iteration-v1':
                # Client-only archive acceptance/code and frozen source bundle
                # remain at their independent source. They are not task outputs.
                value=dict(value)
                for key in ('archive','source_bundle_ref'):value.pop(key,None)
            # COMMITTED rank file names are relative and contain no refs.
            if path.name=='COMMITTED.json' and full_checkpoint and isinstance(value,dict) and 'ranks' in value:
                for rank in value['ranks']:
                    for name,expected in rank['files'].items():
                        shard=(path.parent/name).resolve();assert shard.is_relative_to(path.parent),'checkpoint path escape'
                        add(shard,False,expected)
            if path.name=='COMMITTED.json' and value.get('kind')=='lora_training_state':
                # This source is executed remotely with python -c; stdlib only.
                assert value['storage_mode']=='lora_training_state' and value['world_size']==8
                assert [r['rank'] for r in value['ranks']]==list(range(8))
                for rank,row in enumerate(value['ranks']):
                    assert f'jev-master-rank-{rank}.pt' in row['files'] and f'replay-rng-rank-{rank}.pt' in row['files']
                    assert not any(n.startswith(('model_world_size_','optim_world_size_')) for n in row['files'])
                value={k:v for k,v in value.items() if k not in ('base_identity','source_identity','resident_identity')}
            refs(value)
    def refs(value):
        if isinstance(value,dict):
            if {'path','sha256'}<=value.keys():add(value['path'],expected=value['sha256'])
            else:
                for child in value.values():refs(child)
        elif isinstance(value,list):
            for child in value:refs(child)
    closure=json.loads((root/'CLOSED.json').read_text())
    if closure.get('schema') in ('omni-formal-actor-return-v1','formal-resident-command-v1'):
        resident=closure['schema']=='formal-resident-command-v1'
        actor_root=(root if resident else root.parent).resolve()
        if resident:
            assert root.parent.name=='results' and root.name==f"{closure['sequence']:06d}"
            assert closure['return_ref']['path']==str(root/'RETURN.json') and len(closure['rank_live_refs'])==8
        for ref in closure['projected_members']:
            path=Path(ref['path'])
            assert path==path.resolve() and path.is_relative_to(actor_root), 'formal receipt path escapes actor output'
            add(path,False,ref['sha256'])
        add(root/'RETURN.json',False,closure['return_ref']['sha256'])
        add(root/'CLOSED.json',False)
    else:
        add(root/'RETURN.json');add(root/'CLOSED.json')
    # RETURN has a stable consumer entry as well as logical path custody.
    value=files['files/'+str((root/'RETURN.json').resolve()).lstrip('/')]
    files['RETURN.json']=value
    assert sum(v['bytes'] for v in files.values())<=max_bytes,'closed return total exceeds configured budget'
    return files


def emit(root,output,max_bytes,*,full_checkpoint=False,inventory=None,expected_boot=None,mappings=None):
    files=collect(Path(root),max_bytes,full_checkpoint=full_checkpoint,mappings=mappings)
    closure=json.loads((Path(root)/'CLOSED.json').read_text())
    assert closure['immutable'],'only immutable closure may stream'
    if expected_boot:assert closure['source_boot']==expected_boot,'closed source boot drift'
    assert closure['return_ref']['sha256']==sha(Path(root)/'RETURN.json'),'closed return identity drift'
    known=inventory or {};objects={};sources={}
    for ref in files.values():
        digest=ref['sha256'];size=ref['bytes']
        assert digest not in objects or objects[digest]['bytes']==size,'hash size conflict'
        objects[digest]=dict(bytes=size,member=None if known.get(digest)==size else 'objects/'+digest)
        sources[digest]=str(physical_path(ref['path'],mappings))
    manifest=dict(schema='closed-cas-v1',files=files,objects=objects,closure=dict(
        source_boot=closure['source_boot'],return_sha256=sha(Path(root)/'RETURN.json'),
        closed_path=str((Path(root)/'CLOSED.json').resolve()),closed_sha256=sha(Path(root)/'CLOSED.json')))
    raw=json.dumps(manifest,sort_keys=True).encode()
    with tarfile.open(fileobj=output,mode='w|') as archive:
        info=tarfile.TarInfo('MANIFEST.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
        for digest,row in sorted(objects.items()):
            if not Path(sources[digest]).is_file():
                assert row['member'] is None,'relocated object absent from client; restore immutable permanent archive first'
                continue
            assert sha(sources[digest])==digest,'closed source changed before streaming'
            if row['member'] is None:continue
            info=tarfile.TarInfo(row['member']);info.size=row['bytes'];info.mode=0o444
            with Path(sources[digest]).open('rb') as stream:archive.addfile(info,stream)


def main():
    root=Path(sys.argv[1]);maximum=int(sys.argv[2]);assert maximum>0
    if '--formal-resident' in sys.argv[3:]:
        boot=next(x.split('=',1)[1] for x in sys.argv[3:] if x.startswith('--expected-boot='))
        run=next(x.split('=',1)[1] for x in sys.argv[3:] if x.startswith('--owner='))
        assert boot==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        assert root.parent.name=='results' and root.parent.parent.name=='resident' and root==root.resolve()
        owner=root.parent.parent;hot=owner.parent
        if not (root/'CLOSED.json').is_file():
            if (hot/'EXIT.json').is_file():raise SystemExit(3)
            raise SystemExit(75)
        closure=json.loads((root/'CLOSED.json').read_bytes())
        cfg=json.loads((hot.parent/'TRAIN.local.json').read_bytes())
        cid=json.loads((hot/'CONTAINER.json').read_bytes())['container_id']
        observed=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
        assert observed['Id']==cid and observed['Image']==cfg['image'] and cfg['boot_id']==boot
        assert closure['schema']=='formal-resident-command-v1' and closure['run_id']==cfg['continuous_actor']['run_id']==run
        assert [m['Source'] for m in observed['Mounts'] if m['Destination']==str(hot) and m['RW']]==[str(hot)]
        assert sha(hot.parent/'TRAIN.local.json')==json.loads((hot/'INTENT.json').read_bytes())['config_sha256']
        inventory=json.loads(next((x.split('=',1)[1] for x in sys.argv[3:] if x.startswith('--inventory-json=')),'{}'))
        emit(root,sys.stdout.buffer,maximum,expected_boot=boot,inventory=inventory)
        return
    if '--formal-actor' in sys.argv[3:]:
        boot=next((x.split('=',1)[1] for x in sys.argv[3:] if x.startswith('--expected-boot=')),None)
        if not (root/'CLOSED.json').is_file():
            hot=root.parent
            if (hot/'EXIT.json').is_file():
                terminal=json.loads((hot/'EXIT.json').read_text())
                if terminal['exit_code']!=0 or terminal.get('failure') is not None:
                    raise SystemExit(3)
            raise SystemExit(75)
        assert boot and boot==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        closed=json.loads((root/'CLOSED.json').read_text())
        assert closed['schema']=='omni-formal-actor-return-v1' and closed['source_boot']==boot
        hot=root.parent
        launch=json.loads((hot/'CONTAINER.json').read_text())
        terminal=json.loads((hot/'EXIT.json').read_text())
        cid=terminal['container_id']
        assert cid==launch['container_id'] and terminal['exit_code']==0 and terminal['complete_ranks']==8
        observed=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
        assert observed['Id']==cid and not observed['State']['Running'] and observed['State']['ExitCode']==0
        inventory=json.loads(next((x.split('=',1)[1] for x in sys.argv[3:] if x.startswith('--inventory-json=')),'{}'))
        emit(root,sys.stdout.buffer,maximum,expected_boot=boot,inventory=inventory)
        return
    if not (root/'CREATED.json').exists():raise SystemExit(75)
    inventory=json.loads(next((x.split('=',1)[1] for x in sys.argv[3:] if x.startswith('--inventory-json=')),'{}'))
    expected_boot=next((x.split('=',1)[1] for x in sys.argv[3:] if x.startswith('--expected-boot=')),None)
    cid=json.loads((root/'CREATED.json').read_text())['container_id']
    if (root/'CONTROLLER-RECOVERY.json').is_file():
     recovery=json.loads((root/'CONTROLLER-RECOVERY.json').read_text())
     assert recovery['kind']=='pre-admission-cpu-auxiliary-mount-recovery' and recovery['original_container_id']==cid
     assert recovery['source_boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
     assert recovery['original_failure_sha256']==__import__('hashlib').sha256((root/'FAILED.json').read_bytes()).hexdigest()
     previous=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
     assert previous['Id']==cid and not previous['State']['Running'] and previous['State']['ExitCode']!=0
     cid=recovery['container_id']
    if (root/'CONTROLLER-RECOVERY-CHAIN.json').is_file():
     for recovery in json.loads((root/'CONTROLLER-RECOVERY-CHAIN.json').read_text()):
      allowed={'pre-admission-cpu-auxiliary-mount-recovery','preflight-readonly-anchor-mount','preflight-project-anchor-mount','pre-stop-watchdog-physical-source','pre-stop-watchdog-directory-entry','restored-never-launched-handoff'}
      assert recovery['kind'] in allowed and recovery['original_container_id']==cid
      assert recovery['source_boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
      if recovery.get('original_failure_sha256'):
       assert recovery['original_failure_sha256']==hashlib.sha256(Path(recovery['failure_path']).read_bytes()).hexdigest()
      else:
       # Recorded recovery CIDs are linked to original durable Docker intents;
       # all earlier CPU owners must be terminal before closing this result.
       intent_files=list(root.glob('*INTENT.json'))
       successor=json.loads(subprocess.check_output(['docker','inspect',recovery['container_id']]))[0]
       assert not successor['HostConfig'].get('Devices')
       matched=False
       for intent_path in intent_files:
        intent=json.loads(intent_path.read_text());argv=intent.get('argv',[])
        if '--name' not in argv or argv[argv.index('--name')+1]!=successor['Name'].lstrip('/'):continue
        assert successor['Image'] in argv and successor['Config']['Cmd']==argv[argv.index(successor['Image'])+1:]
        matched=True;break
       assert matched,'recovery lacks original command intent'
      previous=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
      assert previous['Id']==cid and not previous['State']['Running'] and previous['State']['ExitCode']!=0
      assert not previous['HostConfig'].get('Devices')
      cid=recovery['container_id']
    observed=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
    assert observed['Id']==cid,'closed original container identity drift'
    state=observed['State'];mappings=observed['Mounts']
    if '--resident-command' in sys.argv[3:]:
        command=next(x.split('=',1)[1] for x in sys.argv[3:] if x.startswith('--command-root='))
        command=Path(command).resolve()
        assert command.is_relative_to(root.resolve()/'control/package/hot/resident/results'),'command result escapes owner'
        if not (command/'CLOSED.json').is_file():raise SystemExit(75 if state['Running'] else 3)
        assert json.loads((command/'CLOSED.json').read_text())['kind']=='resident-command'
        emit(command,sys.stdout.buffer,maximum,full_checkpoint='--full-checkpoint' in sys.argv[3:],inventory=inventory,expected_boot=expected_boot,mappings=mappings);return
    if state['Running']:raise SystemExit(75)
    if state['ExitCode']!=0 or not (root/'CLOSED.json').exists():raise SystemExit(3)
    assert json.loads((root/'CLOSED.json').read_text())['immutable']
    emit(root,sys.stdout.buffer,maximum,full_checkpoint='--full-checkpoint' in sys.argv[3:],inventory=inventory,expected_boot=expected_boot,mappings=mappings)

if __name__=='__main__':main()
