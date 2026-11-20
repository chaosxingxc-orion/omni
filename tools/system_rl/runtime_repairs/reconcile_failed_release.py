"""Read-only reconciliation before retiring a failed release's client observer.

Uses original Docker owners and their closed ledger; never dispatches or stops.
"""
import argparse
import base64
import datetime
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import tarfile


def closed_bounded_capture(inventory_path, owner):
    """Reconcile an immutable completed cohort against its original ledger."""
    from dependencies import ROOT, load
    from run_package import pin, read
    base = load(ROOT/'collection-dispatch-integration/base.py','bounded_terminal_base')

    inventory_ref = pin(inventory_path)
    inventory = read(inventory_ref)
    root = Path(inventory['cohort_root'])
    assert inventory['closed'] is True and Path(inventory_ref['path']).parent == root
    for ref in inventory['files'].values():
        pin(ref)
    closed = read(inventory['generation_closed_ref'])
    plan = read(closed['selected_cohort_ref'])
    capture = read(inventory['files']['INPUT.json'])
    ledger = Path(capture['execution_registry']['path'])
    observations = []
    with sqlite3.connect(ledger.as_uri()+'?mode=ro', uri=True) as db:
        assert not db.execute("SELECT 1 FROM spend WHERE status NOT IN ('known','closed_unobserved') LIMIT 1").fetchone()
        assert closed['groups'] == plan['group_ids']
        for draw in plan['draws']:
            group = read(inventory['files'][f"groups/{draw['draw_index']}/GENERATED.json"])
            assert group['group_id'] == draw['group_id']
            assert len(group['episodes']) == capture['bounded_train_probe']['G']
            for episode in group['episodes']:
                call, = episode['calls']
                packet = call['packet']
                request = packet['request_id']
                row = db.execute('SELECT status,packet,cost FROM spend WHERE id=?',(request,)).fetchone()
                assert row and row[:2] == ('known',base.digest(base.canonical(packet)))
                ref = call['envelope']['response_ref']
                assert ref == inventory['files'][str(Path(ref['path']).relative_to(root))]
                response = read(ref)
                sha = hashlib.sha256(base64.b64decode(response['response_b64'],validate=True)).hexdigest()
                assert response['request_id'] == request
                assert sha == response['response_sha256'] == call['reply']['response_sha256']
                observations.append(dict(request_id=request,response_ref=ref,cost_units=row[2]))
    assert len(observations) == len({x['request_id'] for x in observations})
    return dict(owner=owner,terminal_verified=True,unknown_requests=0,automatic_retry=False,
                inventory_ref=inventory_ref,observations=observations)


def request_delta(actual, seed):
    assert all(actual[k] == seed[k] for k in ('meta', 'history', 'samples', 'batches')), 'non-request ledger changed'
    old = {row[0]: row for row in seed['spend']}
    current = {row[0]: row for row in actual['spend']}
    assert all(row[2] in ('known','closed_unobserved') for row in old.values()), 'seed contains unknown requests'
    assert all(current.get(key) == row for key, row in old.items()), 'inherited request changed or disappeared'
    return {key: row for key, row in current.items() if key not in old}


def close_refused_local_connections(registry, attempts, output):
    """Retire a drained batch whose clients failed at connect, retaining costs."""
    import hashlib
    from urllib.parse import urlsplit
    from base import canonical, digest
    from run_package import pin, write

    closed = []
    for item in attempts:
        packet = json.loads(Path(item['packet']).read_bytes())
        intent = json.loads(Path(item['intent']).read_bytes())
        dispatch = json.loads(Path(item['dispatch']).read_bytes())
        log = Path(item['client_log']).read_text()
        route = json.loads(Path(item['lane']).read_bytes())['route']
        endpoint = urlsplit(route['local_endpoint'])
        rid = packet['request_id']
        assert endpoint.hostname in ('127.0.0.1', '::1') and endpoint.port
        assert dispatch['kind'] == 'dispatch' and dispatch['request_id'] == rid == intent['request_id']
        assert dispatch['intent_sha256'] == hashlib.sha256(Path(item['intent']).read_bytes()).hexdigest()
        assert dispatch['request_sha256'] == intent['request_sha256']
        assert intent['payload']['wire_sha256'] == packet['wire_sha256']
        assert not Path(item['intent'].replace('.intent.json', '.response.json')).exists()
        assert 'sock.connect(sa)' in log and 'ConnectionRefusedError: [Errno 111]' in log
        assert 'transport.Unknown: native journal retained; URLError: <urlopen error [Errno 111] Connection refused>' in log
        closed.append(dict(request_id=rid, packet_sha256=digest(canonical(packet)),
                           status='closed_unobserved', response_observed=False,
                           automatic_retry=False, training_admission=False,
                           endpoint=route['local_endpoint'], reason='local_connect_refused',
                           evidence={key: pin(path) for key,path in item.items()}))
    ids = {r['request_id'] for r in closed}
    assert closed and len(ids) == len(closed)
    with registry.transaction() as db:
        pending = dict(db.execute("SELECT id,packet FROM spend WHERE status NOT IN ('known','closed_unobserved')"))
        assert ids == set(pending), 'other unresolved requests remain'
        for row in closed:
            rid = row['request_id']
            assert pending[rid] == row['packet_sha256']
            assert db.execute("UPDATE spend SET status='closed_unobserved' WHERE id=? AND status='intent'", (rid,)).rowcount == 1
            db.execute('INSERT INTO meta VALUES (?,?)', ('closed-unobserved:'+rid, canonical(row).decode()))
    return write(output, dict(closed=closed, usage=registry.usage(), replayed_requests=0))


def classify_observed_failures(config, raw_path, destination, terminal_proof=None):
    """Closed-ledger successor: verify original journal, retain all paid costs.

    Never modifies the remote ledger or the downloaded original. Only existing
    intent→known transitions with a validated error response are permitted.
    """
    import sys
    import hashlib
    parent=Path(config['release'])/'tree/parents'
    sys.path[:0]=[str(parent/'collection-dispatch-integration'),
        str(parent/'collection-scale-successor/code'),
        str(parent/'collection-scale-successor/ready-autocontinue/production-next/code')]
    from direct_transport import DirectHTTP
    from reservations import Registry
    from base import canonical,digest
    with sqlite3.connect(Path(raw_path).as_uri()+'?mode=ro',uri=True) as db:
        identity=json.loads(db.execute("SELECT v FROM meta WHERE k='authority'").fetchone()[0])
        rows={row[0]:row for row in db.execute('SELECT id,rank,status,cost,packet FROM spend')}
        pending={key for key,row in rows.items() if row[2] not in ('known','closed_unobserved')}
    evidence=[]
    seen=set()
    outcomes=[(item,False) for item in config.get('observed_failures',[])]+[(item,True) for item in config.get('observed_successes',[])]
    for item,success in outcomes:
        packet_path=Path(item['packet_path']);packet=json.loads(packet_path.read_text())
        rid=packet['request_id'];row=rows[rid]
        assert rid not in seen, 'duplicate failure evidence'
        seen.add(rid);pending.discard(rid)
        assert row[2] in ('intent','known'), 'unsupported request state'
        assert row[4]==digest(canonical(packet)), 'reserved packet differs'
        cfg=dict(item['transport'],package_src=str(parent/'jev-system-rl/original-contact-package'))
        transport=DirectHTTP(item['journal_root'],cfg)
        state=transport.inspect(packet)
        assert state['state']=='observed','response custody incomplete'
        response=json.loads(state['response_bytes'])
        if success:
            expected=Path(item['response_ref']['path']).read_bytes()
            assert hashlib.sha256(expected).hexdigest()==item['response_ref']['sha256'],'success response evidence drift'
            assert expected==state['response_bytes'],'success response differs from actual journal'
            assert not response.get('error') and response.get('id') and response.get('choices'),'observed completed response required'
            assert all(c.get('finish_reason') in ('stop','length') for c in response['choices']),'unfinished response'
            assert row[2]=='known','successful request must already be accounted by original transport'
        else:
            expected=item.get('expected_error_code',400)
            assert type(expected) is int and 400<=expected<=599, 'explicit HTTP failure status'
            assert response.get('error') and response['error'].get('code')==expected,'explicit observed rejection required'
        evidence.append(dict(request_id=rid,cost=row[3],response_sha256=hashlib.sha256(state['response_bytes']).hexdigest(),error=response.get('error'),response_observed=True,training_admission=False))
    lost=[]
    for item in config.get('closed_unobserved',[]):
        # Terminal ownership proves no computation remains live. It does NOT
        # prove success, failure, provider non-execution, or any token/action.
        assert terminal_proof and terminal_proof['all_owners_terminal'] and terminal_proof['original_restored']
        assert terminal_proof['containers']==config['containers'],'terminal owner differs'
        packet=json.loads(Path(item['packet_path']).read_text());rid=packet['request_id'];row=rows[rid]
        assert rid not in seen and row[2]=='intent','only an original unresolved intent may close'
        assert row[4]==digest(canonical(packet)), 'reserved packet differs'
        cfg=dict(item['transport'],package_src=str(parent/'jev-system-rl/original-contact-package'))
        transport=DirectHTTP(item['journal_root'],cfg)
        state=transport.inspect(packet)
        assert state['state'] not in ('absent','observed'), 'missing intent or observed response requires different reconciliation'
        assert item['transport']['endpoint']==terminal_proof['endpoint'],'request endpoint differs from closed owner'
        assert packet['fresh_capture_identity']['boot_id']==terminal_proof['source_boot'],'request boot differs from closed owner'
        seen.add(rid);pending.discard(rid)
        lost.append(dict(request_id=rid,packet_sha256=row[4],cost=row[3],
            status='closed_unobserved',response_observed=False,outcome='unknown',
            training_admission=False,automatic_retry=False,terminal_proof=terminal_proof))
    assert not pending,'unclassified requests remain; do not retry'
    destination=Path(destination)
    assert not destination.exists(),'successor ledger already exists'
    with sqlite3.connect(Path(raw_path).as_uri()+'?mode=ro',uri=True) as src,sqlite3.connect(destination) as dst:
        src.backup(dst)
    registry=Registry(destination,identity['authority'],identity['resources'])
    for item in evidence:registry.observed(item['request_id'])
    with registry.transaction() as db:
        for item in lost:
            db.execute("UPDATE spend SET status='closed_unobserved' WHERE id=? AND status='intent'",(item['request_id'],))
            db.execute('INSERT INTO meta VALUES (?,?)',('closed-unobserved:'+item['request_id'],canonical(item).decode()))
    assert registry.usage()['unknown']==0
    return dict(ledger_path=str(destination),usage=registry.usage(),observations=evidence,closed_unobserved=lost,
                remote_ledger_modified=False,raw_ledger_preserved=True,replayed_requests=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config['output'])
    root.mkdir(parents=True, exist_ok=True)

    def remote(argv):
        import shlex
        return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                               config['host'], shlex.join(argv)], check=True,
                              capture_output=True, timeout=60).stdout

    def copy(cid, source, filename):
        payload = remote(['docker', 'cp', cid + ':' + source, '-'])
        with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
            members = [m for m in archive.getmembers() if m.isfile()]
            assert len(members) == 1, 'expected one evidence file'
            data = archive.extractfile(members[0]).read()
        path = root / filename
        if path.exists():
            assert path.read_bytes() == data, 'closed evidence drift'
        else:
            path.write_bytes(data)
        return path

    states = {}
    for role, cid in config['containers'].items():
        states[role] = json.loads(remote(['docker', 'inspect', '--format',
                                          '{{json .State}}', cid]))
    assert states['watchdog']['Status'] == 'exited' and states['watchdog']['ExitCode'] == 0
    for role in ('controller', 'coordinator', 'service'):
        assert states[role]['Status'] == 'exited', role + ' not terminal'
    assert states['original_service']['Running'], 'original service not running'
    restored = json.loads(copy(config['containers']['watchdog'], config['restoration_path'],
                               'RESTORED.json').read_text())
    assert restored['config_exact'] is True and restored['models']['data']
    assert restored['container_id'] == config['containers']['original_service']
    ledger_path = copy(config['containers']['controller'], config['registry_path'], 'registry.sqlite')

    def ledger(path):
        connection = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
        try:
            return {table: sorted(connection.execute('SELECT * FROM ' + table).fetchall(), key=repr)
                    for table in ('meta', 'history', 'spend', 'samples', 'batches')}
        finally:
            connection.close()

    actual = ledger(ledger_path)
    seed=ledger(config['seed_registry'])
    classified=None
    delta=request_delta(actual,seed)
    terminal_proof=None
    if config.get('closed_unobserved'):
        activation=json.loads(copy(config['containers']['service'],config['activation_path'],'ACTIVATION.json').read_text())
        assert activation['endpoint']==config['closed_endpoint']
        # Require the exact stopped service to have been configured with this
        # overlay root; an unrelated dead container is insufficient evidence.
        inspect=json.loads(remote(['docker','inspect',config['containers']['service']]))[0]
        assert inspect['Id']==config['containers']['service'] and inspect['HostConfig']['RestartPolicy']['Name']=='no'
        assert config['server_config_path'] in inspect['Args'], 'closed service configuration not bound'
        server=json.loads(copy(config['containers']['service'],config['server_config_path'],'SERVER.json').read_text())
        assert server['overlay_root']==config['overlay_root'], 'closed service overlay not bound'
        assert activation['service_identity']['execution_ref']==server['execution_ref']
        assert inspect['Name'].lstrip('/')==activation['service_identity']['container_name']
        terminal_proof=dict(containers=config['containers'],states=states,
            all_owners_terminal=True,original_restored=True,restored=restored,
            endpoint=activation['endpoint'],policy_version=activation['policy_version'],
            source_boot=activation['source_boot'],
            activation_sha256=__import__('hashlib').sha256((root/'ACTIVATION.json').read_bytes()).hexdigest(),
            response_recoverable=False,training_admission=False)
    if config.get('observed_failures') or config.get('observed_successes') or config.get('closed_unobserved'):
        outcomes=config.get('observed_failures',[])+config.get('observed_successes',[])+config.get('closed_unobserved',[])
        evidence_ids=[json.loads(Path(item['packet_path']).read_text())['request_id'] for item in outcomes]
        assert len(evidence_ids)==len(set(evidence_ids)) and set(evidence_ids)==set(delta), 'every new request requires distinct observed response evidence'
        classified=classify_observed_failures(config,ledger_path,root/'RECONCILED-registry.sqlite',terminal_proof)
        after=ledger(classified['ledger_path'])
        assert all(after[k]==actual[k] for k in ('history','samples','batches'))
        assert all(row in after['meta'] for row in actual['meta'])
        assert sorted((r[0],r[1],r[3],r[4]) for r in after['spend'])==sorted((r[0],r[1],r[3],r[4]) for r in actual['spend'])
    else:
        assert actual == seed, 'ledger changed; classify requests before retry'
    report = dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  restored=restored, ledger=actual, remote_ledger_equal_seed=actual==seed,
                  HTTP_requests=len(delta),inherited_HTTP_requests=len(seed['spend']),model_calls=0 if not delta else None,
                  classified_observed_failures=classified,failure=config['failure'], **states)
    path = root / 'TERMINAL-RECONCILIATION.json'
    assert not path.exists(), 'terminal report already sealed'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(passed=True, report=str(path), HTTP_requests=len(delta),classified=classified)))


if __name__ == '__main__':
    main()
