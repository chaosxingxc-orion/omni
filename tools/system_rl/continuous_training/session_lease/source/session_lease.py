"""Local rollout admission backed by the experiment owner's asynchronous lease probe."""
import threading
import time


class SessionHealth:
    """A single owner calls renew after each authenticated remote observation.

    The stamp is in the clock domain of the owner. A late successful probe
    cannot resurrect a stopped experiment; restarting needs a new session.
    """
    def __init__(self, identity, max_age, *, clock=time.monotonic):
        import math
        if not identity or type(max_age) not in (int,float) or not math.isfinite(max_age) or max_age<=0:
            raise ValueError('invalid lease identity/lifetime')
        self.identity=tuple(identity)
        self.max_age=max_age
        self.clock=clock
        self.deadline=None
        self.stopped=False
        self.lock=threading.Lock()

    def renew(self, identity, probe_started):
        with self.lock:
            now=self.clock()
            if (self.stopped or tuple(identity)!=self.identity or not 0<=now-probe_started<self.max_age
                    or self.deadline is not None and now>=self.deadline):
                self.stopped=True
                raise ValueError('lease expired, revoked or changed owner')
            self.deadline=probe_started+self.max_age

    def revoke(self):
        with self.lock:self.stopped=True

    def admit(self):
        with self.lock:
            if self.deadline is None:raise ValueError('initial resource admission incomplete')
            if self.stopped or self.clock()>=self.deadline:
                self.stopped=True
                raise ValueError('experiment no longer holds a fresh lease')


from contextlib import contextmanager
from pathlib import Path
import json
import os

FIELDS=('owner','pid','start_ticks','boot','guard_inode','guard_device')
_active=None


class LocalLease:
    """Workers consume one controller heartbeat; they never contact SSH."""
    def __init__(self, cfg):
        self.cfg=cfg
        self.root=Path(cfg['root'])
        self.session=Path(cfg['session'])
        self.ref=cfg['owner_ref']
        from run_package import read
        self.owner=read(self.ref)
        if self.owner['boot']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():
            raise ValueError('session belongs to another boot')

    def check_local(self):
        if (self.root/'DRAIN').exists() or (self.session/'REVOKED.json').exists():
            raise ValueError('session or capture drained')
        stamp=json.loads((self.session/'HEARTBEAT.json').read_bytes())
        if (stamp['epoch']!=self.owner['epoch'] or stamp['ref']!=self.ref['sha256']
                or not 0<=time.monotonic()-stamp['started']<self.owner['max_age']):
            raise ValueError('expired or crossed experiment heartbeat')
    @contextmanager
    def admission(self):
        self.check_local()
        yield
    def start(self):
        self.check_local()
        return self
    def close(self):pass


def heartbeat(path,state):
    # Volatile monotonic observation only. Losing it on reboot closes admission.
    tmp=path.with_suffix('.'+str(os.getpid())+'.tmp')
    tmp.write_text(json.dumps(state));os.replace(tmp,path)


def observe(config,path,owner,reference,remote,probe,stop,parent):
    from run_package import write
    expected=tuple(owner['identity'][k] for k in FIELDS)
    health=SessionHealth(expected,owner['max_age'])
    try:
        while os.getppid()==parent and not stop.is_set():
            started=time.monotonic()
            actual=probe(remote)
            health.renew(tuple(actual[k] for k in FIELDS),started)
            heartbeat(path/'HEARTBEAT.json',dict(epoch=owner['epoch'],ref=reference['sha256'],started=started))
            if stop.wait(config['interval']):return
    except BaseException as exc:
        write(path/'REVOKED.json',dict(owner_ref=reference,error=repr(exc)))
        raise


def enabled(config):
    return config.get('formal_rollout',{}).get('resource_session')


def get(config):
    from run_package import pin
    if _active is None or _active['config_ref']!=pin(config['_path']):
        raise ValueError('no active experiment owner')
    _active['local'].check_local()
    return _active


@contextmanager
def lifetime(config):
    global _active
    opts=enabled(config)
    if not opts:
        yield
        return
    from multiprocessing import get_context
    from formal_rollout import settings,lease_plan,acquire,provider_modules,release
    from run_package import read,write,pin
    from dependencies import require
    import uuid
    spec,primary=settings(config);root=Path(config['output_root'])/'resident-lease'
    require(spec.get('continuous_training') and 0<opts['interval']<opts['max_age_seconds']
            and opts['max_age_seconds']<=30,'invalid dedicated lease health window')
    ref=pin(config['_path'])
    require(not (root/'RELEASED.json').exists() and not (root/'REVOKED.json').exists(),
            'terminated lease requires explicit reconciliation')
    remote,ready,service=lease_plan(config,primary,root,config['run_id']+'-session',dict(iteration_ref=ref))
    acquire(root,remote,primary,ready,service)
    held=read(root/'REMOTE-LEASE.json');identity={k:held['state'][k] for k in FIELDS}
    # A restarted controller has a new epoch: stale child workers cannot inherit authorisation.
    epoch=uuid.uuid4().hex
    owner=dict(epoch=epoch,boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),identity=identity,
               max_age=opts['max_age_seconds'],iteration_ref=ref)
    receipt=write(root/'epochs'/epoch/'OWNER.json',owner)
    cfg=dict(root=str(root),session=str(root),owner_ref=receipt)
    mp=get_context('fork');stop=mp.Event()
    with provider_modules() as (module,_,_,_):probe=module.probe
    worker=mp.Process(target=observe,args=(opts,root,owner,receipt,remote,probe,stop,os.getpid()),daemon=True)
    worker.start()
    local=LocalLease(cfg)
    try:
        until=time.monotonic()+opts['max_age_seconds']
        while worker.is_alive() and time.monotonic()<until:
            try:local.check_local();break
            except (ValueError,FileNotFoundError):time.sleep(.05)
        local.check_local()
        _active=dict(config_ref=ref,local=local,source=cfg,remote=remote,ready=ready,service=service,root=root)
        yield
        local.check_local()
        # iteration exits normally only after the original terminal accounting and actor end.
        terminal=read(Path(config['output_root'])/'CURSOR.json')
        require(terminal['phase']=='done' and not terminal.get('pending_stage'),'session still has pending work')
        stop.set();worker.join(timeout=35)
        write(root/'RELEASED.json',release(root,remote,service,dict(terminal_verified=True,unknown_requests=0,
              iteration_ref=ref,outcome_ref=pin(Path(config['output_root'])/'OUTCOME.json')),spec['drain_timeout_seconds']))
    finally:
        _active=None;stop.set();worker.join(timeout=35)
        if worker.is_alive():worker.terminate();worker.join()


def boundary(config,terminal):
    from run_package import pin
    actor=get(config)
    if not terminal['terminal_verified'] or terminal['unknown_requests']!=0:
        raise ValueError('unknown requests at session boundary')
    return dict(status='session_held',terminal=terminal,lease_ref=pin(actor['root']/'REMOTE-LEASE.json'))
