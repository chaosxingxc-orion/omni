"""Thin entry bridge to the pinned, unchanged S1 native runtime and host.

No alternative journal, Admission, HTTP transport, queue or lease is introduced.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
import time
import types

from . import configured_s1 as shared


def runtime_module(config, workspace):
    c = shared.read(config)
    root = Path(workspace)
    task = shared.read(shared.resolve(root, c['configured']['task']))
    shim = types.ModuleType('common')
    shim.__dict__.update(vars(shared))
    shim.R = root
    shim.O = shared.resolve(root, c['configured']['prepared']).parent
    shim.MODEL = task['model']
    shim.imports = lambda: shared.imports(root, task)
    previous = sys.modules.get('common')
    sys.modules['common'] = shim
    try:
        module = shared.load('configured_s1_existing_runtime', shared.pinned(root, task['legacy_runtime']))
    finally:
        if previous is None:
            sys.modules.pop('common', None)
        else:
            sys.modules['common'] = previous
    module.preflight = shared.preflight
    return module


def bind(config, workspace, state, output):
    shared.require(not Path(state).exists(), 'existing run state requires reconciliation')
    c, rails, prepared, old = shared.preflight(config, workspace)
    module = runtime_module(config, workspace)
    journal = module.journal_for(state, c, prepared)
    shared.require(all(journal.inspect(p['request_id'])['state'] == 'absent'
                       for row, packets in prepared for p in packets),
                   'unknown/known inventory requires resume reconciliation')
    shared.save(output, dict(schema='s1-native-execution-v1', run_id=c['run_id'], attempt_id=c['attempt_id'],
        config_sha256=shared.file_sha(config), runtime_sha256=shared.file_sha(module.__file__),
        original_launch_epoch=time.time(), concurrency=c['concurrency'], model_calls=0))


def main(mode=None):
    parser = argparse.ArgumentParser()
    if mode != 'bind':
        parser.add_argument('mode', choices=['check', 'run'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--workspace')
    parser.add_argument('--output')
    parser.add_argument('--state')
    parser.add_argument('--execution-config')
    args = parser.parse_args()
    c = shared.read(args.config)
    workspace = args.workspace or c['configured']['workspace_root']
    if mode == 'bind':
        bind(args.config, workspace, args.state, args.output)
        print('PASS_CONFIGURED_S1_BIND')
    elif args.mode == 'check':
        c, rails, prepared, old = shared.preflight(args.config, workspace)
        print('PASS_CONFIGURED_S1_RESTORE', len(prepared), c['expected_rows'])
    else:
        shared.require(args.output and args.execution_config, 'run requires output/execution config')
        module = runtime_module(args.config, workspace)
        asyncio.run(module.run(args.config, workspace, args.output, args.execution_config))


def _host(root):
    """Validate current READY before delegating to the original detached host."""
    h = Path(root)
    spec = shared.read(h / 'S1-TASK.json')
    workspace = Path(spec['workspace'])
    config = shared.read(workspace / spec['config'])
    binding = config['configured']
    shared.validate_ready(workspace, workspace / binding['prepared'],
        workspace / binding['task'], workspace / binding['lane'], workspace / binding['index'], config)
    task = shared.read(workspace / binding['task'])
    lane = shared.read(workspace / binding['lane'])
    old = spec['predecessor']
    for key in ('boot_id', 'session_id', 'service_pid', 'service_id', 'shared_guard', 'shared_guard_inode'):
        shared.require(old['runtime'][key] == lane['service'][key], 'service identity changed: ' + key)
    shared.require(spec['expected_requests'] == config['expected_rows'], 'host request denominator')
    sys.path.insert(0, str(workspace / task['host_helpers_root']))
    import host_ops
    driver = host_ops.driver(workspace)
    shared.require(driver.M == task['container_workspace'], 'existing host container mount contract')
    original_launch = driver.launch
    def launch_with_dependencies(args):
        execution_task = shared.read(args.spec)
        execution_task['readonly_mounts'] = [dict(source=task['model_root'], target=task['model_root'])]
        shared.save(args.spec, execution_task)
        return original_launch(args)
    driver.launch = launch_with_dependencies
    host_ops.driver = lambda _workspace: driver
    module = shared.load('configured_s1_existing_host', shared.pinned(workspace, task['legacy_host']))
    module.main(root)


def host(root):
    """Retain structured terminal failure even when delegated __main__ is bypassed."""
    try:
        return _host(root)
    except BaseException as error:
        import traceback
        shared.save(Path(root) / 'ERROR.json', dict(status='FAILED_HOST',
                    at=time.time(), error=repr(error), traceback=traceback.format_exc()))
        raise
