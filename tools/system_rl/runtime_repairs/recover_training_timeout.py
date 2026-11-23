"""Recover the scoring prefix of a terminal formal training attempt."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_bytes())


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def reuse_files(previous):
    """Select atomic scorer outputs tied to this immutable generation."""
    selected = load(previous / 'SELECTED.json')
    groups = {str(draw['draw_index']): draw for draw in selected['draws']}
    files = []
    for path in sorted((previous / 'groups').glob('*/GROUP.json')):
        draw = groups.get(path.parent.name)
        require(draw is not None, 'scoring group is outside the selected roster')
        row = load(path)
        generated = load(path.with_name('GENERATED.json'))
        group, = row['online']['behavior_bundle']['groups']
        require(all(value['group_id'] == draw['group_id']
                    and value['query_draw_id'] == draw['query_draw_id']
                    for value in (row, generated, group))
                and group['closed'] is True and group['fresh_on_policy'] is True
                and row['synthetic'] is False and row['policy'] == generated['policy']
                and group['closure']['episode_ids'] == [e['episode_id'] for e in generated['episodes']]
                and all(e['physical_requests_drained'] is True for e in group['episodes']),
                'scoring group does not close its original generation')
        files.append(path)
    group_count = len(files)
    # The original scorer verifies cache key, exact input and payload at read.
    files.extend(sorted((previous / 'reference-scores').glob('*.json')))
    return files, group_count


def resume_scoring(config, cfg, hot, guard, plan_path):
    """Reopen an exact input under the guard already held by the launcher."""
    plan = load(plan_path)
    archived = hot.with_name(hot.name + '-before-scoring-timeout-recovery')
    terminal = load(hot / 'EXIT.json')
    container = load(hot / 'CONTAINER.json')
    intent = load(hot / 'INTENT.json')
    live_boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    require(plan['schema'] == 'formal-actor-scoring-timeout-v1'
            and cfg.get('formal_sampled') is True and cfg['devices'] == list(range(8))
            and cfg['boot_id'] == intent['boot_id'] == plan['boot'] == live_boot
            and intent['guard_inode'] == os.fstat(guard.fileno()).st_ino
            and intent['config_sha256'] == plan['config_sha256'] == file_sha(config)
            and intent['devices'] == cfg['devices']
            and plan['generation_sha256'] == cfg['generation']['generation_inventory_ref']['sha256']
            and plan['checkpoint_sha256'] == cfg['restore_checkpoint']['sha256'],
            'recovery does not match the original boot, guard, configuration or generation')
    require(not archived.exists() and not (hot / 'SCORING-TIMEOUT-RECOVERY.json').exists()
            and terminal['failure'] in ("TimeoutError('bounded same-input forward deadline')",
                                        "TimeoutError('bounded formal actor deadline')")
            and terminal['exit_code'] != 0 and terminal['complete_ranks'] == 0
            and terminal['container_id'] == container['container_id'] == plan['container_id']
            and not any(hot.glob('step-*'))
            and not any(p.exists() for p in (hot/'forward-return', hot/'new-lease'))
            and not any(p.name in {'LOSS.json', 'RESULT.json', 'UPDATE-INTENT.json', 'BACKWARDS.json'}
                        for p in hot.glob('rank-*/*.json')),
            'actor must be terminal from a scoring timeout before any optimizer attempt')
    timeout = plan['timeout_seconds']
    require(type(timeout) is int and cfg['timeout_seconds'] < timeout <= 86400,
            'invalid bounded scoring recovery duration')
    observed, = json.loads(subprocess.check_output(['docker', 'inspect', plan['container_id']]))
    state = observed['State']
    require(observed['Id'] == plan['container_id'] and state['Status'] == 'exited'
            and state['Running'] is False and state['ExitCode'] == terminal['exit_code']
            and any(m['Source'] == str(hot) and m['Destination'] == '/omni-hot'
                    for m in observed['Mounts']),
            'original Docker owner has not exited from the expected hot directory')
    cohort = hot / cfg['generation']['runtime']['cohort_dir']
    require(cohort.resolve().parent == hot and cohort.is_dir(), 'invalid scoring directory')
    files, group_count = reuse_files(cohort)
    require(group_count > 0, 'no closed scoring groups to resume')
    recovery = dict(plan, previous_hot=str(archived), exit_sha256=file_sha(hot/'EXIT.json'),
                    continued_at=time.time(), sealed_groups=group_count, reused_files={})
    work = Path(tempfile.mkdtemp(prefix='hot-scoring-recovery-', dir=hot.parent))
    for src in files:
        require(src.resolve() == src and src.is_file(), 'scoring source must be a regular local file')
        relative = src.relative_to(hot)
        dest = work / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        with dest.open('rb') as file:
            os.fsync(file.fileno())
        recovery['reused_files'][str(relative)] = file_sha(dest)
        require(recovery['reused_files'][str(relative)] == file_sha(src), 'scoring copy changed')
    from dispatch_retained_hf_boundary import save
    save(work / 'SCORING-TIMEOUT-RECOVERY.json', recovery)
    for directory, _, _ in os.walk(work, topdown=False):
        sync_directory(directory)
    hot.rename(archived)
    try:
        work.rename(hot)
    except OSError:
        archived.rename(hot)
        raise
    finally:
        sync_directory(hot.parent)
    return recovery
