#!/usr/bin/env python3
"""Prepare a P candidate using pinned, existing coverage pipeline artifacts.

No model or launcher is invoked. Run on the host with access to the pinned roots.
"""
import argparse
import hashlib
import json
import pathlib
import re
import shutil
import subprocess


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(pathlib.Path(path).read_text())


def write(path, value):
    pathlib.Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def within(root, rel):
    p = pathlib.Path(rel)
    require(not p.is_absolute() and '..' not in p.parts, 'unsafe relative path: ' + rel)
    result = (root / p).resolve()
    require(result.is_relative_to(root.resolve()), 'path escapes root: ' + rel)
    return result


def preflight(c):
    require(c['schema'] == 'saea-p-candidate-preparation-v1', 'invalid schema')
    for key in ('id', 'name', 'run_id', 'generation_name', 'admission_name'):
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]*', c['candidate'][key]) is not None, 'invalid candidate ' + key)
    for key in ('parent_root', 'target_root'):
        require(pathlib.Path(c[key]).is_absolute(), key + ' must be absolute')
    require(pathlib.Path(c['release']['root']).is_absolute(), 'release root must be absolute')
    parent = pathlib.Path(c['parent_root']).resolve(strict=True)
    root = pathlib.Path(c['target_root']).resolve()
    release = pathlib.Path(c['release']['root']).resolve(strict=True)
    require(root.is_absolute() and not root.exists(), 'target root must be new')
    require(not root.is_relative_to(parent) and not parent.is_relative_to(root) and not root.is_relative_to(release) and not release.is_relative_to(root), 'roots overlap')
    manifest_path = release / 'source-manifest.json'
    require(sha(manifest_path) == c['release']['manifest_sha256'], 'release manifest SHA mismatch')
    manifest = read(manifest_path)
    require(manifest['commit'] == c['release']['commit'], 'release commit mismatch')
    template = 'studies/speech-aware-evidence-acquisition/' + c['template']['relpath']
    template_path = within(release, template)
    require(template_path.is_file(), 'template missing from release')
    require(manifest['files'].get(template) == c['template']['sha256'], 'template manifest pin mismatch')
    require(sha(template_path) == c['template']['sha256'], 'template content mismatch')
    # Check every manifest entry, including code, not just the requested template.
    for rel, digest in manifest['files'].items():
        path = within(release, rel)
        require(path.is_file() and sha(path) == digest, 'release file mismatch: ' + rel)
    pins = c['parent_pins']
    for rel, digest in pins.items():
        require(sha(within(parent, rel)) == digest, 'parent pin mismatch: ' + rel)
    required = {'stage-argv.json', 'prepare-argv.json', 'register-argv.json', 'admission-argv.json', 'launcher.json', 'workspace/job-config.private.json', 'config/runtime.json', 'admission-control/config.json'}
    require(required <= pins.keys(), 'missing parent pins')
    require(pathlib.Path(c['asset_parent_root']).is_absolute(), 'asset parent must be absolute')
    legacy = pathlib.Path(c['asset_parent_root']).resolve(strict=True)
    require(not root.is_relative_to(legacy) and not legacy.is_relative_to(root), 'target and asset parent overlap')
    require(sha(legacy / 'assets.json') == c['asset_parent_sha256'], 'asset parent mismatch')
    require((legacy / 'work').is_dir(), 'asset parent work missing')
    for d in c['prepared_caches']:
        require('/' not in d and d not in ('.', '..') and (parent / 'prepared' / d).is_dir(), 'invalid or missing cache: ' + d)
    launcher = read(parent / 'launcher.json')
    require(launcher['image'] == c['image'], 'launcher image mismatch')
    old_release = next(m['source'] for m in launcher['mounts'] if m['target'] == '/opt/saea-release' and m['readonly'])
    old_sha = launcher['environment']['SAEA_RELEASE_MANIFEST_SHA256']
    old_manifest = read(pathlib.Path(old_release) / 'source-manifest.json')
    require(sha(pathlib.Path(old_release) / 'source-manifest.json') == old_sha, 'parent release manifest mismatch')
    require(old_manifest['commit'] == launcher['expected_commit'], 'parent release commit mismatch')
    # A prompt-only successor must preserve the runtime and launch entry points.
    prefix = 'studies/speech-aware-evidence-acquisition/'
    protected = ('src/', 'scripts/analysis/p_only_workload.py', 'scripts/analysis/run_real_engine_candidate.py', 'scripts/analysis/admit_real_engine_candidate.py', 'scripts/environment/configured_job_launch.py')
    old_files = old_manifest['files']
    protected_files = {k for k in old_files if k.startswith(prefix) and any(k[len(prefix):].startswith(p) for p in protected)}
    require(protected_files, 'parent runtime files missing')
    require(all(manifest['files'].get(k) == old_files[k] for k in protected_files), 'runtime source differs from parent')
    for mode in ('stage', 'prepare', 'register', 'admission'):
        argv = read(parent / (mode + '-argv.json'))
        require(c['image'] in argv and any(old_sha in x for x in argv) and any(old_release in x for x in argv), mode + ' source/image mismatch')
    old_name = read(parent / 'workspace/job-config.private.json')['name']
    old_run = read(parent / 'workspace/job-config.private.json')['run_id']
    require(launcher['argv'][launcher['argv'].index('--candidate') + 1] == old_name, 'parent candidate mismatch')
    require(launcher['environment']['SAEA_OBSERVABILITY_RUN_ID'] == old_run, 'parent run mismatch')
    old_generation = launcher['container_name']
    return parent, root, release, launcher, old_release, old_sha, old_name, old_run, old_generation


def prepare(c, dry_run):
    parent, root, release, launcher, old_release, old_sha, old_name, old_run, old_generation = preflight(c)
    if dry_run:
        return {'status': 'preflight-passed', 'model_calls': 0, 'target_created': False, 'release_files_verified': len(read(release / 'source-manifest.json')['files'])}
    name, run = c['candidate']['name'], c['candidate']['run_id']
    replacements = [(str(parent), str(root)), (old_release, str(release)), (old_sha, c['release']['manifest_sha256']), (launcher['expected_commit'], c['release']['commit']), (old_run, run), (old_name, name)]
    def replace(s):
        for a, b in replacements:
            s = s.replace(a, b)
        return s
    root.mkdir(parents=False)
    for d in ('workspace', 'config'):
        shutil.copytree(parent / d, root / d, symlinks=True)
    for d in ('prepared', 'data', 'attempts', 'runs', 'coordination', 'runtime-root', 'admission-root', 'control', 'contacts', 'framework/logs', 'framework/timing', 'queue-state', 'launcher'):
        (root / d).mkdir(parents=True, exist_ok=True)
    for d in c['prepared_caches']:
        require('/' not in d and d not in ('.', '..'), 'invalid cache name')
        shutil.copytree(parent / 'prepared' / d, root / 'prepared' / d, symlinks=True)
    job = read(root / 'workspace/job-config.private.json')
    job.update(name=name, run_id=run)
    job['axis']['template'] = c['template']
    write(root / 'workspace/job-config.private.json', job)
    for mode in ('stage', 'prepare', 'register'):
        argv = [replace(s) for s in read(parent / (mode + '-argv.json'))]
        write(root / (mode + '-argv.json'), argv)
        with (root / (mode + '.log')).open('w') as f:
            subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, check=True)
    reg = json.loads((root / 'register.log').read_text().splitlines()[-1])
    digest = reg['workspace_manifest']['sha256']
    manifest_path = root / 'workspace/manifests' / (digest + '.json')
    require(sha(manifest_path) == digest, 'registered workspace manifest mismatch')
    m = read(manifest_path)
    require(m['source_manifest_sha256'] == c['release']['manifest_sha256'] and m['source_commit'] == c['release']['commit'], 'registered workspace source identity mismatch')
    files = {k: {'path': str(root / 'workspace' / k), 'sha256': v} for k, v in m['files'].items()}
    legacy = pathlib.Path(c['asset_parent_root'])
    require(sha(legacy / 'assets.json') == c['asset_parent_sha256'], 'asset parent mismatch')
    for k, v in read(legacy / 'assets.json')['files'].items():
        files.setdefault(k, v)
    assets = root / 'assets.json'
    write(assets, {'schema': 'saea-experiment-assets-v1', 'files': files})
    pre = 'docs/readiness/' + name + '/'
    def pin(rel):
        return {'relpath': rel, 'sha256': m['files'][rel]}
    admission = {'schema': 'saea-external-fresh-admission-v1', 'candidate': name, 'candidate_pin': pin(pre + 'candidate.json'), 'plans': {k: pin(pre + k + ('-plan-' if k != 'protocol' else '-') + 'registered-v1.json') for k in ('generation', 'fit', 'protocol')}, 'decision_relpath': 'coordination/' + name + '/root-admission-v1.json', 'workspace': '/admission-root/successor-v1', 'runtime_config': {'path': '/config/runtime.json', 'sha256': sha(root / 'config/runtime.json')}}
    write(root / 'admission.json', admission)
    shutil.copytree(legacy / 'work', root / 'work', symlinks=True)
    argv = [replace(s) for s in read(parent / 'admission-argv.json')]
    argv[argv.index('--name') + 1] = c['candidate']['admission_name']
    argv = ['SAEA_EXPERIMENT_MANIFEST_SHA256=' + sha(assets) if s.startswith('SAEA_EXPERIMENT_MANIFEST_SHA256=') else s for s in argv]
    write(root / 'admission-argv.json', argv)
    cid = subprocess.check_output(argv, text=True).strip()
    code = subprocess.check_output(['docker', 'wait', cid], text=True).strip()
    with (root / 'admission.log').open('w') as f:
        subprocess.run(['docker', 'logs', cid], stdout=f, stderr=subprocess.STDOUT, check=True)
    require(code == '0', 'admission failed: ' + cid)
    marker = root / 'admission-root/successor-v1/completion.json'
    digest = read(marker)['workspace_manifest']['sha256']
    admitted_manifest = root / 'admission-root/successor-v1/manifests' / (digest + '.json')
    require(sha(admitted_manifest) == digest, 'admission workspace manifest mismatch')
    admitted = read(admitted_manifest)
    for workspace in (m, admitted):
        require(workspace['source_manifest_sha256'] == c['release']['manifest_sha256'] and workspace['source_commit'] == c['release']['commit'], 'workspace source identity mismatch')
    completion = read(marker)
    require(completion['candidate'] == name, 'admission candidate mismatch')
    for key in ('candidate', 'generation', 'fit', 'protocol'):
        rel = completion['binding'][key]
        require(rel.startswith(pre) and rel in admitted['files'], 'admission binding missing: ' + key)
        require(sha(within(root / 'admission-root/successor-v1', rel)) == admitted['files'][rel], 'admission binding content mismatch: ' + key)
    x = json.loads(replace(json.dumps(launcher)))
    # The generation name identifies the container, not the inherited --out path.
    x['container_name'] = c['candidate']['generation_name']
    (root / 'admission-control').mkdir(exist_ok=True)
    ctrl = read(parent / 'admission-control/config.json')
    ctrl['run_id'] = run
    write(root / 'admission-control/config.json', ctrl)
    x['environment']['SAEA_EXPERIMENT_MANIFEST_SHA256'] = sha(assets)
    x['environment']['SAEA_EXPERIMENT_WORKSPACE_MANIFEST'] = '/experiment-workspace/manifests/' + digest + '.json'
    x['environment']['SAEA_EXPERIMENT_WORKSPACE_SHA256'] = digest
    x['external_admission']['sha256'] = sha(marker)
    x['external_admission']['workspace_manifest'] = str(admitted_manifest)
    for i in x['required_inputs']:
        i['sha256'] = sha(i['path'])
    (root / 'prepared' / (run + '-FIT')).mkdir(exist_ok=True)
    for mount in x['mounts']:
        target = pathlib.Path(mount['target'])
        ancestors = [v for v in x['mounts'] if target != pathlib.Path(v['target']) and target.is_relative_to(pathlib.Path(v['target']))]
        if ancestors:
            ancestor = max(ancestors, key=lambda v: len(pathlib.Path(v['target']).parts))
            dest = pathlib.Path(ancestor['source']) / target.relative_to(ancestor['target'])
            if ancestor['readonly'] and not dest.exists():
                require(dest.is_relative_to(root), 'mount placeholder outside target: ' + str(dest))
                if pathlib.Path(mount['source']).is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.touch()
    write(root / 'launcher.json', x)
    return {'status': 'prepared', 'model_calls': 0, 'admission_container': cid, 'launcher_sha256': sha(root / 'launcher.json'), 'workspace_manifest_sha256': digest}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--execute', action='store_true', help='run CPU preparation and real admission; never launch model')
    args = p.parse_args()
    print(json.dumps(prepare(read(args.config), not args.execute), indent=2))


if __name__ == '__main__':
    main()
