"""Retire only a confirmed resident input transport copy. No checkpoint traversal."""
import argparse, hashlib, json, os, stat
from pathlib import Path
import fcntl


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def retire(directory, apply=False):
    directory = Path(directory)
    if directory.resolve() != directory or directory.name != 'update':
        raise ValueError('expected canonical update directory')
    def doc(name):
        p = directory/name
        if p.is_symlink() or not p.is_file():
            raise ValueError('missing direct receipt: ' + name)
        return json.loads(p.read_bytes())
    archive = directory/'actor-input.tar'
    closed = doc('INPUT-CLOSED.json')
    proj = doc('PROJECTION.json')
    intent = doc('INPUT-INTENT.json')
    ref, remote = closed['archive_ref'], closed['remote']
    intent_ref = dict(path=str(directory/'INPUT-INTENT.json'), sha256=digest(directory/'INPUT-INTENT.json'),
                      bytes=(directory/'INPUT-INTENT.json').stat().st_size)
    expected = dict(identity=intent['source'], files=len(intent['members']), archive_sha256=ref['sha256'])
    if (closed['intent_ref'] != intent_ref or proj['source_intent_ref'] != intent_ref or
            remote != expected or proj['archive_sha256'] != ref['sha256'] or proj['files'] != remote['files'] or
            ref['path'] != str(archive)):
        raise ValueError('input transfer and acknowledgment differ')
    if not archive.exists() and not archive.is_symlink():
        return 0
    with archive.open('rb') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        s = os.fstat(f.fileno())
        try: live = archive.lstat()
        except FileNotFoundError: return 0
        if (not stat.S_ISREG(live.st_mode) or not os.path.samestat(s, live) or
                s.st_nlink != 1 or s.st_size != ref['bytes'] or digest(archive) != ref['sha256']):
            raise ValueError('transfer file changed; retain it')
        if apply:
            archive.unlink()
            fd = os.open(directory, os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
        return s.st_size


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path, nargs='+')
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    print(json.dumps({str(d):retire(d, a.apply) for d in a.directory}, sort_keys=True))
