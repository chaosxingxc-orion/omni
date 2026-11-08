"""Collect terminal teacher receipts, without dispatch or retry.

A separate --out preserves historical compiled artifacts as well as contacts.
"""
import argparse
import ctypes
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import select
import time
import traceback


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def parse_proposal(raw):
    """Accept one complete object, optionally in one entire JSON code fence.

    Only surrounding whitespace is ignored. Never search for an embedded object
    or silently remove prose, multiple fences, trailing data, or broken fences.
    """
    if not isinstance(raw, str):
        raise ValueError('teacher content must be text')
    body = raw.strip()
    if body.startswith('```'):
        fenced = re.fullmatch(r'```(?:json)?[ \t]*\r?\n(.*?)\r?\n```', body, re.DOTALL)
        if fenced is None:
            raise ValueError('expected one complete JSON fence without surrounding prose')
        body = fenced.group(1)

    def reject_constant(value):
        raise ValueError('non-JSON constant: ' + value)

    proposal = json.loads(body, parse_constant=reject_constant)
    if not isinstance(proposal, dict):
        raise ValueError('teacher proposal must be a single JSON object')
    return proposal


def collect(name, preflight, response, output):
    terminal = read(response / 'result.json')
    dispatch = read(preflight / name / 'dispatch.json')
    out = output / name / 'compiled'
    out.mkdir(exist_ok=True, parents=True)
    checks = dict(
        terminal_received=terminal.get('state') == 'response_received',
        http200=terminal.get('http_status') == 200,
        finish_stop=terminal.get('finish_reasons') == ['stop'],
        config_unchanged=sha(Path(dispatch['argv'][-1])) == dispatch['config_sha256'],
        response_sha_matches=(response / 'response.sse').exists()
        and sha(response / 'response.sse') == terminal.get('response_sha256'))
    error = None
    proposal = None
    try:
        raw = read(response / 'response.json')['choices'][0]['message']['content']
        if not isinstance(raw, str):
            raise ValueError('teacher content must be text')
        raw_path = out / 'proposal-raw.txt'
        raw_bytes = raw.encode('utf-8')
        if raw_path.exists() and raw_path.read_bytes() != raw_bytes:
            raise ValueError('refusing to replace different original response content')
        with raw_path.open('wb') as stream:
            stream.write(raw_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        if not all(checks.values()):
            raise ValueError('terminal receipt/source checks failed')
        proposal = parse_proposal(raw)
        candidates = proposal.get('candidates')
        checks['structured_candidates'] = isinstance(candidates, list)
        if not checks['structured_candidates']:
            raise ValueError('candidates must be a list')
        save(out / 'proposal.json', proposal)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    report = dict(
        algorithm=name, terminal=terminal, checks=checks, parse_error=error,
        candidate_count=len(proposal['candidates']) if error is None else None,
        acceptance='PENDING_MAIN_REVIEW: parsing is not semantic acceptance or reward',
        automatic_retry=False, launch_authorized=False)
    save(out / 'result.json', report)
    paths = [preflight / name / 'dispatch.json', preflight / name / 'request.json',
             preflight / name / 'template.md', Path(dispatch['argv'][-1]),
             response / 'result.json', response / 'response.json', response / 'response.sse',
             Path(__file__).resolve()]
    save(out / 'sourcepins.json', dict(files=[dict(path=str(p), sha256=sha(p))
                                            for p in paths if p.exists()]))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--out', type=Path, help='independent collection root')
    parser.add_argument('--once', action='store_true', help='require both terminal receipts; never wait')
    args = parser.parse_args()
    root = args.preflight.resolve()
    output = args.out.resolve() if args.out else root
    output.mkdir(exist_ok=True, parents=True)
    # Keep the original shared guard even when writing to a separate output root.
    with (root / 'observer.guard').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        roots = {name: Path(read(root / name / 'dispatch.json')['output_root'])
                 for name in ('ace', 'tfgroup')}
        if args.once and not all((r / 'result.json').is_file() for r in roots.values()):
            raise RuntimeError('--once requires all requests terminal; never retry unknown contacts')
        fd = None
        try:
            if not args.once:
                lib = ctypes.CDLL(None, use_errno=True)
                fd = lib.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
                if fd < 0:
                    raise OSError(ctypes.get_errno())
                for response in roots.values():
                    if lib.inotify_add_watch(fd, os.fsencode(response), 0x8 | 0x80) < 0:
                        raise OSError(ctypes.get_errno(), str(response))
            done = set()
            failed = False
            deadline = time.monotonic() + 600
            while len(done) < len(roots):
                for name, response in roots.items():
                    if name in done or not (response / 'result.json').exists():
                        continue
                    try:
                        report = collect(name, root, response, output)
                        failed |= report['parse_error'] is not None
                        done.add(name)
                    except json.JSONDecodeError:
                        if args.once:
                            raise
                    except Exception:
                        save(output / name / 'observer-error.json',
                             dict(traceback=traceback.format_exc(), automatic_retry=False))
                        failed = True
                        done.add(name)
                save(output / 'observer-status.json', dict(
                    as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    completed=sorted(done), total=2, interval_seconds=600, automatic_retry=False,
                    response_roots={name: str(p) for name, p in roots.items()}))
                if len(done) == len(roots):
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    deadline = time.monotonic() + 600
                    remaining = 600
                ready, _, _ = select.select([fd], [], [], remaining)
                if ready:
                    os.read(fd, 1 << 20)
            return int(failed)
        finally:
            if fd is not None:
                os.close(fd)


if __name__ == '__main__':
    raise SystemExit(main())
