"""Probe DNS/TCP/TLS/HTTP without credentials, then optionally gate one batch command."""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import urllib.error


def validate(cfg):
    fields = {'schema', 'endpoint', 'probe_path', 'timeout_seconds', 'total_timeout_seconds', 'accepted_statuses'}
    if not isinstance(cfg, dict) or not fields <= set(cfg) <= fields | {'ca_file', 'proxy_mode'} or cfg['schema'] != 'saea-external-network-v1':
        raise ValueError('exact network probe configuration required')
    url = urllib.parse.urlsplit(cfg['endpoint'])
    if (url.scheme != 'https' or not url.hostname or url.username or url.password
            or url.query or url.fragment or not url.path.endswith('/chat/completions')):
        raise ValueError('credential-free HTTPS chat endpoint required')
    path = cfg['probe_path']
    if not isinstance(path, str) or not path.startswith('/') or path.startswith('//') or any(c in path for c in '\r\n?#'):
        raise ValueError('explicit same-origin credential-free probe path required')
    for key in ('timeout_seconds', 'total_timeout_seconds'):
        if type(cfg[key]) is not int or not 1 <= cfg[key] <= 300:
            raise ValueError('bounded probe timeout required')
    statuses = cfg['accepted_statuses']
    if not isinstance(statuses, list) or not statuses or any(type(s) is not int or not 200 <= s <= 599 for s in statuses):
        raise ValueError('explicit accepted HTTP statuses required')
    if 'ca_file' in cfg and (not isinstance(cfg['ca_file'], str) or not Path(cfg['ca_file']).is_file()):
        raise ValueError('existing CA file required')
    if cfg.get('proxy_mode', 'inherit-environment') not in ('inherit-environment', 'direct'):
        raise ValueError('explicit proxy mode required')
    return url


def error_fields(error):
    result = {'error_type': type(error).__name__}
    if type(getattr(error, 'errno', None)) is int:
        result['errno'] = error.errno
    return result


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def probe(cfg):
    url = validate(cfg); started = time.monotonic()
    mode = cfg.get('proxy_mode', 'inherit-environment')
    result = {'schema': 'saea-external-network-result-v1', 'state': 'pre-dispatch-rejected',
              'generation_dispatched': False, 'automatic_retry': False, 'method': 'HEAD',
              'credential_supplied': False, 'credential_scope': 'origin/API credentials; proxy authentication may come from environment',
              'proxy_mode': mode, 'redirects': 'disabled',
              'transport': 'urllib-default-environment' if mode == 'inherit-environment' else 'urllib-direct',
              'dns_policy': 'resolved by actual transport; target DNS may be delegated to proxy'}
    # Match configured_external_learner: urllib's environment ProxyHandler,
    # including no_proxy, unless direct is explicitly selected. Never pre-resolve
    # the target: CONNECT proxies can resolve it where local DNS cannot.
    handlers = [NoRedirect(), urllib.request.HTTPSHandler(
        context=ssl.create_default_context(cafile=cfg.get('ca_file')))]
    if mode == 'direct':
        handlers.append(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(urllib.parse.urlunsplit(
        ('https', url.netloc, cfg['probe_path'], '', '')), method='HEAD',
        headers={'User-Agent': 'saea-network-preflight/1'})
    response = None
    try:
        try:
            response = urllib.request.build_opener(*handlers).open(request, timeout=cfg['timeout_seconds'])
        except urllib.error.HTTPError as error:
            response = error  # HTTP reached the origin; status is checked below.
        result['http_status'] = response.status
        result['tls_verified'] = True
        if response.status in cfg['accepted_statuses'] and not 300 <= response.status < 400:
            result['state'] = 'ready'
        else:
            result.update(failure_phase='http-head', error_type='UnexpectedHTTPStatus')
    except Exception as error:
        reason = error.reason if isinstance(error, urllib.error.URLError) else error
        result.update(error_fields(reason), failure_phase='request-open')
        result['outer_error_type'] = type(error).__name__
        if isinstance(reason, socket.gaierror):
            result['failure_phase'] = 'transport-dns'
        elif isinstance(reason, ssl.SSLError):
            result['failure_phase'] = 'transport-tls'
    finally:
        if response is not None:
            response.close()
    result['elapsed_seconds'] = time.monotonic() - started
    result['meaning'] = 'network observation only; does not establish provider model capacity or future request delivery'
    return result


def bounded_probe(config_path):
    raw = Path(config_path).read_bytes(); cfg = json.loads(raw); validate(cfg)
    try:
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '_worker', '--config', str(Path(config_path).resolve())],
                                   capture_output=True, timeout=cfg['total_timeout_seconds'], check=True)
        result = json.loads(completed.stdout)
    except subprocess.TimeoutExpired:
        result = {'schema': 'saea-external-network-result-v1', 'state': 'pre-dispatch-rejected',
                  'failure_phase': 'probe-total-deadline', 'error_type': 'TimeoutExpired',
                  'generation_dispatched': False, 'automatic_retry': False, 'credential_supplied': False}
    except (subprocess.CalledProcessError, ValueError):
        result = {'schema': 'saea-external-network-result-v1', 'state': 'pre-dispatch-rejected',
                  'failure_phase': 'probe-worker', 'error_type': 'ProbeWorkerFailure',
                  'generation_dispatched': False, 'automatic_retry': False, 'credential_supplied': False}
    result['config_sha256'] = hashlib.sha256(raw).hexdigest()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'run', '_worker'])
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--receipt', type=Path)
    args, command = parser.parse_known_args(argv)
    args.command = command
    if args.action != 'run' and command:
        parser.error('unexpected command')
    if args.action == '_worker':
        print(json.dumps(probe(json.loads(args.config.read_bytes())))); return 0
    if args.action == 'run' and (args.receipt is None or not args.command):
        parser.error('run requires a fresh receipt and batch command after --')
    # Reject old receipts before probing or executing a command.
    if args.receipt is not None and args.receipt.exists():
        raise ValueError('fresh probe receipt required; never retry a prior batch blindly')
    result = bounded_probe(args.config)
    if args.receipt is not None:
        with args.receipt.open('x') as stream:
            json.dump(result, stream, indent=2); stream.write('\n'); stream.flush()
            import os
            os.fsync(stream.fileno())
    print(json.dumps(result), flush=True)
    if result['state'] != 'ready': return 4
    if args.action == 'run':
        command = args.command[1:] if args.command[0] == '--' else args.command
        if not command: parser.error('nonempty batch command required')
        return subprocess.run(command).returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
