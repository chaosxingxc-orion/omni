"""Consume the original collector's JSON or sealed content-ref JSONL receipts."""
import hashlib
import json
from pathlib import Path


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _bound(row):
    packet = row['packet']
    _need((row['request_id'], row['sample_id'], row['stage']) ==
          (packet['provider_request_id'], packet['sample_id'], packet['stage']),
          'receipt row and original packet differ')
    reply = row['receipt']
    if reply is not None:
        _need(reply['provider_request_id'] == row['request_id'] and
              reply['request_id'] == packet['request_id'] and
              reply['wire_sha256'] == packet['wire_sha256'], 'receipt belongs to another call')
    return row


def collection_calls(version, locate=Path):
    ref = version['receipts']
    path = Path(locate(ref['path']))
    if ref.get('format') is None:
        raw = path.read_bytes()
        _need(hashlib.sha256(raw).hexdigest() == ref['sha256'] and
              len(raw) == ref.get('bytes', len(raw)), 'historical receipt digest')
        calls = json.loads(raw)
        _need(isinstance(calls, list), 'historical receipt array required')
        yield from map(_bound, calls)
        return
    prepared = version.get('content_format') == 'prepared-panel-refs-v1'
    _need(ref['format'] == 'jsonl' and (prepared or version.get('content_format') == 'collection-content-v1'),
          'unknown receipt encoding')
    # Use logical roots for relocated files, and verify every object after resolving it.
    root = Path(ref['path']).parent / 'content'
    def expand(value):
        if isinstance(value, list):
            return [expand(x) for x in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {'$collection_content_v1'}:
            obj = value['$collection_content_v1']
            sha = obj['sha256']
            _need(isinstance(sha,str) and len(sha)==64 and all(c in '0123456789abcdef' for c in sha)
                  and Path(obj['path']) == root/sha[:2]/(sha+'.json'), 'foreign receipt object')
            raw = Path(locate(obj['path'])).read_bytes()
            _need(len(raw)==obj['bytes'] and hashlib.sha256(raw).hexdigest()==sha, 'changed receipt object')
            return expand(json.loads(raw))
        return {k:expand(v) for k,v in value.items()}
    with path.open('rb') as f:
        # Some diagnostic consumers stop after a small prefix. Authenticate the compact
        # stream before exposing any row, without materializing the audio objects.
        digest = hashlib.sha256()
        count = size = 0
        for line in f:
            digest.update(line)
            size += len(line)
            count += 1
        _need(digest.hexdigest() == ref['sha256'] and size == ref['bytes'] and
              count == ref['count'], 'incomplete or changed receipt stream')
        f.seek(0)
        for line in f:
            row=json.loads(line)
            if prepared:
                obj=row['packet_ref']
                raw=Path(locate(obj['path'])).read_bytes()
                _need(len(raw)==obj['bytes'] and hashlib.sha256(raw).hexdigest()==obj['sha256'],
                      'changed prepared panel packet')
                pkt=json.loads(raw)
                _need(pkt['request_id']==row['request_id'], 'prepared request id differs')
                yield _bound(dict(row,packet=dict(pkt,provider_request_id=pkt['request_id'])))
                continue
            for key in ('packet','receipt'):
                if row[key] is not None:
                    # Engine.seal exports the raw SQLite TEXT columns in each JSONL row.
                    row[key]=expand(json.loads(row[key]))
            yield _bound(row)
