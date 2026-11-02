"""Validate a local-disk generation release without waiting for CPU scoring."""
import hashlib
import json
import math
from pathlib import Path


def verify_local_release(path, *, local_root, logical_data, expected):
    doc = json.loads(Path(path).read_bytes())
    required = {'schema':'saea-generation-released-v1', 'generation_sealed':True,
                'generation_lease_released':True, 'scoring_complete':False,
                'consumer_requires_own_lease_and_readiness':True, **expected}
    if any(type(doc.get(k)) is not type(v) or doc[k] != v for k,v in required.items()):
        raise ValueError('release identity or state differs')
    for key in ('released_at_unix','campaign_seconds_to_release'):
        if type(doc.get(key)) not in (int,float) or not math.isfinite(doc[key]) or doc[key] < 0:
            raise ValueError('release timing invalid')
    root=Path(local_root);logical=Path(logical_data)
    run=expected['run_id'];attempt=expected['attempt_id']
    for value in (run,attempt):
        if not value or Path(value).name!=value or value in ('.','..'):
            raise ValueError('confined run and attempt required')
    files={'generation_manifest':run+'.generation-manifest.json',
           'score_ledger_cells':'score-ledger-cells.json','score_jobs':'score-jobs.jsonl'}
    if set(doc.get('artifacts',{}))!=set(files):raise ValueError('release artifacts incomplete')
    for key,filename in files.items():
        pin=doc['artifacts'][key];declared=str(logical/'runs'/run/attempt/filename)
        if pin['path']!=declared or doc['seal_receipt'].get(key+'_path')!=declared:
            raise ValueError('release path differs')
        target=root/'runs'/run/attempt/filename
        if target.resolve()!=target.absolute():raise ValueError('release path redirected')
        digest=hashlib.sha256(target.read_bytes()).hexdigest()
        if digest!=pin['sha256'] or doc['seal_receipt'].get(key+'_sha256')!=digest:
            raise ValueError('sealed bytes differ')
    journal=root/'contacts'/run/attempt
    if journal.resolve()!=journal.absolute():raise ValueError('journal path redirected')
    ids=lambda kind:{p.name.removesuffix('.'+kind+'.json') for p in journal.glob('*.'+kind+'.json')}
    intents,responses,fits=ids('intent'),ids('response'),ids('fit')
    # FIT files use a semantic request digest, unlike transport request IDs.
    # Exact FIT payload binding is checked by the scientific custody reader.
    if not intents or intents!=responses or len(fits)!=len(intents):
        raise ValueError('unresolved or unmatched native contacts')
    return {'generation_release_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'rows':expected['rows'],'native_contacts':len(intents),'unknown_contacts':0,
            'cpu_scoring_may_continue':True,'successor_requires_own_admission_and_lease':True}
