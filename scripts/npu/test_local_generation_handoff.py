import hashlib
import json
from pathlib import Path

import pytest
from local_generation_handoff import verify_local_release


@pytest.mark.parametrize('fault',[None,'missing_response','wrong_path','changed_artifact','wrong_identity','symlink'])
def test_local_release_rejects_unknown_contacts_and_binding_changes(tmp_path,fault):
    local=tmp_path/'local';logical=Path('/logical/data');expected={'run_id':'run','attempt_id':'a','rows':100}
    root=local/'runs/run/a';root.mkdir(parents=True)
    artifacts={};seal={}
    for key,filename in [('generation_manifest','run.generation-manifest.json'),('score_jobs','score-jobs.jsonl'),('score_ledger_cells','score-ledger-cells.json')]:
        p=root/filename;p.write_text('{}');digest=hashlib.sha256(p.read_bytes()).hexdigest()
        declared=str(logical/'runs/run/a'/filename)
        artifacts[key]={'path':declared,'sha256':digest};seal[key+'_path']=declared;seal[key+'_sha256']=digest
    journal=local/'contacts/run/a';journal.mkdir(parents=True)
    for kind in ('intent','response'):(journal/f'c.{kind}.json').write_text('{}')
    (journal/'semantic-request-digest.fit.json').write_text('{}')
    doc={'schema':'saea-generation-released-v1','generation_sealed':True,'generation_lease_released':True,
         'scoring_complete':False,'consumer_requires_own_lease_and_readiness':True,**expected,
         'released_at_unix':10,'campaign_seconds_to_release':5,'artifacts':artifacts,'seal_receipt':seal}
    if fault=='missing_response':(journal/'c.response.json').unlink()
    if fault=='wrong_path':artifacts['score_jobs']['path']='/elsewhere'
    if fault=='changed_artifact':(root/'score-jobs.jsonl').write_text('changed')
    if fault=='wrong_identity':doc['rows']=99
    if fault=='symlink':
        p=root/'score-jobs.jsonl';p.unlink();outside=tmp_path/'other';outside.write_text('{}');p.symlink_to(outside)
    release=tmp_path/'release.json';release.write_text(json.dumps(doc))
    if fault:
        with pytest.raises(ValueError):verify_local_release(release,local_root=local,logical_data=logical,expected=expected)
    else:
        result=verify_local_release(release,local_root=local,logical_data=logical,expected=expected)
        assert result['native_contacts']==1 and result['cpu_scoring_may_continue']
