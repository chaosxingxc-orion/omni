"""Official GEPA reads cannot select an arbitrary similarly named successor."""
import json
from dataclasses import asdict, replace
from types import SimpleNamespace
import pytest
from gepa_perception_read import verify_attempt_lineage
from test_direct_only_fit import _plan
from test_direct_repeated_recovery import abort, declare
from speech_aware_evidence_acquisition.core.contact_journal import canonical


def setup(tmp_path):
    repo=tmp_path/'repo';data=tmp_path/'data';name='fixed-pair'
    source=_plan();target=replace(source,attempt_id='recovery-v1')
    abort(data,source,calls=1);declare(data,source,target)
    package=repo/'docs/readiness'/name;package.mkdir(parents=True)
    (package/'generation-plan-registered-v1.json').write_bytes(canonical({'execution_plan':asdict(source)}))
    base=repo/'configs/baseline'/f'{name}.json';base.parent.mkdir(parents=True)
    base.write_bytes(canonical({'arm':'raw-speech-agent-native-tools'}))
    spec={'run_id':source.run_id,'attempt_id':source.attempt_id}
    return repo,data,name,source,target,spec


def test_recovered_reader_retains_registered_plan_and_known_contacts(tmp_path):
    repo,data,name,source,target,spec=setup(tmp_path)
    for plan in (source,target):
        bundle=SimpleNamespace(manifest={'plan':asdict(plan)})
        verify_attempt_lineage(repo,data,name,spec,bundle,plan.attempt_id)


@pytest.mark.parametrize('fault',['foreign-plan','foreign-run','predecessor','receipt','missing','unsafe'])
def test_recovered_reader_refuses_unregistered_or_changed_custody(tmp_path,fault):
    repo,data,name,source,target,spec=setup(tmp_path)
    bundle=SimpleNamespace(manifest={'plan':asdict(target)})
    path=data/'runs'/target.run_id/target.attempt_id/'resume-declaration.json'
    doc=json.loads(path.read_bytes());attempt=target.attempt_id
    if fault=='foreign-plan':bundle.manifest['plan']['protocol_hash']='c'*64
    elif fault=='foreign-run':bundle.manifest['plan']['run_id']='OTHER-RUN'
    elif fault=='predecessor':doc['predecessors'][0]['pin_sha256']='a'*64
    elif fault=='receipt':doc['native_recovery_receipt']['sha256']='b'*64
    elif fault=='missing':path.unlink()
    elif fault=='unsafe':attempt='../escape'
    if fault in ('predecessor','receipt'):path.write_bytes(canonical(doc))
    with pytest.raises((ValueError,FileNotFoundError)):
        verify_attempt_lineage(repo,data,name,spec,bundle,attempt)
