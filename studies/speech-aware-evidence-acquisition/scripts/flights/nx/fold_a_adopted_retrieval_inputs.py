"""Build Fold A retrieval inputs from the adopted sealed CPU encoding; no re-encoding."""
import argparse,json,subprocess,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(REPO/'src'),str(Path(__file__).parent)]
from fold_a_retrieval_inputs import read,sha,raw,confined,verify_rows,SEGMENTS,CONTENT,ALIAS_RECEIPT,PROVENANCE,ROSTER,DATASET,CONTENT_RUN
from fold_a_cpu_overlap import RUN,PROTOCOL,REGISTRATION
RECEIPT=f'preparation-runs/{RUN}/encode-v1/receipt.json'
ADOPTION='work/codex/fold-a-cpu-adopt-v1/complete.json'
VECTOR_DIR=str(Path(CONTENT).parent)


def source_config(encoder_repo,data):
    adoption=read(data/ADOPTION);receipt=read(data/RECEIPT)
    head=subprocess.check_output(['git','-c','safe.directory='+str(encoder_repo),'-C',str(encoder_repo),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-c','safe.directory='+str(encoder_repo),'-C',str(encoder_repo),'status','--porcelain']).strip():
        raise ValueError('encoder source dirty')
    expected={'schema':'saea-cpu-preparation-adoption-v1','status':'ADOPTED-SEALED',
        'source_run_id':RUN,'source_attempt_id':'encode-v1','source_study_head':head,
        'source_receipt_sha256':sha(data/RECEIPT),'rows':1080,'logical_encoder_contacts':1080,
        'model_contacts_added':0,'encoder_contacts_added':0,'replayed':False,'downstream_adopted':True}
    if any(adoption.get(k)!=v for k,v in expected.items()):raise ValueError('adopted source differs')
    protocol=read(encoder_repo/PROTOCOL);registration=read(encoder_repo/REGISTRATION)
    plan=registration['plans'][0]['execution_plan'];roster=read(encoder_repo/ROSTER)
    expected={'status':'SEALED','run_id':RUN,'attempt_id':'encode-v1','study_head':head,
        'protocol_sha256':sha(encoder_repo/PROTOCOL),'encoder_inventory_sha256':protocol['encoder']['sha256'],
        'source_inventory_sha256':protocol['sources']['sha256'],'logical_encoder_contacts':1080,
        'cpu_isolation':protocol['cpu_isolation']}
    if any(receipt.get(k)!=v for k,v in expected.items()):raise ValueError('sealed encoder receipt differs')
    attempt=read(data/'attempts'/RUN/'encode-v1.json')
    if (attempt['state']!='FINALIZED' or attempt['run_id']!=RUN or attempt['attempt_id']!='encode-v1'
            or plan['run_id']!=RUN or plan['attempt_id']!='encode-v1' or plan['protocol_hash']!=sha(encoder_repo/PROTOCOL)
            or plan['split_identity_hash']!=roster['split_identity_hash']
            or protocol['roster_sha256']!=sha(encoder_repo/ROSTER) or protocol['expected_unique_spans']!=1080):
        raise ValueError('registered complete encoder identity differs')
    for key in ('sources','encoder'):
        p=confined(data,protocol[key]['path'].removeprefix('${SAEA_DATA_ROOT}/'))
        if sha(p)!=protocol[key]['sha256']:raise ValueError('encoder input changed')
    inventory=read(confined(data,protocol['sources']['path'].removeprefix('${SAEA_DATA_ROOT}/')))
    if (inventory['roster_sha256']!=sha(encoder_repo/ROSTER)
            or inventory['split_identity_hash']!=roster['split_identity_hash']
            or inventory['dataset_config']!={'relpath':DATASET,'sha256':sha(encoder_repo/DATASET)}):
        raise ValueError('real Fold A inventory binding differs')
    query=confined(data,protocol['query_rows_relpath']);index=confined(data,protocol['output_relpath'])
    if (Path(receipt['query_rows_path']).resolve()!=query or sha(query)!=receipt['query_rows_sha256']
            or sha(index)!=receipt['index_sha256'] or adoption['query_rows_sha256']!=sha(query)
            or adoption['index_sha256']!=sha(index)):
        raise ValueError('adopted vector bytes changed')
    rows=[json.loads(line) for line in query.read_text().splitlines()]
    verify_rows(rows,inventory['rows'],protocol['dimension'])
    if (read(index)['output']['rows']!=rows or {r['id'] for r in rows}!={r['sample_id'] for r in roster['items']}):
        raise ValueError('query index or roster differs')
    alias={'schema':'saea-sealed-segment-vector-dimension-alias-v1',
        'source':{'kind':'sealed-segment-vectors','dimension':'content','relpath':protocol['query_rows_relpath'],'sha256':sha(query)},
        'target_dimension':'intent-boundary','expected_rows':1080,'vector_width':protocol['dimension'],
        'output_relpath':SEGMENTS,'receipt_relpath':ALIAS_RECEIPT,
        'source_receipts':[{'root':scope,'relpath':rel,'sha256':sha(root/rel)} for scope,root,rel in
            [('data',data,RECEIPT),('data',data,ADOPTION),('repository',encoder_repo,ROSTER),
             ('repository',encoder_repo,DATASET),('repository',encoder_repo,PROTOCOL),('repository',encoder_repo,REGISTRATION)]]}
    return alias,rows


def prepare(encoder_repo, data, out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    from speech_aware_evidence_acquisition.core.preparation_vectors import prepare_vector_dimension_alias
    sys.path.insert(0, str(REPO / 'scripts'))
    from build_slurp_query_content_derivative import build_query_content_derivative
    assert_execution_environment()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO).strip():
        raise ValueError('derivative implementation must be committed and clean')
    tool_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    out = confined(data, str(out))
    if not out.is_relative_to((data / 'work/codex').resolve()) or out.exists():
        raise ValueError('fresh owned external output required')
    for name in (SEGMENTS, CONTENT, ALIAS_RECEIPT, PROVENANCE):
        if confined(data, name).exists():
            raise ValueError('existing derivative; inspect partial or complete result without replay')
    alias_config, rows = source_config(encoder_repo, data)
    out.mkdir(parents=True)
    publish_new(out / 'intent.json', raw({'tool_head': tool_head,
        'adoption_sha256': sha(data/ADOPTION), 'alias_config': alias_config, 'new_model_contacts': 0}))
    try:
        alias = prepare_vector_dimension_alias(alias_config, repo_root=encoder_repo, data_root=data)
        result = build_query_content_derivative(data, source_relpath=SEGMENTS,
            source_sha256=sha(data / SEGMENTS), out_dir_relpath=VECTOR_DIR,
            out_name=Path(CONTENT).name, ledger_row=CONTENT_RUN, check=False)
        import json
        flat = [json.loads(line) for line in (data / CONTENT).read_text().splitlines()]
        if (len(flat) != 1080 or {r['id']: r['vectors']['content'] for r in flat}
                != {r['id']: r['segments'][0]['vectors']['content'] for r in rows}
                or alias['output']['values_unchanged'] is not True
                or alias['output']['max_absolute_delta'] != 0 or result['row_count'] != 1080):
            raise ValueError('derivative changed source vectors')
        record = {'schema': 'saea-fold-a-adopted-retrieval-inputs-v1', 'rows': 1080,
            'tool_head': tool_head, 'encoder_run_id': RUN, 'adoption_sha256': sha(data/ADOPTION),
            'pins': {name: sha(data / name) for name in
                (RECEIPT, ADOPTION, alias_config['source']['relpath'], SEGMENTS, CONTENT, ALIAS_RECEIPT, PROVENANCE)},
            'values_unchanged': True, 'new_model_contacts': 0, 'new_encoder_contacts': 0,
            'claims_tfrl_admitted': False,
            'next_required': 'Bind real Fold A inputs into a frozen paired confirmation protocol; build agreement and own supplies before admission.'}
        publish_new(out / 'complete.json', raw(record))
        return record
    except BaseException as error:
        publish_new(out / 'failed.json', raw({'reason': str(error), 'automatic_retry': False}))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('encoder-repo','data','out'):p.add_argument('--'+key,type=Path,required=True)
    print(json.dumps(prepare(**vars(p.parse_args()))))
