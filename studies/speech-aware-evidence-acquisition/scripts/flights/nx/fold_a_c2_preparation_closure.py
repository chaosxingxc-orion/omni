"""Verify completed Fold A preparation and export a source-preserving ledger closure.

Original encoder registration remains immutable. Derivative ledger omissions are
reported retrospectively, never relabelled as prospective registrations.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / 'src'), str(Path(__file__).parent)]
import fold_a_adopted_retrieval_inputs as inputs
import fold_a_current_agreement as agreement
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
read, sha, raw, confined = inputs.read, inputs.sha, inputs.raw, inputs.confined
PACKAGE = 'docs/readiness/2026-10-31-fold-a-c2-preparation-closure-v1'
REPORT = PACKAGE + '/closure-v1.json'
ALIAS_RUN = 'SAEA-SLURP-INTENT-BOUNDARY-TRAIN-FOLD-A-ALIAS-NPU-PORT-V1'


def verify_derivatives(data, source_rows, retrieval, agreement_complete, ids):
    if (retrieval['schema'] != 'saea-fold-a-adopted-retrieval-inputs-v1' or retrieval['rows'] != 1080
            or retrieval['encoder_run_id'] != inputs.RUN or not retrieval['values_unchanged']
            or retrieval['new_model_contacts'] != 0 or retrieval['new_encoder_contacts'] != 0):
        raise ValueError('completed model-free Fold A derivative required')
    required = {inputs.RECEIPT, inputs.ADOPTION, inputs.SEGMENTS, inputs.CONTENT, inputs.ALIAS_RECEIPT, inputs.PROVENANCE}
    if not required <= set(retrieval['pins']):
        raise ValueError('incomplete derivative pins')
    for rel, digest in retrieval['pins'].items():
        if sha(confined(data, rel)) != digest:
            raise ValueError('derivative pin changed')
    source = {r['id']: r['segments'][0]['vectors']['content'] for r in source_rows}
    segments = [json.loads(x) for x in (data / inputs.SEGMENTS).read_text().splitlines()]
    flat = [json.loads(x) for x in (data / inputs.CONTENT).read_text().splitlines()]
    if (len(source_rows) != 1080 or len(source) != 1080 or set(source) != set(ids)
            or len(segments) != 1080 or len(flat) != 1080
            or {r['id']: r['segments'][0]['vectors']['intent-boundary'] for r in segments} != source
            or {r['id']: r['vectors']['content'] for r in flat} != source):
        raise ValueError('derivative values or complete roster differ')
    alias = read(data / inputs.ALIAS_RECEIPT)
    if (alias['contact_class'] != 'model-free' or alias['output']['row_count'] != 1080
            or alias['output']['output_sha256'] != sha(data / inputs.SEGMENTS)
            or alias['output']['values_unchanged'] is not True
            or alias['output']['max_absolute_delta'] != 0
            or any(alias['output'][k] != 0 for k in ('encoder_contacts', 'model_contacts', 'tokenizer_contacts'))):
        raise ValueError('alias receipt differs')
    provenance = read(data / inputs.PROVENANCE)
    if (provenance['exposure_ledger_row'] != inputs.CONTENT_RUN
            or provenance['row_count'] != 1080 or provenance['parent_sha256'] != sha(data / inputs.SEGMENTS)
            or provenance['sha256'] != sha(data / inputs.CONTENT)):
        raise ValueError('content provenance differs')
    if (agreement_complete['status'] != 'complete' or agreement_complete['rows'] != 1080
            or agreement_complete['artifact_id'] != agreement.ARTIFACT
            or agreement_complete['model_contacts'] != 0
            or agreement_complete['artifact_sha256'] != sha(data / agreement.OUTPUT / 'agreement.jsonl')
            or agreement_complete['receipt_sha256'] != sha(data / agreement.OUTPUT / 'receipt.json')):
        raise ValueError('agreement completion differs')
    return agreement.verify_artifact(data, read(data / agreement.OUTPUT / 'receipt.json'), ids)


def closed_encoder_cells(original, receipt_sha):
    cells = copy.deepcopy(original)
    if len(cells) != 15 or cells[1] != inputs.RUN or cells[8] != 'no' or cells[9:12] != ['1080', '3085', '2.0']:
        raise ValueError('exact original unconsumed registration required')
    cells[8] = 'yes'
    cells[14] += (' COMPLETED: 1080 encoder contacts, 3084.5284375 audio seconds, 0 NPU hours; '
                  'original prospective registration preserved. SEALED receipt SHA256 ' + receipt_sha + '; closure ' + REPORT + '.')
    return cells


AGREEMENT_REG = 'docs/readiness/2026-10-31-fold-a-c2-agreement-v1/registration.json'


def export(repo, encoder_repo, data, retrieval_complete, agreement_completion, out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    import subprocess
    assert_execution_environment()
    head = require_clean_study_repo(repo)
    out = confined(data, str(out))
    if out.exists() or not out.is_relative_to(data.resolve() / 'work/codex'):
        raise ValueError('fresh owned closure output required')
    alias, source_rows = inputs.source_config(encoder_repo, data)
    retrieval, agreed = read(retrieval_complete), read(agreement_completion)
    if retrieval['adoption_sha256'] != sha(data / inputs.ADOPTION):
        raise ValueError('adopted retrieval identity changed')
    ids = [x['sample_id'] for x in read(encoder_repo / inputs.ROSTER)['items']]
    rows = verify_derivatives(data, source_rows, retrieval, agreed, ids)
    original = read(encoder_repo / inputs.REGISTRATION)['plans'][0]['ledger_row_cells']
    line = '| ' + ' | '.join(original) + ' |'
    if (encoder_repo / 'docs/exposure-ledger.md').read_text().splitlines().count(line) != 1:
        raise ValueError('original encoder registration missing or duplicated')
    reg = read(repo / AGREEMENT_REG)
    if (reg['artifact_id'] != agreement.ARTIFACT or reg['rows'] != 1080
            or reg['builder_sha256'] != agreed['builder_sha256']
            or reg['comparator_complete_sha256'] != agreed['comparator_completion_sha256']
            or reg['split_identity_hash'] != original[7]
            or any(reg[k] != 0 for k in ('model_calls','encoder_calls','tokenizer_calls','scoring_jobs'))):
        raise ValueError('agreement prospective registration differs')
    for rel, digest in reg['output_hashes'].items():
        if sha(repo/rel) != digest: raise ValueError('registered comparator hash differs')
    prefix = subprocess.check_output(['git','-C',str(repo),'rev-parse','--show-prefix'],text=True).strip()
    for rel in (AGREEMENT_REG, *reg['output_hashes']):
        old = subprocess.check_output(['git','-C',str(repo),'show',agreed['source_commit']+':'+prefix+rel])
        if old != (repo/rel).read_bytes():
            raise ValueError('agreement source registration/configuration changed')
    ledger = (repo/'docs/exposure-ledger.md').read_text()
    matches = [l for l in ledger.splitlines() if '| '+agreement.ARTIFACT+' |' in l]
    if len(matches) != 1: raise ValueError('one prospective agreement row required')
    original_agree = [c.strip() for c in matches[0].strip('|').split('|')]
    if len(original_agree)!=15 or original_agree[8]!='no' or original_agree[13]!=AGREEMENT_REG:
        raise ValueError('unconsumed agreement row required')
    closed_agree = copy.deepcopy(original_agree); closed_agree[8]='yes'
    closed_agree[14] += ' COMPLETED: 1080 rows; receipt SHA256 '+agreed['receipt_sha256']+'; closure '+REPORT+'.'
    closed = closed_encoder_cells(original,sha(data/inputs.RECEIPT))
    derivatives=[]
    for run, detail in ((ALIAS_RUN,'1080 sealed segment vectors; values unchanged'),
                         (inputs.CONTENT_RUN,'1080 content vectors; values unchanged')):
        if '| '+run+' |' in ledger: raise ValueError('derivative already registered; inspect before closure')
        derivatives.append(['2026-10-31',run,'model-free-check',detail,'Local arithmetic on adopted CPU vectors',
            'slurp TRAIN Fold A','discovery',original[7],'yes','0','0','0.0',
            'Historical Fold A exposure and original encoder costs retained',REPORT,
            'COMPLETED; retrospective ledger synchronization, not prospective registration; no generation admission.'])
    if '| '+inputs.RUN+' |' in ledger: raise ValueError('encoder already synchronized')
    pins=dict(retrieval['pins'])
    for rel in (agreement.OUTPUT+'/receipt.json',agreement.OUTPUT+'/agreement.jsonl'):
        pins[rel]=sha(data/rel)
    report={'schema':'saea-fold-a-c2-preparation-closure-v1','rows':1080,'source_commit':head,
        'registration_sha256':sha(encoder_repo/inputs.REGISTRATION),'protocol_sha256':sha(encoder_repo/inputs.PROTOCOL),
        'agreement_registration_sha256':sha(repo/AGREEMENT_REG),'retrieval_completion_sha256':sha(retrieval_complete),
        'agreement_completion_sha256':sha(agreement_completion),'pins':pins,
        'derivatives_values_unchanged':True,'agreement_rows':len(rows),'new_model_contacts':0,
        'new_encoder_contacts':0,'generation_admitted':False,
        'encoder_ledger_row':closed,'derivative_ledger_rows':derivatives,
        'original_agreement_line':matches[0],'closed_agreement_line':'| '+' | '.join(closed_agree)+' |',
        'derivative_registration_scope':'Retrospective; encoder and agreement original prospective evidence preserved.'}
    header='| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n| '+' | '.join(['---']*15)+' |\n'
    table='\n## Verified Fold A preparation closure\n\n'+header+'\n'.join('| '+' | '.join(r)+' |' for r in [closed,*derivatives])+'\n'
    outputs={inputs.REGISTRATION:(encoder_repo/inputs.REGISTRATION).read_bytes(),
             inputs.PROTOCOL:(encoder_repo/inputs.PROTOCOL).read_bytes(),REPORT:raw(report),
             PACKAGE+'/ledger-addition.md':table.encode()}
    if require_clean_study_repo(repo)!=head: raise ValueError('source changed during closure')
    out.mkdir(parents=True)
    for rel, blob in outputs.items():
        path=out/rel;path.parent.mkdir(parents=True,exist_ok=True);publish_new(path,blob)
    publish_new(out/'complete.json',raw({'files':{rel:sha(out/rel) for rel in outputs},'rows':1080,
        'new_model_contacts':0,'generation_admitted':False}))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','encoder-repo','data','retrieval-complete','agreement-completion','out'):
        p.add_argument('--'+k,type=Path,required=True)
    print(json.dumps(export(**vars(p.parse_args()))))
