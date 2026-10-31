"""Read complete real K4 results with source and official-score receipt checks.

No scorer, tokenizer or model call. The executing study remains immutable.
"""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis')]
from tfrl_current_half_result import read,sha,analyze,validate_recipe,ARMS,SCORER,audit


def pinned(root,pin):
    path=(root/pin['relpath']).resolve()
    if not path.is_relative_to(root.resolve()) or sha(path)!=pin['sha256']:
        raise ValueError('source pin differs')
    return path


def verify_score_receipt(receipt,*,manifest,manifest_sha,outputs_sha,run,attempt,ids,copy_path,score_path,column):
    import hashlib
    expected={'schema':'saea-derived-complete-score-receipt-v1','parent_manifest':str(manifest.resolve()),
        'parent_manifest_sha256':manifest_sha,'parent_run_id':run,'parent_attempt_id':attempt,
        'parent_outputs_sha256':outputs_sha,'roster_count':len(ids),
        'roster_ids_sha256':hashlib.sha256(json.dumps(sorted(ids),separators=(',',':')).encode()).hexdigest(),
        'outputs_copy':str(copy_path.resolve()),'outputs_copy_sha256':sha(copy_path),'scorer_id':SCORER,
        'column_binding':column,'result_sha256':sha(score_path)}
    if receipt!=expected:raise ValueError('official score receipt differs from exact source column')


def load_half(source,data,half,recipe_pin):
    from speech_aware_evidence_acquisition.core.bundle import verify_bundle
    from speech_aware_evidence_acquisition.core.config import compose
    if half not in ('h1','h2'):raise ValueError('fixed half required')
    recipe_path=pinned(source,recipe_pin);recipe=validate_recipe(source,recipe_path)
    run=f'SAEA-SLURP-TFRL-CURRENT-SEED-I1-{half.upper()}-V1';attempt='generation-v1';n=313 if half=='h1' else 312
    if (recipe['run_id']!=run or recipe['attempt_id']!=attempt or recipe['rows']!=n):
        raise ValueError('unexpected real half recipe')
    pp=pinned(source,recipe['sampling_plan']);rp=pinned(source,recipe['roster']);plan=read(pp);roster=read(rp)
    ids=[r['sample_id'] for r in roster['items']]
    if (plan['half']!='R01C-'+half.upper() or plan['run_id']!=run or plan['attempt_id']!=attempt
            or len(ids)!=n or len(set(ids))!=n or [r['row_id'] for r in plan['row_plan']]!=ids
            or plan['roster']['file_sha256']!=sha(rp)
            or plan['roster']['split_identity_hash']!=recipe['split_identity_hash']
            or roster['split_identity_hash']!=recipe['split_identity_hash']):
        raise ValueError('whole-half plan and roster differ')
    directory=data/'analysis'/(run+'-READ');complete_path=directory/'complete.json';complete=read(complete_path)
    manifest=data/'runs'/run/attempt/(run+'.run-manifest.json');bundle=verify_bundle(manifest)
    if (bundle.attempt_state!='FINALIZED' or bundle.manifest['status']!='COMPLETED'
            or bundle.manifest['run_id']!=run or bundle.manifest['attempt']['attempt_id']!=attempt):
        raise ValueError('complete finalized source required')
    check=audit.verify(plan,bundle.outputs_rows,bundle.trace_records)
    fragments=bundle.manifest['config']['fragments']
    cfg=compose(source,*(fragments[k] for k in ('model','dataset','baseline','experiment')))
    if cfg.config_hash!=bundle.manifest['config']['config_hash']:raise ValueError('source configuration differs')
    expected={'schema':'saea-tfrl-current-half-result-v1','rows':n,'run_id':run,'attempt_id':attempt,
        'half':plan['half'],'iteration':plan['iteration'],'recipe_sha256':sha(recipe_path),'plan_sha256':sha(pp),
        'manifest_sha256':sha(manifest),'credit_sha256':sha(directory/'credit.jsonl'),
        'new_model_calls':0,'official_scoring_jobs':7,'request_audit':check}
    if any(complete.get(k)!=v for k,v in expected.items()):raise ValueError('half result completion differs')
    for rel,digest in complete['artifacts'].items():pinned(directory,{'relpath':rel,'sha256':digest})
    columns=directory/'columns';render_path=columns/'render-manifest.json';render=read(render_path)
    if (render['source']['manifest_sha256']!=sha(manifest) or render['source']['roster_sha256']!=sha(rp)
            or render['source']['config_hash']!=cfg.config_hash or render['rows']!=n
            or render['arm_order']!=list(ARMS)):
        raise ValueError('rendered columns differ from source')
    results={}
    for arm in (*ARMS,'vote'):
        score_path=columns/f'arm-{arm}-{SCORER}-scores.json';receipt_path=score_path.with_name(score_path.name+'.receipt.json')
        copy_path=columns/('arm-vote-outputs.jsonl' if arm=='vote' else f'arm-{arm}-first-visit-outputs.jsonl')
        for path in (score_path,receipt_path,copy_path,render_path):
            if complete['artifacts'].get(str(path.relative_to(directory)))!=sha(path):
                raise ValueError('required official artifact missing from completion')
        if arm=='vote':
            column={'arm':'vote','plan_sha256':sha(pp),'render_manifest_sha256':sha(render_path),
                'recipe_sha256':sha(recipe_path),'rule':'plurality intent; support>=2/4 entities; lowest-k intent tie'}
        else:
            entry=render['copies'][arm]
            if Path(entry['path']).resolve()!=copy_path.resolve() or entry['sha256']!=sha(copy_path):
                raise ValueError('render copy identity differs')
            column={'arm':arm,'arm_declaration':next(r for r in render['arm_declarations'] if r['arm_id']==arm),
                'render_manifest_sha256':sha(render_path),'roster_sha256':sha(rp),'visit':'first-solver'}
        verify_score_receipt(read(receipt_path),manifest=manifest,manifest_sha=sha(manifest),
            outputs_sha=bundle.manifest['outputs']['sha256'],run=run,attempt=attempt,ids=ids,
            copy_path=copy_path,score_path=score_path,column=column)
        result=read(score_path)
        if set(result)!={SCORER}:raise ValueError('official scorer changed')
        results[arm]=result[SCORER]
    credit,summary=analyze(plan,results)
    if credit!=[json.loads(line) for line in (directory/'credit.jsonl').read_text().splitlines()]:
        raise ValueError('recorded credit differs from complete official scores')
    if any(complete.get(k)!=v for k,v in summary.items()):raise ValueError('half summary differs from official scores')
    return {'plan':plan,'roster':roster,'results':results,'credit':credit,'bundle':bundle,'config':cfg,
        'source':{'run_id':run,'attempt_id':attempt,'complete_relpath':str(complete_path.relative_to(data)),
            'complete_sha256':sha(complete_path),'manifest_sha256':sha(manifest),'recipe_sha256':sha(recipe_path)}}
