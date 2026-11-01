"""Bind a learned experience candidate to the existing Solver template path.

This release is for development evaluation only. It changes no knowledge
supply, perception, parameter, answer selection or task reward.
"""
import hashlib
import json
from pathlib import Path
import re
from ..core.atomic_file import publish_new
from .tfgrpo_learning import validate_experience, experience_context, canonical

COMPONENT = 'tfgrpo-solver-experience'
FIELDS = {'candidate', 'source_manifest', 'model_id', 'base_template', 'template', 'consumer_ids', 'purpose'}
SEPARATOR = '\n\nLearned evidence-use experiences (apply only when their conditions hold):\n'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_pin(root, pin):
    if not isinstance(pin, dict) or set(pin) != {'relpath','sha256'}:
        raise ValueError('exact artifact pin required')
    rel = pin['relpath']
    if (not isinstance(rel,str) or not rel or Path(rel).is_absolute()
            or '..' in Path(rel).parts or '\\' in rel
            or not isinstance(pin['sha256'],str) or not re.fullmatch('[a-f0-9]{64}',pin['sha256'])):
        raise ValueError('confined artifact and digest required')
    path = (Path(root)/rel).resolve()
    if not path.is_relative_to(Path(root).resolve()) or digest(path) != pin['sha256']:
        raise ValueError('experience artifact changed or escaped root')
    return path


def validate_params(params):
    optional={'target_arm','rendering','render_manifest','compact_params'}
    if not isinstance(params,dict) or not FIELDS <= set(params) or set(params)-FIELDS-optional or params['purpose'] != 'development-evaluation':
        raise ValueError('development-only Solver experience binding required')
    if 'target_arm' in params and params['target_arm'] != 'learned':
        raise ValueError('only the learned paired Solver arm is supported')
    if ('rendering' in params)!=('render_manifest' in params) or ('rendering' in params and params['rendering']!='short-display-id-v1'):
        raise ValueError('explicit short rendering and manifest required')
    if 'compact_params' in params:
        compact=params['compact_params']
        if params.get('target_arm')!='learned' or 'rendering' in params or not isinstance(compact,dict) or 'compact_params' in compact or 'target_arm' in compact:
            raise ValueError('exact three-arm rendering declaration required')
        validate_params(compact)
        if compact.get('rendering')!='short-display-id-v1' or any(compact[k]!=params[k] for k in ('candidate','source_manifest','model_id','base_template','consumer_ids','purpose')):
            raise ValueError('same candidate required for display-only contrast')
    ids = params['consumer_ids']
    if (not isinstance(ids,list) or not ids or len(ids)!=len(set(ids))
            or any(not isinstance(x,str) or not x.startswith('slurp:train:') for x in ids)):
        raise ValueError('unique TRAIN development consumer identities required')
    if not isinstance(params['model_id'],str) or not params['model_id']:
        raise ValueError('source frozen model required')
    for key in ('candidate','source_manifest','base_template','template',*(['render_manifest'] if 'render_manifest' in params else [])):
        pin=params[key]
        if (not isinstance(pin,dict) or set(pin)!={'relpath','sha256'}
                or not isinstance(pin['relpath'],str) or not pin['relpath']
                or Path(pin['relpath']).is_absolute() or '..' in Path(pin['relpath']).parts
                or '\\' in pin['relpath'] or not re.fullmatch('[a-f0-9]{64}',str(pin['sha256']))):
            raise ValueError('exact confined source and template pins required')
    if any(not params[k]['relpath'].startswith('probes/') for k in ('base_template','template')):
        raise ValueError('existing probe template path required')
    if 'render_manifest' in params and not params['render_manifest']['relpath'].startswith('probes/'):
        raise ValueError('render manifest must be a probe artifact')
    return params


def verify_learning_seal(params, *, data, candidate):
    """Require the shared verifier and the exact proposal in finalized trace."""
    from ..core.bundle.verify import verify_bundle
    manifest = checked_pin(data, params['source_manifest'])
    bundle = verify_bundle(manifest)
    proposals = [record['org_learning_proposal'] for record in bundle.trace_records.values()
                 if 'org_learning_proposal' in record]
    if bundle.attempt_state != 'FINALIZED' or len(proposals) != 1:
        raise ValueError('one finalized model-learning proposal required')
    proposal = proposals[0]
    expected_pin = {'root':'data', **params['candidate'],
                    'experience_sha256':candidate['sha256'], 'deployment_accepted':False}
    if (proposal.get('schema') != 'saea-tfgrpo-flight-proposal-v1'
            or proposal.get('result_kind') != 'artifact'
            or proposal.get('candidate') != expected_pin
            or proposal.get('batch_sha256') != candidate['batch_sha256']
            or proposal.get('contact_response_sha256') != candidate.get('contact_response_sha256')
            or proposal.get('usage') != candidate.get('learning_usage')):
        raise ValueError('candidate differs from sealed learning proposal')
    return bundle


def expected_text(candidate, base, consumer_ids, *, rendering=None):
    validate_experience(candidate,candidate['model_id'])
    if (candidate.get('status')!='candidate' or candidate.get('deployment_accepted') is not False
            or not candidate.get('items') or not candidate.get('batch_sha256')
            or candidate.get('accounting',{}).get('unknown_contacts') != 0):
        raise ValueError('complete nonempty model-learned candidate required')
    count=candidate.get('planned_learning_calls')
    accounting=candidate['accounting']
    if (type(count) is not int or count < 1
            or accounting.get('attempted_contacts') != count
            or accounting.get('observed_contacts') != count
            or not isinstance(candidate.get('contact_response_sha256'),dict)
            or len(candidate['contact_response_sha256']) != count
            or candidate.get('learning_usage') != accounting.get('known_usage')
            or not isinstance(candidate.get('learning_usage'),dict)
            or candidate['learning_usage'].get('calls') != count):
        raise ValueError('complete actual learning accounting required')
    if rendering is None:
        texts=[experience_context(candidate,sid,purpose='held-out-evaluation') for sid in consumer_ids]
    elif rendering=='short-display-id-v1':
        from .tfgrpo_rendering import render
        texts=[render(candidate,sid)[0] for sid in consumer_ids]
    else:raise ValueError('unknown rendering version')
    if len(set(texts))!=1: raise ValueError('fixed experience context required')
    return base+SEPARATOR+texts[0]+'\n'


def verify(params, *, repo, data, model_id, consumer_ids):
    validate_params(params)
    if model_id != params['model_id'] or set(consumer_ids) != set(params['consumer_ids']):
        raise ValueError('consumer model or exact development roster differs')
    candidate=json.loads(checked_pin(data,params['candidate']).read_bytes())
    validate_experience(candidate,model_id)
    verify_learning_seal(params,data=data,candidate=candidate)
    base=checked_pin(Path(repo)/'configs',params['base_template']).read_text()
    template=checked_pin(Path(repo)/'configs',params['template']).read_text()
    if 'rendering' in params:
        from .tfgrpo_rendering import render
        _,expected_manifest=render(candidate,params['consumer_ids'][0])
        if json.loads(checked_pin(Path(repo)/'configs',params['render_manifest']).read_bytes())!=expected_manifest:
            raise ValueError('rendering map differs from immutable experience')
    if template != expected_text(candidate,base,params['consumer_ids'],rendering=params.get('rendering')):
        raise ValueError('Solver template differs from exact learned experience')
    return candidate['sha256']


def verify_for_flight(values, *, repo, data, model_id, consumer_ids):
    block=values.get('native_episode',{}).get('consumption_module',{})
    matches=[e for e in block.get('components',[]) if e.get('component_id')==COMPONENT and e.get('state')=='on']
    if not matches:return None
    if len(matches)!=1:raise ValueError('one learned Solver component required')
    params=matches[0]['params']
    if params.get('target_arm'):
        if values['phase_material']['solver']['instruction_template'] != params['base_template']['relpath']:
            raise ValueError('paired evaluation changed the base Solver template')
        if not paired_layout_matches(values,params):raise ValueError('exact source-bound Solver arms required')
        if 'compact_params' in params:
            verify(params['compact_params'],repo=repo,data=data,model_id=model_id,consumer_ids=consumer_ids)
    elif values['phase_material']['solver']['instruction_template'] != params['template']['relpath']:
        raise ValueError('declared experience template not actually selected')
    return verify(params,repo=repo,data=data,model_id=model_id,consumer_ids=consumer_ids)


def prepare(*, candidate_path, source_manifest, repo, data, base_template, template, consumer_ids, model_id, rendering=None):
    """Write a deterministic candidate template; no admission or model contact."""
    path=Path(candidate_path).resolve(); data=Path(data).resolve(); repo=Path(repo).resolve()
    if not path.is_relative_to(data):raise ValueError('owned candidate required')
    manifest=Path(source_manifest).resolve()
    if not manifest.is_relative_to(data):raise ValueError('owned learning manifest required')
    base=repo/'configs'/base_template
    candidate=json.loads(path.read_bytes());validate_experience(candidate,model_id)
    text=expected_text(candidate,base.read_text(),consumer_ids,rendering=rendering)
    params={'candidate':{'relpath':str(path.relative_to(data)),'sha256':digest(path)},
        'source_manifest':{'relpath':str(manifest.relative_to(data)),'sha256':digest(manifest)},
        'model_id':model_id,'base_template':{'relpath':base_template,'sha256':digest(base)},
        'template':{'relpath':template,'sha256':hashlib.sha256(text.encode()).hexdigest()},
        'consumer_ids':list(consumer_ids),'purpose':'development-evaluation'}
    if rendering is not None:
        from .tfgrpo_rendering import render
        _,manifest_doc=render(candidate,consumer_ids[0])
        manifest_rel=template+'.render.json'
        manifest_bytes=(json.dumps(manifest_doc,sort_keys=True,indent=2)+'\n').encode()
        params.update(rendering=rendering,render_manifest={'relpath':manifest_rel,'sha256':hashlib.sha256(manifest_bytes).hexdigest()})
    validate_params(params)
    verify_learning_seal(params,data=data,candidate=candidate)
    target=(repo/'configs'/template).resolve()
    if not target.is_relative_to(repo/'configs'):raise ValueError('template escaped repository')
    target.parent.mkdir(parents=True,exist_ok=True)
    if rendering is not None:publish_new(repo/'configs'/manifest_rel,manifest_bytes)
    publish_new(target,text.encode())
    verify(params,repo=repo,data=data,model_id=model_id,consumer_ids=consumer_ids)
    return params


def paired_layout_matches(values,params):
    """Narrow shape shared by runtime verification and checkpoint eligibility."""
    validate_params(params)
    group=values['native_episode']['roles']['solver'].get('template_variants',{})
    arms=group.get('arms',[])
    specs=[('learned',params)]
    if 'compact_params' in params:specs.append(('compact',params['compact_params']))
    if group.get('node')!='solver' or len(arms)!=len(specs)+1:return False
    for arm,(name,p) in zip(arms,specs):
        pin={'relpath':'configs/'+p['template']['relpath'],'sha256':p['template']['sha256']}
        if arm.get('arm_id')!=name or arm.get('instruction_template')!=pin or arm.get('deliver') or arm.get('replicate_of'):return False
    lock=arms[-1]
    return (lock.get('arm_id')=='lock' and lock.get('deliver') is True and not lock.get('instruction_template')
        and not lock.get('replicate_of') and values['phase_material']['solver']['instruction_template']==params['base_template']['relpath'])
