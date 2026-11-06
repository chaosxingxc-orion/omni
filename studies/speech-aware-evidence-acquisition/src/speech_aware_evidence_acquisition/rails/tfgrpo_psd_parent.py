"""Replay reviewed target-bound parent before allowing another PSD learning round."""
import json
from pathlib import Path
from . import external_exploration as external
from .tfgrpo_learning import empty_experience, sha
from .tfgrpo_consumption import checked_pin, validate_params


def verify(parent, target, model_id, binding):
    if binding is None:
        if parent != empty_experience(model_id):
            raise ValueError('nonempty or exposed parent requires reviewed PSD parent binding')
        return []
    if not isinstance(binding, dict) or set(binding) != {'schema','repo','data','params'} or binding['schema'] != 'saea-psd-tfgrpo-parent-v1':
        raise ValueError('exact reviewed PSD parent binding required')
    repo=Path(binding['repo']).resolve(strict=True); data=Path(binding['data']).resolve(strict=True)
    params=external.read(external.checked(binding['params']))
    validate_params(params)
    kind=params.get('source_kind')
    if kind == 'external-tfgrpo-v1':
        from .tfgrpo_external_release import verify_source, render
    elif kind == 'external-tfgrpo-reference-v1':
        from .tfgrpo_reference_release import verify_source, render
    else: raise ValueError('reviewed TF-GRPO parent required')
    candidate=json.loads(checked_pin(data,params['candidate']).read_bytes())
    if candidate.get('psd_target') != target or candidate.get('model_id') != model_id:
        raise ValueError('parent target or actor differs')
    verify_source(params,repo=repo,data=data,candidate=candidate)
    base=checked_pin(repo/'configs',params.get('contract_control',params['base_template'])).read_text()
    if checked_pin(repo/'configs',params['template']).read_text() != render(candidate,base,params['consumer_ids']):
        raise ValueError('parent reviewed template differs')
    body={'schema':'saea-tfgrpo-experience-v1','model_id':model_id,'items':candidate['items'],
          'learned_query_ids':candidate['learned_query_ids'],'parent_sha256':candidate['sha256'],
          'status':'candidate','deployment_accepted':False}
    if parent != {**body,'sha256':sha(body)}:
        raise ValueError('parent must exactly derive from reviewed candidate and exposure')
    own = candidate.get('usage_identity', candidate.get('request_sha256'))
    ancestors=candidate.get('ancestor_usage_identities',[])
    if not isinstance(own,str) or not isinstance(ancestors,list) or len(set(ancestors+[own])) != len(ancestors)+1:
        raise ValueError('distinct parent accounting identities required')
    return ancestors+[own]
