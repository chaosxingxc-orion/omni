"""Explicit candidate-only bank space reservation; no calls or label access."""
import copy
from collections.abc import Mapping

KEY = 'tfrl_supply_allocation'
SCHEMA = 'saea-tfrl-keep-prefix-values-v1'
RECEIPT_SCHEMA = 'saea-supply-variant-arms-v7'
ARMS = ('s1','s2','s3','s4','aa','lock')


def declaration(control_bound):
    if type(control_bound) is not int or control_bound <= 8:
        raise ValueError('control supply bound must exceed eight')
    return {'schema':SCHEMA,'control_max_items':control_bound,
            'candidate_max_items':control_bound-8,'reserved_items':8,
            'order':'existing-render-order','scope':'s1-s4-only',
            'empty_exemplar':'reserve-anyway'}


def validate(value):
    if not isinstance(value,Mapping) or dict(value)!=declaration(value.get('control_max_items')):
        raise ValueError('invalid bank supply allocation declaration')
    return dict(value)


def from_block(block):
    if KEY not in block:return None
    value=validate(block[KEY])
    if 'tfrl_v2_bank' not in block:raise ValueError('allocation requires a declared bank')
    if block.get('decoding_frame',{}).get('max_supplied_items')!=value['control_max_items']:
        raise ValueError('allocation control bound differs from decoding frame')
    return value


def keep_indices(items,allocation):
    """Preserve every non-value part and the existing first N value parts."""
    allocation=validate(allocation);n=0;keep=[]
    for i,part in enumerate(items):
        if isinstance(part,Mapping) and part.get('knowledge_role')=='value':
            n+=1
            if n>allocation['candidate_max_items']:continue
        keep.append(i)
    if n>allocation['control_max_items']:raise ValueError('ordinary supply exceeds declared control bound')
    return keep


def render(messages,allocation):
    result=copy.deepcopy(messages);parts=result[1]['content']
    result[1]['content']=[parts[i] for i in keep_indices(parts,allocation)]
    return result


def project_receipt(receipt):
    """Validate actual candidate selections, then project a COPY for v6 checks."""
    arms=receipt['arms']
    if [a['arm_id'] for a in arms]!=list(ARMS):raise ValueError('allocated bank requires registered K4 arms')
    allocation=validate(arms[-1]['bank_supply_allocation'])
    original=list(arms[-1]['supplied_value_ids'])
    if len(original)>allocation['control_max_items']:raise ValueError('control receipt exceeds supply bound')
    result=copy.deepcopy(receipt);result['schema']='saea-supply-variant-arms-v6'
    for arm in result['arms']:
        if validate(arm.pop('bank_supply_allocation'))!=allocation:raise ValueError('arm allocation differs')
        expected=original[:allocation['candidate_max_items']] if arm['arm_id'] in ARMS[:4] else original
        if (arm['supplied_value_ids']!=expected or type(arm['supplied_item_count']) is not int
                or arm['supplied_item_count']!=len(expected)):
            raise ValueError('actual arm supply differs from declared prefix allocation')
        if arm['max_supplied_items']!=allocation['control_max_items']:
            raise ValueError('arm receipt bound differs from allocation')
        arm['supplied_value_ids']=list(original);arm['supplied_item_count']=len(original)
    return result
