"""Versioned display IDs; internal experience identities and values stay intact."""
import hashlib
from .tfgrpo_learning import experience_context, sha

SHORT = 'short-display-id-v1'


def render(candidate, query_id, *, purpose='held-out-evaluation'):
    # Retain all source and evaluation identity checks before changing display.
    experience_context(candidate,query_id,purpose=purpose)
    items=sorted(candidate['items'].items())
    if not items:raise ValueError('nonempty experience required')
    if any(key in value for key,_ in items for _,value in items):
        raise ValueError('stable ID references require a separately verified rendering')
    text='\n'.join(f'[{i}]. {value}' for i,(_,value) in enumerate(items,1))
    body={'schema':'saea-tfgrpo-rendering-v1','rendering':SHORT,'candidate_sha256':candidate['sha256'],
          'mapping':[{'display_id':str(i),'stable_id':key,'value_sha256':hashlib.sha256(value.encode()).hexdigest()}
                     for i,(key,value) in enumerate(items,1)],
          'text_sha256':hashlib.sha256(text.encode()).hexdigest()}
    return text,{**body,'sha256':sha(body)}
