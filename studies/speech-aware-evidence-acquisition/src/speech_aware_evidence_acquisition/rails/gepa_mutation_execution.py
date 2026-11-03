"""One native, durable GEPA reflection; candidate promotion is separate."""
import hashlib
import json
from . import solver_prompt_candidate as prompt
from .tfgrpo_learning import ExperienceLearner, LearningError, canonical, sha
from ..core.atomic_file import publish_new

STAGE = 'gepa_solver_mutation'
SCHEMA = 'saea-gepa-mutation-execution-v1'
RESULT_SCHEMA = 'saea-gepa-mutation-result-v1'


def require(ok, message):
    if not ok:
        raise LearningError(message)


def request(binding):
    return {'schema':'saea-tfgrpo-learning-request-v1','batch_sha256':sha(binding),
        'model_id':binding['model_id'],'learning_config_sha256':binding['learning_config_sha256'],
        'stage':STAGE,'identity':binding['identity'],'messages':binding['messages']}


def execute(learner, binding, *, verify_source, count_tokens):
    require(isinstance(learner, ExperienceLearner) and callable(learner.preflight)
        and learner.max_calls == 1, 'shared learner and native FIT with one contact required')
    require(binding.get('schema') == SCHEMA and learner.model_id == binding['model_id']
        and learner.learning_config_sha256 == binding['learning_config_sha256'],
        'exact mutation model and learning configuration required')
    source = verify_source()
    require(sha(source) == binding['source_custody_sha256'], 'mutation source custody changed')
    contract, parent = source['contract'], source['parent']
    expected = request(binding); key = sha({'stage':STAGE,'identity':binding['identity']})
    with learner._exclusive_batch():
        batch = learner.root/'batch.json'
        if batch.exists():
            require(batch.read_bytes() == canonical(binding), 'mutation binding changed')
        else:
            publish_new(batch, canonical(binding))
        learner._batch = sha(binding)
        learner.last_invocation = {'new_calls':0,'reused_calls':0}
        for path in (learner.root/'contacts').glob('*.intent.json'):
            require(path.name == key+'.intent.json' and path.read_bytes() == canonical(expected),
                'foreign or changed mutation contact')
        for path in (learner.root/'contacts').glob('*.response.json'):
            require(path.name == key+'.response.json'
                and (learner.root/'contacts'/(key+'.intent.json')).exists(), 'orphan mutation response')
        text = learner._call(STAGE,binding['identity'],binding['messages'])
        # _call persists invalid responses and their full cost before validation.
        child = prompt.derive_candidate(text,parent=parent,expected_parent_sha256=parent['candidate_sha256'],
            contract=contract,expected_contract_sha256=contract['contract_sha256'],count_tokens=count_tokens)
        require(set(child['mutation']['evidence_query_ids']) <= set(source['feedback_query_ids']),
            'mutation cites evidence absent from its exact reflection input')
        accounting = learner.accounting()
        require(accounting['unknown_contacts'] == 0 and accounting['observed_contacts'] == 1,
            'one completely observed mutation contact required')
        body = {'schema':RESULT_SCHEMA,'candidate_kind':'solver_prompt',
            'model_id':learner.model_id,'strategy':prompt.STRATEGY,'child':child,
            'source_custody_sha256':sha(source),'batch_sha256':sha(binding),
            'learning_usage':accounting['known_usage'],'accounting':accounting,
            'historical_usage':source['historical_usage'],
            'deployment_accepted':False,'semantic_review_required':True,
            'contact_response_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((learner.root/'contacts').glob('*.response.json'))}}
        result = {**body,'sha256':sha(body)}
        target = learner.root/'candidate.json'
        if target.exists():
            require(target.read_bytes() == canonical(result), 'mutation candidate changed')
        else:
            publish_new(target, canonical(result))
        return result
