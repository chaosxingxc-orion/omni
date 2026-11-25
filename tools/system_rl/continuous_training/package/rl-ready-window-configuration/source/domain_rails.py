"""Domain-qualified successor of the original registry rails factory.

Original historical source and its pins remain immutable.
"""
import types
from source_bridge import require,stage_class,sha,canonical

def make_rails(resolver, knowledge, cache, policy):
    for ref in resolver.cfg['code_pins']:
        resolver.reader.raw(ref)
    Base = stage_class(resolver.root, resolver.cfg['package_src'])

    class RegistryStageRails(Base):
        def _component(self, query):
            return resolver.check_query(query)

        def before_model_call(self, stage, query, branch):
            component = self._component(query)
            if stage == 'S1':
                row = self.s1_consumer.queries.get(query.sample_id) if self.s1_consumer else None
                # Legacy cache has no domain. It is reusable only for registered
                # real TRAIN recordings, never a colliding synthetic sample ID.
                require(row is not None,
                        'S1_DYNAMIC_CACHE_MISSING: domain-qualified audio sentence ranking required')
                binding = cache.get('query_source_bindings', {}).get(query.sample_id)
                if binding is None:
                    require(query.domain == 'train_real',
                            'S1_DYNAMIC_CACHE_MISSING: legacy cache is real TRAIN only')
                else:
                    # This rail authenticates an observation, not training
                    # admission. DEV/TEST retain their official split identity.
                    splits={'train_real':'TRAIN','train_synth':'TRAIN',
                            'official_dev':'DEV','official_test':'TEST'}
                    require(binding.get('schema') == 's1-domain-qualified-query-v1' and
                            splits.get(query.domain) == query.split,
                            'S1_DYNAMIC_CACHE_DOMAIN')
                    fields = ('domain','split','source_id','reference_id','recording_file','source_record_sha256')
                    require(all(binding.get(k) == getattr(query,k) for k in fields) and
                            binding.get('source_component') == component and
                            binding.get('audio_sha256') == row['audio_sha256'] == query.audio_sha256 and
                            binding.get('audio_object_sha256') == row['audio_object_sha256'] == sha(canonical(query.audio)),
                            'S1_DYNAMIC_CACHE_SOURCE_IDENTITY')
                require(row['source_component'] == component, 'S1 cache/source mismatch')
            packet = super().before_model_call(stage, query, branch)
            packet['source_binding'] = dict(domain=query.domain, split=query.split, source_id=query.source_id,
                reference_id=query.reference_id, source_component=component, bridge_version=resolver.version)
            return packet

    rails = RegistryStageRails(knowledge, policy, retrieval_cache=cache, provenance_root=resolver.root)
    rails.version = sha(canonical(dict(parent=rails.version, source_bridge=resolver.version)))
    return rails

def policy_rails(policy,resolver,knowledge,cache,runtime):
    original=policy.policy_rails
    scope=dict(original.__globals__,make_rails=make_rails)
    bound=types.FunctionType(original.__code__,scope,original.__name__,original.__defaults__,original.__closure__)
    return bound(resolver,knowledge,cache,runtime)
