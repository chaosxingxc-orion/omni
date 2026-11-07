"""Restore the moved vLLM tokenizer import for older lm-format-enforcer.

The alias is installed only if the original module cannot be found.  It does
not modify vLLM, the tokenizer, or the structured-output implementation.
"""
import importlib.abc
import importlib.machinery
import importlib.util
import sys


class _TokenizerAlias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    name = 'vllm.transformers_utils.tokenizer'

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.name:
            return None
        # Other finders get first refusal: preserve a real legacy module.
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        from vllm.tokenizers.mistral import MistralTokenizer
        module.MistralTokenizer = MistralTokenizer


sys.meta_path.append(_TokenizerAlias())


class _TransformersCompat(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Supply a removed re-export just before the old LMFE module imports it."""
    name = 'lmformatenforcer.integrations.transformers'

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.name:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        self.loader = spec.loader
        spec.loader = self
        return spec

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        import transformers
        import transformers.tokenization_utils as tokenizer_module
        if not hasattr(tokenizer_module, 'PreTrainedTokenizerBase'):
            tokenizer_module.PreTrainedTokenizerBase = transformers.PreTrainedTokenizerBase
        self.loader.exec_module(module)


sys.meta_path.insert(0, _TransformersCompat())
