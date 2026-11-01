"""Load the pristine Reflexion submodule with host-side compatibility bindings.

No files in Reflexion/ are rewritten. AST changes only replace legacy imports
and eager OpenAI defaults while loading modules into a private namespace.
"""
import ast
from functools import lru_cache
from pathlib import Path
from types import ModuleType

SOURCE = Path(__file__).resolve().parent / 'Reflexion' / 'hotpotqa_runs'


def _load(name, bindings, agents=False):
    path = SOURCE / (name + '.py')
    if not path.exists():
        raise RuntimeError('Initialize upstream: git submodule update --init --recursive')
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    # All legacy dependency names are supplied explicitly below.
    tree.body = [node for node in tree.body
                 if not isinstance(node, ast.ImportFrom)
                 or node.module not in {'langchain', 'langchain.llms.base',
                     'langchain.chat_models', 'langchain.chat_models.base',
                     'langchain.schema', 'langchain.agents.react.base',
                     'langchain.docstore.base', 'langchain.prompts',
                     'llm', 'prompts', 'fewshots'}]
    if agents:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                node.args.defaults = [ast.Constant(None)
                    if isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == 'AnyOpenAILLM' else value
                    for value in node.args.defaults]
    module = ModuleType('reflectsteer_upstream_' + name)
    module.__file__ = str(path)
    module.__dict__.update(bindings)
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), module.__dict__)
    return module


@lru_cache(maxsize=1)
def load_upstream():
    from langchain_core.prompts import PromptTemplate
    from langchain_core.language_models.llms import LLM
    from langchain_core.documents import Document
    from langchain_classic.agents.react.base import DocstoreExplorer
    from langchain_community.docstore.wikipedia import Wikipedia
    prompts = _load('prompts', {'PromptTemplate': PromptTemplate})
    fewshots = _load('fewshots', {})
    bindings = dict(vars(prompts), **{k: v for k, v in vars(fewshots).items()
                                    if not k.startswith('__')})
    bindings.update(PromptTemplate=PromptTemplate, AnyOpenAILLM=LLM,
                    DocstoreExplorer=DocstoreExplorer, Wikipedia=Wikipedia,
                    Docstore=object, BaseLLM=LLM, Document=Document)
    agents = _load('agents', bindings, agents=True)
    # Host-side compatibility fix: preserve invalid-action feedback instead of
    # the upstream tuple-unpacking crash. The submodule remains untouched.
    original_parse = agents.parse_action
    agents.parse_action = lambda text: original_parse(text) or (None, None)
    return agents, prompts, fewshots
