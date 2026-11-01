"""Run upstream Reflexion agents; LangGraph schedules notebook-style trial rounds."""
import hashlib
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.language_models.llms import LLM
from langgraph.graph import END, START, StateGraph
from pydantic import Field

ROOT = Path(__file__).resolve().parent


class QwenOriginalLLM(LLM):
    """One HumanMessage per prompt, matching the original AnyOpenAILLM chat path."""
    backend: Any = Field(exclude=True)
    max_tokens: int = 100
    stop_tokens: list[str] = Field(default_factory=lambda: ['\n'])
    calls: list[dict] = Field(default_factory=list, exclude=True)

    @property
    def _llm_type(self):
        return 'local-qwen-original-reflexion'

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        response = self.backend.generate(
            [{'role': 'user', 'content': prompt}], temperature=0,
            max_new_tokens=self.max_tokens, stop=self.stop_tokens if stop is None else stop)
        self.calls.append({'prompt': prompt, **response})
        return response['text']

    def __call__(self, prompt, **kwargs):
        # The upstream agents use the pre-invoke LangChain callable interface.
        return self.invoke(prompt, **kwargs)


def upstream():
    # Import lazily: this module constructs the original p50k_base token counter.
    from reflexion_adapter import load_upstream
    return load_upstream()


def supporting_context(row):
    """Match CotQA_context.ipynb, including repeated supporting article titles."""
    pages = dict(row['context'])
    return '\n\n'.join(''.join(pages[title]) for title in row['supporting_facts']['title'])


def make_agents(rows, backend, mode='cot-context', strategy='reflexion', max_steps=6):
    original, prompts, examples = upstream()
    strategy = original.ReflexionStrategy(strategy)
    agents = []
    for row in rows:
        if mode == 'react':
            action = QwenOriginalLLM(backend=backend, max_tokens=100)
            reflection = QwenOriginalLLM(backend=backend, max_tokens=250, stop_tokens=[])
            kwargs = dict(question=row['question'], key=row['answer'],
                          max_steps=max_steps, react_llm=action)
            if strategy == original.ReflexionStrategy.NONE:
                agent = original.ReactAgent(**kwargs)
            else:
                agent = original.ReactReflectAgent(**kwargs, reflect_llm=reflection)
        else:
            action = QwenOriginalLLM(backend=backend, max_tokens=250)
            reflection = QwenOriginalLLM(backend=backend, max_tokens=250)
            no_context = mode == 'cot-no-context'
            base = strategy == original.ReflexionStrategy.NONE
            agent = original.CoTAgent(
                question=row['question'], key=row['answer'],
                context='' if no_context else supporting_context(row),
                agent_prompt=(prompts.cot_simple_agent_prompt if base else prompts.cot_simple_reflect_agent_prompt)
                    if no_context else (prompts.cot_agent_prompt if base else prompts.cot_reflect_agent_prompt),
                reflect_prompt=prompts.cot_simple_reflect_prompt if no_context else prompts.cot_reflect_prompt,
                cot_examples=examples.COTQA_SIMPLE6 if no_context else examples.COT,
                reflect_examples=examples.COT_SIMPLE_REFLECTION if no_context else examples.COT_REFLECT,
                action_llm=action, self_reflect_llm=reflection)
        agents.append((agent, action, reflection))
    return agents


class TrialState(TypedDict):
    trial: int
    summaries: list[dict]


def run_trials(agents, mode='cot-context', strategy='reflexion', trials=5, on_result=None):
    """Same trial ordering and solved-question freezing as the upstream notebooks."""
    if trials < 1 or not agents:
        raise ValueError('At least one trial and one agent are required')
    original, _, _ = upstream()
    strategy = original.ReflexionStrategy(strategy)

    def trial_node(state):
        trial = state['trial'] + 1
        for index, (agent, action, reflection) in enumerate(agents):
            if agent.is_correct():
                continue
            a_start, r_start = len(action.calls), len(reflection.calls)
            if mode == 'react':
                if strategy == original.ReflexionStrategy.NONE:
                    agent.run()
                else:
                    agent.run(reflect_strategy=strategy)
            else:
                agent.run(reflexion_strategy=strategy)
            event = {'type': 'attempt', 'index': index, 'trial': trial,
                     'answer': agent.answer, 'success': agent.is_correct(),
                     'finished': agent.is_finished(), 'scratchpad': agent.scratchpad,
                     'reflections': list(getattr(agent, 'reflections', [])),
                     'action_calls': action.calls[a_start:], 'reflection_calls': reflection.calls[r_start:]}
            if on_result:
                on_result(event)
        correct = sum(agent.is_correct() for agent, _, _ in agents)
        summary = {'type': 'trial_summary', 'trial': trial, 'correct': correct,
                   'total': len(agents), 'exact_match': correct / len(agents)}
        if mode == 'react':
            summary['halted'] = sum(agent.is_halted() for agent, _, _ in agents)
        if on_result:
            on_result(summary)
        return {'trial': trial, 'summaries': state['summaries'] + [summary]}

    builder = StateGraph(TrialState)
    builder.add_node('original_trial', trial_node)
    builder.add_edge(START, 'original_trial')
    builder.add_conditional_edges('original_trial', lambda s: END if s['trial'] >= trials else 'original_trial')
    return builder.compile().invoke({'trial': 0, 'summaries': []}, config={'recursion_limit': trials + 2})


def source_hashes():
    directory = ROOT / 'Reflexion' / 'hotpotqa_runs'
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ('agents.py', 'prompts.py', 'fewshots.py')}
