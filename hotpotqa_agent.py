"""Offline HotPotQA ReAct + Reflexion using distractor-context retrieval."""
import json
import re
import string
from typing import TypedDict

from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, ConfigDict, Field

SYSTEM = '''Answer the question using evidence from the passage tools.
Return exactly one action per turn: Search[title], Lookup[keyword], or Finish[answer].
Search retrieves a supplied passage by title (case insensitive); otherwise it suggests
matching titles. Lookup returns the next matching sentence in the last searched passage.
Use Search and Lookup to connect evidence across passages. Finish with a short answer,
not an explanation. Do not invent observations. Available titles are listed below.
'''


def normalize(text):
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


class PassageTools:
    def __init__(self, context):
        if isinstance(context, dict):
            context = zip(context['title'], context['sentences'])
        self.pages = {title: list(sentences) for title, sentences in context}
        self.current = None
        self.offsets = {}

    def step(self, raw):
        match = re.fullmatch(r'(Search|Lookup|Finish)\[([^\[\]\n]+)\]', raw.strip())
        if not match:
            return 'Invalid action. Return Search[title], Lookup[keyword], or Finish[answer].', None
        action, arg = match.groups()
        arg = arg.strip()
        if not arg:
            return 'Action argument cannot be empty.', None
        if action == 'Finish':
            return 'Answer submitted.', arg
        if action == 'Search':
            title = next((t for t in self.pages if t.casefold() == arg.casefold()), None)
            if title is None:
                suggestions = [t for t in self.pages if arg.casefold() in t.casefold()]
                return 'No exact title. Available matches: ' + json.dumps(suggestions or list(self.pages)), None
            self.current, self.offsets = title, {}
            return title + ': ' + ''.join(self.pages[title]), None
        if self.current is None:
            return 'Search for a passage before Lookup.', None
        key = arg.casefold()
        matches = [s for s in self.pages[self.current] if key in s.casefold()]
        offset = self.offsets.get(key, 0)
        self.offsets[key] = offset + 1
        return (matches[offset] if offset < len(matches) else 'No more matching sentences.'), None

    def as_langchain_tools(self):
        class Argument(BaseModel):
            model_config = ConfigDict(extra='forbid', strict=True)
            argument: str = Field(min_length=1, pattern=r'\S')

        @tool('Search', args_schema=Argument)
        def search(argument: str) -> str:
            """Retrieve a supplied passage by title, or suggest matching titles."""
            return self.step(f'Search[{argument}]')[0]

        @tool('Lookup', args_schema=Argument)
        def lookup(argument: str) -> str:
            """Read the next matching sentence from the last searched passage."""
            return self.step(f'Lookup[{argument}]')[0]

        @tool('Finish', args_schema=Argument)
        def finish(argument: str) -> str:
            """Submit a short final answer for external evaluation."""
            return self.step(f'Finish[{argument}]')[0]

        return [search, lookup, finish]


def bounded_generate(lm, messages):
    """Drop oldest complete action/observation pairs, preserving task and memory."""
    messages = list(messages)
    while lm.count_tokens(messages) > lm.max_input_tokens and len(messages) > 3:
        del messages[2:4]
    return lm.generate(messages)


class AgentState(TypedDict):
    messages: list
    memories: list[str]
    trials: list[dict]
    attempt: int
    steps: int
    turns: list[dict]
    answer: str | None
    success: bool
    input_tokens: int
    output_tokens: int


def chat_messages(messages):
    """Adapt LangChain messages to the local Qwen text-action protocol.

    Qwen emits bracket actions; the graph converts these to structured tool calls.
    Tool observations become user turns for this existing chat protocol.
    """
    roles = {'system': 'system', 'human': 'user', 'ai': 'assistant', 'tool': 'user'}
    return [{'role': roles[m.type], 'content': m.content} for m in messages]


class HotPotQAAgent:
    """LangGraph reset -> model -> ToolNode -> evaluate -> reflect/retry workflow."""

    def __init__(self, lm, row, attempts=3, max_steps=8, use_memory=True, trace=False):
        if attempts < 1 or max_steps < 1:
            raise ValueError('attempts and max_steps must be positive')
        self.lm, self.row = lm, row
        self.attempts, self.max_steps = attempts, max_steps
        self.use_memory, self.trace = use_memory, trace
        builder = StateGraph(AgentState)
        builder.add_node('reset', self.reset)
        builder.add_node('model', self.model)
        builder.add_node('tools', self.execute_tools)
        builder.add_node('evaluate', self.evaluate)
        builder.add_node('reflect', self.reflect)
        builder.add_edge(START, 'reset')
        builder.add_edge('reset', 'model')
        builder.add_edge('model', 'tools')
        builder.add_conditional_edges('tools', self.after_tools, ['model', 'evaluate'])
        builder.add_conditional_edges('evaluate', self.after_evaluate, [END, 'reflect', 'reset'])
        builder.add_edge('reflect', 'reset')
        self.graph = builder.compile()

    def reset(self, state):
        self.tools = PassageTools(self.row['context'])
        self.tool_node = ToolNode(self.tools.as_langchain_tools(), handle_tool_errors=True)
        task = 'Question: ' + self.row['question'] + '\nTitles: ' + json.dumps(list(self.tools.pages))
        if state['memories']:
            task += '\nLessons from failed attempts:\n' + '\n'.join(state['memories'][-3:])
        return {'messages': [SystemMessage(content=SYSTEM), HumanMessage(content=task)],
                'attempt': state['attempt'] + 1, 'steps': 0, 'turns': [],
                'answer': None, 'success': False}

    def model(self, state):
        response = bounded_generate(self.lm, chat_messages(state['messages']))
        raw = response['text'].strip()
        match = re.fullmatch(r'(Search|Lookup|Finish)\[([^\[\]\n]+)\]', raw)
        name, argument = match.groups() if match else ('__invalid_request__', '')
        # Every model turn executes exactly one validated tool call. ToolNode returns
        # error ToolMessages for malformed requests; these consume the same budget.
        call = {'name': name, 'args': {'argument': argument.strip()},
                'id': f"call_{state['attempt']}_{state['steps'] + 1}", 'type': 'tool_call'}
        return {'messages': state['messages'] + [AIMessage(content=raw, tool_calls=[call])],
                'steps': state['steps'] + 1,
                'input_tokens': state['input_tokens'] + response['input_tokens'],
                'output_tokens': state['output_tokens'] + response['output_tokens']}

    def execute_tools(self, state):
        update = self.tool_node.invoke({'messages': state['messages']})
        message = update['messages'][-1]
        request = state['messages'][-1]
        call = request.tool_calls[0]
        # Submission is accepted only after successful LangChain tool execution.
        answer = (call['args']['argument'].strip()
                  if call['name'] == 'Finish' and message.status != 'error' else None)
        turn = {'action': request.content, 'observation': message.content,
                'tool': call['name'], 'arguments': call['args'],
                'tool_call_id': message.tool_call_id, 'tool_status': message.status}
        if self.trace:
            print(f"Trial {state['attempt']}, step {state['steps']}: "
                  f"{request.content}\n{message.content}", flush=True)
        return {'messages': state['messages'] + update['messages'],
                'turns': state['turns'] + [turn], 'answer': answer}

    def after_tools(self, state):
        return 'evaluate' if state['answer'] is not None or state['steps'] >= self.max_steps else 'model'

    def evaluate(self, state):
        success = state['answer'] is not None and normalize(state['answer']) == normalize(self.row['answer'])
        trial = {'attempt': state['attempt'], 'answer': state['answer'], 'success': success,
                 'status': 'finished' if state['answer'] is not None else 'step_limit',
                 'trace': state['turns']}
        return {'success': success, 'trials': state['trials'] + [trial]}

    def after_evaluate(self, state):
        if state['success'] or state['attempt'] >= self.attempts:
            return END
        return 'reflect' if self.use_memory else 'reset'

    def reflect(self, state):
        # Evaluator supplies only binary feedback, never the gold answer.
        messages = [SystemMessage(content='The previous attempt failed exact-match evaluation. '
                    'Diagnose evidence or answer-format mistakes and write a concise plan for the retry. '
                    'Use only the observed evidence; the correct answer is not supplied.')]
        messages.extend(state['messages'][1:])
        messages.append(HumanMessage(content='Write a short reflection and a revised plan.'))
        response = bounded_generate(self.lm, chat_messages(messages))
        trials = state['trials'][:-1] + [{**state['trials'][-1], 'reflection': response['text']}]
        return {'memories': state['memories'] + [response['text']], 'trials': trials,
                'input_tokens': state['input_tokens'] + response['input_tokens'],
                'output_tokens': state['output_tokens'] + response['output_tokens']}

    def run(self):
        state = self.graph.invoke(
            {'messages': [], 'memories': [], 'trials': [], 'attempt': 0, 'steps': 0,
             'turns': [], 'answer': None, 'success': False, 'input_tokens': 0, 'output_tokens': 0},
            config={'recursion_limit': self.attempts * (2 * self.max_steps + 3) + 2})
        return {'id': str(self.row.get('id', self.row.get('_id', ''))),
                'question': self.row['question'], 'gold_answer': self.row['answer'],
                'success': state['success'], 'trials': state['trials'],
                'input_tokens': state['input_tokens'], 'output_tokens': state['output_tokens']}


def run_question(lm, row, attempts=3, max_steps=8, use_memory=True, trace=False):
    return HotPotQAAgent(lm, row, attempts, max_steps, use_memory, trace).run()
