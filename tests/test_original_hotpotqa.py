import copy
import os
from pathlib import Path
import unittest

os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(Path(__file__).resolve().parents[1] / 'data/tiktoken'))

from original_hotpotqa import QwenOriginalLLM, make_agents, run_trials, supporting_context, upstream

ROW = {'id': 'test', 'question': 'Where was the inventor born?', 'answer': 'Secretville',
       'context': [['Inventor', ['Evidence one. ', 'Evidence two.']], ['Distractor', ['Unrelated.']]],
       'supporting_facts': {'title': ['Inventor', 'Inventor'], 'sent_id': [0, 1]}}


class Backend:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), kwargs))
        return {'text': next(self.responses), 'input_tokens': 10, 'output_tokens': 3}


class OriginalTests(unittest.TestCase):
    def test_context_matches_notebook_including_repeated_titles(self):
        self.assertEqual(supporting_context(ROW), 'Evidence one. Evidence two.\n\nEvidence one. Evidence two.')

    def test_original_class_prompt_and_reflection_used(self):
        original, prompts, examples = upstream()
        backend = Backend(['Think.', 'Finish[wrong]', 'Check the evidence.', 'Think again.', 'Finish[Secretville]'])
        agents = make_agents([ROW], backend)
        agent = agents[0][0]
        self.assertIs(type(agent), original.CoTAgent)
        expected = prompts.cot_reflect_agent_prompt.format(
            examples=examples.COT, context=supporting_context(ROW), question=ROW['question'],
            reflections='', scratchpad='\nThought:')
        events = []
        result = run_trials(agents, trials=5, on_result=events.append)
        self.assertEqual(backend.calls[0][0], [{'role': 'user', 'content': expected}])
        self.assertEqual(len(backend.calls), 5)
        self.assertEqual(len([e for e in events if e['type'] == 'attempt']), 2)
        self.assertEqual([s['correct'] for s in result['summaries']], [0, 1, 1, 1, 1])
        self.assertEqual(agent.reflections, ['Check the evidence.'])
        self.assertIn('Check the evidence.', backend.calls[3][0][0]['content'])
        self.assertTrue(all('Secretville' not in messages[0]['content'] for messages, _ in backend.calls))
        self.assertTrue(all(options['stop'] == ['\n'] for _, options in backend.calls))
        self.assertTrue(all(options['max_new_tokens'] == 250 for _, options in backend.calls))

    def test_trial_order_and_baseline(self):
        backend = Backend(['Think.', 'Finish[wrong]'] * 4)
        agents = make_agents([ROW, ROW], backend, strategy='base')
        events = []
        run_trials(agents, strategy='base', trials=2, on_result=events.append)
        self.assertEqual([(e['trial'], e['index']) for e in events if e['type'] == 'attempt'],
                         [(1, 0), (1, 1), (2, 0), (2, 1)])
        self.assertEqual(agents[0][2].calls, [])

    def test_no_context_uses_original_examples(self):
        _, prompts, examples = upstream()
        backend = Backend(['Think.', 'Finish[Secretville]'])
        agents = make_agents([ROW], backend, mode='cot-no-context')
        agent = agents[0][0]
        self.assertEqual(agent.context, '')
        self.assertEqual(agent.cot_examples, examples.COTQA_SIMPLE6)
        self.assertIs(agent.reflect_prompt, prompts.cot_simple_reflect_prompt)

    def test_react_original_thought_action_and_docstore(self):
        from langchain_core.documents import Document
        class Store:
            def search(self, query):
                return Document(page_content='Evidence from live-style docstore.')
        original, _, _ = upstream()
        backend = Backend(['Find evidence.', 'Search[Inventor]', 'Answer now.', 'Finish[Secretville]'])
        action = QwenOriginalLLM(backend=backend, max_tokens=100)
        reflection = QwenOriginalLLM(backend=backend, max_tokens=250, stop_tokens=[])
        agent = original.ReactReflectAgent(ROW['question'], ROW['answer'], docstore=Store(),
                                           react_llm=action, reflect_llm=reflection)
        result = run_trials([(agent, action, reflection)], mode='react', trials=1)
        self.assertEqual(result['summaries'][0]['correct'], 1)
        self.assertIn('Observation 1: Evidence from live-style docstore.', agent.scratchpad)
        self.assertEqual(len(backend.calls), 4)
        self.assertTrue(all(options['max_new_tokens'] == 100 for _, options in backend.calls))

    def test_invalid_action_returns_original_feedback(self):
        original, _, _ = upstream()
        self.assertEqual(original.parse_action('invalid'), (None, None))
        backend = Backend(['Think.', 'invalid'])
        agents = make_agents([ROW], backend, mode='react', max_steps=1)
        result = run_trials(agents, mode='react', trials=1)
        self.assertEqual(result['summaries'][0]['halted'], 1)
        self.assertIn('Invalid Action.', agents[0][0].scratchpad)


if __name__ == '__main__':
    unittest.main()
