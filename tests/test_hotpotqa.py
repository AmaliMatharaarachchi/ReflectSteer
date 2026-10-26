import copy
import unittest
from pathlib import Path

from hotpotqa_agent import HotPotQAAgent, PassageTools, bounded_generate, normalize, run_question
from run_hotpotqa_qwen import load_rows

ROW = {'id': 'sample', 'question': 'Where was the inventor born?',
       'answer': 'Secretville', 'context': [['Inventor', ['Born in a village. ', 'Worked in Paris.']]]}


class ScriptedLM:
    max_input_tokens = 100000

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.prompts = []

    def count_tokens(self, messages):
        return sum(len(m['content']) for m in messages)

    def generate(self, messages):
        self.prompts.append(copy.deepcopy(messages))
        return {'text': next(self.outputs), 'input_tokens': 10, 'output_tokens': 3}


class AgentTests(unittest.TestCase):
    def test_graph_executes_langchain_tools(self):
        lm = ScriptedLM(['Search[Inventor]', 'Lookup[Paris]', 'Finish[Secretville]'])
        agent = HotPotQAAgent(lm, ROW)
        self.assertTrue({'reset', 'model', 'tools', 'evaluate', 'reflect'} <=
                        set(agent.graph.get_graph().nodes))
        result = agent.run()
        turns = result['trials'][0]['trace']
        self.assertEqual([t['tool'] for t in turns], ['Search', 'Lookup', 'Finish'])
        self.assertTrue(all(t['tool_status'] == 'success' for t in turns))
        self.assertEqual(turns[1]['observation'], 'Worked in Paris.')
        self.assertEqual(len({t['tool_call_id'] for t in turns}), 3)

    def test_tool_error_recovery(self):
        lm = ScriptedLM(['not an action', 'Finish[   ]', 'Finish[Secretville]'])
        result = run_question(lm, ROW, max_steps=3)
        self.assertTrue(result['success'])
        self.assertEqual([t['tool_status'] for t in result['trials'][0]['trace']],
                         ['error', 'error', 'success'])

    def test_retry_resets_passage_state(self):
        lm = ScriptedLM(['Search[Inventor]', 'Finish[wrong]', 'Search before lookup.',
                         'Lookup[Paris]', 'Finish[Secretville]'])
        result = run_question(lm, ROW, max_steps=2)
        self.assertIn('before Lookup', result['trials'][1]['trace'][0]['observation'])

    def test_langchain_schema_rejects_extra_arguments(self):
        from pydantic import ValidationError
        search = PassageTools(ROW['context']).as_langchain_tools()[0]
        with self.assertRaises(ValidationError):
            search.invoke({'argument': 'Inventor', 'unexpected': True})

    def test_tools(self):
        tools = PassageTools(ROW['context'])
        self.assertIn('before Lookup', tools.step('Lookup[Paris]')[0])
        self.assertIn('Born', tools.step('Search[inventor]')[0])
        self.assertEqual(tools.step('Lookup[paris]')[0], 'Worked in Paris.')
        self.assertEqual(tools.step('Lookup[paris]')[0], 'No more matching sentences.')
        self.assertIsNone(tools.step('Finish[]')[1])
        self.assertIsNone(tools.step('Search[Inventor]\nFinish[Paris]')[1])

    def test_reflection_retry_and_no_gold_leak(self):
        lm = ScriptedLM(['Finish[wrong]', 'Check the birthplace evidence.', 'Finish[Secretville]'])
        result = run_question(lm, ROW)
        self.assertTrue(result['success'])
        self.assertEqual(len(result['trials']), 2)
        self.assertIn('Check the birthplace', lm.prompts[-1][1]['content'])
        self.assertNotIn('Secretville', str(lm.prompts))
        self.assertEqual(result['input_tokens'], 30)
        self.assertEqual(result['output_tokens'], 9)

    def test_no_memory_and_step_limit(self):
        lm = ScriptedLM(['bad', 'Finish[wrong]'])
        result = run_question(lm, ROW, attempts=2, max_steps=1, use_memory=False)
        self.assertFalse(result['success'])
        self.assertEqual(result['trials'][0]['status'], 'step_limit')
        self.assertNotIn('reflection', result['trials'][0])
        self.assertEqual(len(lm.prompts[-1]), 2)

    def test_success_stops_and_memory_resets(self):
        lm = ScriptedLM(['Finish[Secretville]', 'Finish[Secretville]'])
        self.assertEqual(len(run_question(lm, ROW)['trials']), 1)
        run_question(lm, ROW)
        self.assertEqual(lm.prompts[0], lm.prompts[1])

    def test_context_pruning(self):
        lm = ScriptedLM(['ok'])
        lm.max_input_tokens = 5
        messages = [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'q'},
                    {'role': 'assistant', 'content': 'old'}, {'role': 'user', 'content': 'old'},
                    {'role': 'assistant', 'content': 'a'}, {'role': 'user', 'content': 'o'}]
        bounded_generate(lm, messages)
        self.assertEqual([m['content'] for m in lm.prompts[0]], ['s', 'q', 'a', 'o'])
        self.assertEqual(len(messages), 6)

    def test_normalization(self):
        self.assertEqual(normalize('The Paris!'), normalize('paris'))

    def test_converted_dataset(self):
        rows = load_rows(Path(__file__).resolve().parents[1] / 'data/hotpotqa_sample.json')
        self.assertEqual(len(rows), 100)
        for row in rows:
            tools = PassageTools(row['context'])
            self.assertGreater(len(tools.pages), 0)
            self.assertTrue(all(isinstance(s, str) for page in tools.pages.values() for s in page))


if __name__ == '__main__':
    unittest.main()
