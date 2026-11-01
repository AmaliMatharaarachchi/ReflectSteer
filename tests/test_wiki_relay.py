import tempfile
import threading
import time
import unittest
from pathlib import Path

from wiki_relay import RelayDocstore, RelayFailure, serve_once
from original_hotpotqa import QwenOriginalLLM, upstream


class RelayTests(unittest.TestCase):
    def test_round_trip_and_original_lookup(self):
        from langchain_core.documents import Document
        from langchain_classic.agents.react.base import DocstoreExplorer

        class Store:
            queries = []

            def search(self, query):
                self.queries.append(query)
                return Document(page_content='First paragraph.\n\nThe keyword is here.',
                                metadata={'source': 'test'})

        with tempfile.TemporaryDirectory() as directory:
            client = RelayDocstore(directory, timeout=10)
            store = Store()
            stop = threading.Event()

            def worker():
                while not stop.is_set():
                    serve_once(directory, store)
                    stop.wait(0.01)

            thread = threading.Thread(target=worker)
            thread.start()
            try:
                client.ping()
                explorer = DocstoreExplorer(client)
                self.assertEqual(explorer.search('Agent selected title'), 'First paragraph.')
                self.assertIn('The keyword is here.', explorer.lookup('keyword'))
                self.assertEqual(store.queries, ['Agent selected title'])
                self.assertEqual(len(list((Path(directory) / 'responses').glob('*.json'))), 2)
            finally:
                stop.set()
                thread.join()

    def test_error_and_timeout_abort(self):
        class Broken:
            def search(self, query):
                raise OSError('network unavailable')

        with tempfile.TemporaryDirectory() as directory:
            client = RelayDocstore(directory, timeout=0.01)
            with self.assertRaises(RelayFailure):
                client.search('test')
            serve_once(directory, Broken())
            response = next((Path(directory) / 'responses').glob('*.json'))
            self.assertIn('network unavailable', response.read_text())
            # A transport failure must escape upstream's broad except Exception.
            self.assertFalse(issubclass(RelayFailure, Exception))

    def test_qwen_action_label_is_removed_before_original_parser(self):
        class Backend:
            def generate(self, *args, **kwargs):
                return {'text': 'Action 1: Search[VIVA Media AG]'}

        llm = QwenOriginalLLM(backend=Backend())
        text = llm('Question: test\nAction 1:')
        self.assertEqual(upstream()[0].parse_action(text), ('Search', 'VIVA Media AG'))
        self.assertEqual(llm.calls[0]['text'], 'Action 1: Search[VIVA Media AG]')
        self.assertEqual(llm.calls[0]['agent_text'], 'Search[VIVA Media AG]')


if __name__ == '__main__':
    unittest.main()
