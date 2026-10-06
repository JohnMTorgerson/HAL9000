"""Query-based memory integration without microphone access or paid requests."""
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from conversation_memory import ConversationMemory, MemorySettings
from llm_client import LLMClient


class SelectiveMemoryIntegrationTests(unittest.TestCase):
    def test_current_speech_selects_one_snapshot_reused_by_api_continuations(self):
        memory = Mock()
        memory.read_context.return_value = ([], 'Selected memory: Torgo is a Vikings fan.')
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(
                content='Understood.', refusal=None))], service_tier='default')
        with patch('llm_client.OpenAI', return_value=client):
            llm = LLMClient('openai', 'gpt-6-luna', openai_api_key='fixture', memory=memory)
        memory.read_context.reset_mock()
        llm.begin_turn('What do you think about football?')
        llm.get_response('What do you think about football?')
        llm.get_response('[EXTERNAL_API_RESPONSE] A football result.')
        memory.read_context.assert_called_once_with(query='What do you think about football?')
        self.assertEqual(client.chat.completions.create.call_count, 2)
        for request in client.chat.completions.create.call_args_list:
            self.assertEqual(request.kwargs['messages'][1]['content'], memory.read_context.return_value[1])
        memory.record_turn.assert_not_called()

    def test_query_and_budget_reach_local_store_without_updater_call(self):
        memory = ConversationMemory.__new__(ConversationMemory)
        memory.available = True
        memory.store, memory.updater, memory.logger = Mock(), Mock(), Mock()
        memory.settings = MemorySettings(soft_tokens=1200)
        facts = json.dumps({'personal': [{'text': 'Torgo is a Vikings fan.'}], 'hal': [], 'topics': []})
        history = [{'role': 'assistant', 'content': 'Shall we discuss football?'}]
        memory.store.recall.return_value = (history, facts)
        recalled, context = memory.read_context('Yes, football.')
        memory.store.recall.assert_called_once_with(query='Yes, football.', token_budget=1200)
        memory.updater.update.assert_not_called()
        self.assertEqual(recalled, history)
        self.assertIn(facts, context)

    def test_worker_processes_tag_only_startup_batch(self):
        store, updater = Mock(), Mock()
        batch = {'cursor': 3, 'new_turns': [], 'tagging_entries': [{'id': 'p1'}]}
        store.next_batch.side_effect = [batch, None]
        completed = threading.Event()
        store.apply.side_effect = lambda *_: (completed.set() or 1)
        memory = ConversationMemory(store, updater, MemorySettings(soft_tokens=1200), Mock(), Mock())
        try:
            self.assertTrue(completed.wait(2), 'startup tagging was not processed')
            updater.update.assert_called_once_with(batch)
            store.apply.assert_called_once_with(batch, updater.update.return_value)
            self.assertTrue(all(call.kwargs == {'token_budget': 1200}
                                for call in store.next_batch.call_args_list))
        finally:
            memory.close()


if __name__ == '__main__':
    unittest.main()
