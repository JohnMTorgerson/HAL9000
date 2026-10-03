"""Persistent memory boundaries, crash recovery and background behavior; no paid calls."""
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import httpx
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from conversation_memory import ConversationMemory, MemorySettings, MemoryUpdater, memory_logger
from memory_store import MemoryStore
from llm_client import LLMClient


def changes(*operations, forget=False, forget_evidence=None):
    return {'forget': forget, 'forget_evidence': forget_evidence or [], 'operations': list(operations)}


def operation(turn, text, *, action='add', ident='', section='personal', basis='explicit', expires=None):
    return {'action': action, 'section': section, 'id': ident, 'text': text, 'basis': basis,
            'expires_on': expires, 'reason': 'New user information supports this change.',
            'evidence': [{'turn_id': turn['id'], 'quote': turn['user_speech']}]}


def completion(content, finish='stop', refusal=None):
    return httpx.Response(200, json={'id': 'fixture', 'object': 'chat.completion',
        'created': 0, 'model': 'gpt-6-luna', 'choices': [{'index': 0, 'finish_reason': finish,
        'message': {'role': 'assistant', 'content': json.dumps(content), 'refusal': refusal}}],
        'usage': {'prompt_tokens': 500, 'completion_tokens': 250, 'total_tokens': 750,
                  'completion_tokens_details': {'reasoning_tokens': 150}}})


class StoreTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.now = datetime(2026, 10, 2, 23, 45, tzinfo=timezone.utc)
        self.decisions = Mock()
        self.store = self.open_store()

    def open_store(self):
        store = MemoryStore(self.directory, max_history=2, logger=self.decisions, now=lambda: self.now)
        self.addCleanup(store.close)
        return store

    def say(self, speech, reply='Understood.'):
        self.store.record_turn(speech, reply)
        return self.store.next_batch()

    def remember_cat(self):
        batch = self.say('My cat is called Miso.')
        self.store.apply(batch, changes(operation(batch['new_turns'][0], 'Torgo has a cat named Miso.')))
        return self.store.memory['personal'][0]['id']

    def test_pet_survives_restart_old_age_and_many_weather_turns_without_history_growth(self):
        self.remember_cat()
        for day in range(15):
            self.now += timedelta(days=7)
            batch = self.say('What is the weather?', 'The forecast is sunny.')
            self.store.apply(batch, changes())
        self.store.close()
        restored = self.open_store()
        history, facts = restored.recall()
        self.assertIn('cat named Miso', facts)
        self.assertNotIn('forecast', facts)
        self.assertEqual(len(history), 4)
        self.assertEqual(len(restored.recent['turns']), 2)
        self.assertIsNone(restored.next_batch())

    def test_correction_keeps_id_replaces_text_and_old_supporting_quote(self):
        ident = self.remember_cat()
        self.now += timedelta(days=1)
        batch = self.say('Actually, his name is Milo.')
        self.store.apply(batch, changes(operation(batch['new_turns'][0], 'Torgo has a cat named Milo.',
                                                 action='update', ident=ident)))
        entry = self.store.memory['personal'][0]
        self.assertEqual(entry['id'], ident)
        self.assertEqual(entry['basis'], 'explicit')
        self.assertNotIn('Miso', json.dumps(entry))
        self.assertEqual(entry['evidence_dates'], ['2026-10-03'])
        self.assertEqual(entry['evidence_count'], 1)  # Old spelling isn't evidence for the correction.

    def test_reinforcement_is_counted_once_and_dates_distinguish_same_day_questions(self):
        batch = self.say('I enjoy Harry Potter trivia.')
        op = operation(batch['new_turns'][0], 'Enjoys Harry Potter trivia.')
        self.store.apply(batch, changes(op))
        ident = self.store.memory['personal'][0]['id']
        batch = self.say('I still enjoy Harry Potter trivia.')
        op = operation(batch['new_turns'][0], 'Enjoys Harry Potter trivia.', action='reinforce', ident=ident)
        self.store.apply(batch, changes(op))
        with self.assertRaisesRegex(ValueError, 'Stale'):
            self.store.apply(batch, changes(op))
        entry = self.store.memory['personal'][0]
        self.assertEqual(entry['evidence_count'], 2)
        self.assertEqual(entry['evidence_dates'], ['2026-10-02'])

    def test_topic_expires_but_personal_fact_does_not(self):
        self.remember_cat()
        batch = self.say('I am planning a costume for Halloween.')
        op = operation(batch['new_turns'][0], 'Planning a Halloween costume.',
                       section='topics', expires='2026-10-31')
        self.store.apply(batch, changes(op))
        self.assertIn('Halloween', self.store.recall()[1])
        self.now += timedelta(days=31)
        facts = self.store.recall()[1]
        self.assertNotIn('Halloween', facts)
        self.assertIn('Miso', facts)
        self.store.apply(self.say('What time is it?'), changes())
        self.assertEqual(self.store.memory['topics'], [])

    def test_forget_clears_topic_and_recent_context_and_is_not_replayed_after_restart(self):
        ident = self.remember_cat()
        batch = self.say('Let us discuss Miso tomorrow.')
        self.store.apply(batch, changes(operation(batch['new_turns'][0], 'Discussing Miso.',
                                                 section='topics', expires='2026-10-03')))
        batch = self.say('Forget my cat\'s name.')
        turn = batch['new_turns'][0]
        # A later statement queued while reasoning is in flight must survive.
        self.store.record_turn('My new project is building a clock.', 'Understood.')
        self.store.apply(batch, changes(operation(turn, 'Torgo has a cat named Miso.', action='delete', ident=ident),
                                       forget=True, forget_evidence=[{'turn_id': turn['id'], 'quote': turn['user_speech']}]))
        self.store.close()
        self.store = self.open_store()
        history, facts = self.store.recall()
        self.assertEqual(len(history), 2)
        self.assertIn('building a clock', history[0]['content'])
        self.assertNotIn('Miso', facts)
        batch = self.store.next_batch()
        self.assertEqual(batch['earlier_context'], [])
        self.assertNotIn('Miso', json.dumps(batch))
        self.assertIn('building a clock', batch['new_turns'][0]['user_speech'])

    def test_invalid_second_operation_rolls_back_whole_patch_and_cursor(self):
        batch = self.say('My cat is named Miso.')
        path = self.directory / 'memory.json'
        before = path.read_bytes()
        good = operation(batch['new_turns'][0], 'Cat named Miso.')
        bad = operation(batch['new_turns'][0], 'Other fact.', action='update', ident='unknown')
        with self.assertRaises(ValueError):
            self.store.apply(batch, changes(good, bad))
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.store.memory['personal'], [])
        self.assertIsNotNone(self.store.next_batch())

    def test_assistant_claims_and_old_turns_cannot_be_used_as_new_evidence(self):
        batch = self.say('Tell me a story.', 'Your cat is named Dave.')
        op = operation(batch['new_turns'][0], 'Cat named Dave.')
        op['evidence'][0]['quote'] = 'Your cat is named Dave.'
        with self.assertRaisesRegex(ValueError, 'exact user speech'):
            self.store.apply(batch, changes(op))
        op['evidence'][0] = {'turn_id': 999, 'quote': 'Tell me a story.'}
        with self.assertRaisesRegex(ValueError, 'NEW accepted'):
            self.store.apply(batch, changes(op))
        self.assertEqual(self.store.memory['personal'], [])

    def test_pending_work_is_durable_beyond_recent_limit(self):
        for number in range(7):
            self.say(f'My project detail number {number}.')
        self.store.close()
        self.store = self.open_store()
        self.assertEqual(len(self.store.recall()[0]), 4)
        processed = []
        while (batch := self.store.next_batch()) is not None:
            processed.extend(t['id'] for t in batch['new_turns'])
            self.store.apply(batch, changes())
        self.assertEqual(processed, list(range(1, 8)))
        self.assertEqual(len(self.store.recent['turns']), 2)

    def test_crash_between_memory_commit_and_recent_cleanup_does_not_reapply_change(self):
        batch = self.say('My cat is Miso.')
        op = operation(batch['new_turns'][0], 'Cat named Miso.')
        original = self.store._save
        def fail_recent(name, value):
            if name == 'recent_conversation.json':
                raise OSError('simulated disk failure')
            return original(name, value)
        with patch.object(self.store, '_save', side_effect=fail_recent), self.assertRaises(OSError):
            self.store.apply(batch, changes(op))
        self.store.close()
        self.store = self.open_store()
        self.assertIsNone(self.store.next_batch())
        self.assertEqual(self.store.memory['personal'][0]['evidence_count'], 1)

    def test_invalid_json_is_preserved_and_manual_edits_load_after_restart(self):
        self.remember_cat()
        self.store.close()
        path = self.directory / 'memory.json'
        valid = json.loads(path.read_text())
        valid['personal'][0]['text'] = 'Torgo has a cat named Milo.'
        path.write_text(json.dumps(valid))
        self.store = self.open_store()
        self.assertIn('Milo', self.store.recall()[1])
        self.store.close()
        path.write_text('{broken manual edit')
        with self.assertRaises(ValueError):
            self.open_store()
        self.assertEqual(path.read_text(), '{broken manual edit')
        self.assertTrue(path.with_suffix('.json.bak').exists())

    def test_running_manual_edits_and_second_writer_are_not_overwritten(self):
        with self.assertRaises(BlockingIOError):
            self.open_store()
        batch = self.say('My cat is Miso.')
        path = self.directory / 'memory.json'
        edited = path.read_text() + '\n'
        path.write_text(edited)
        with self.assertRaisesRegex(ValueError, 'changed outside HAL'):
            self.store.apply(batch, changes())
        self.assertEqual(path.read_text(), edited)


class APITests(unittest.TestCase):
    def client(self, handler, reasoning='medium'):
        requests = []
        def capture(request):
            requests.append(json.loads(request.content))
            return handler(len(requests))
        api = OpenAI(api_key='fixture-not-a-key', max_retries=0,
                     http_client=httpx.Client(transport=httpx.MockTransport(capture)))
        self.addCleanup(api.close)
        with patch('conversation_memory.OpenAI', return_value=api):
            updater = MemoryUpdater(MemorySettings(reasoning=reasoning), 'fixture-not-a-key', Mock())
        return updater, requests

    def test_background_reasoning_and_token_budget_are_separate_from_foreground(self):
        updater, requests = self.client(lambda _: completion(changes()), reasoning='high')
        result = updater.update({'memory': {}, 'new_turns': []})
        self.assertEqual(result, changes())
        body = requests[0]
        self.assertEqual(body['reasoning_effort'], 'high')
        self.assertEqual(body['max_completion_tokens'], 8192)
        self.assertEqual(body['service_tier'], 'default')
        self.assertTrue(body['response_format']['json_schema']['strict'])
        self.assertNotIn('tools', body)
        self.assertNotIn('temperature', body)
        self.assertNotIn('pod bay', body['messages'][0]['content'])

    def test_truncated_refused_or_malformed_decisions_do_not_become_updates(self):
        for response in (completion(changes(), finish='length'), completion(changes(), refusal='No'),
                         completion(None)):
            with self.subTest(response=response):
                updater, requests = self.client(lambda _: response)
                # null JSON is rejected transactionally by the store; incomplete
                # API output is rejected before it can reach the store.
                if response.json()['choices'][0]['message']['content'] == 'null':
                    self.assertIsNone(updater.update({}))
                else:
                    with self.assertRaises(ValueError):
                        updater.update({})
                self.assertEqual(len(requests), 1)

    def test_recall_uses_one_foreground_call_and_preserves_followup_controls(self):
        updater, requests = self.client(lambda _: completion({'decision': 'respond', 'reply': 'Miso.'}))
        memory = Mock()
        memory.read_context.return_value = ([{'role': 'user', 'content': '[yesterday] My cat is Miso.'},
                                             {'role': 'assistant', 'content': 'Understood.'}],
                                            'Saved memory: cat named Miso; explicit.')
        with patch('llm_client.OpenAI', return_value=updater.client):
            llm = LLMClient('openai', 'gpt-6-luna', openai_api_key='fixture-not-a-key',
                            service_tier='fast', memory=memory)
        llm.begin_turn()
        self.assertEqual(llm.get_followup_response('What is my cat called?').reply, 'Miso.')
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]['reasoning_effort'], 'none')
        self.assertEqual(requests[0]['max_completion_tokens'], 512)
        self.assertEqual(requests[0]['service_tier'], 'priority')
        self.assertIn('FOLLOW-UP CONTROL', requests[0]['messages'][0]['content'])
        self.assertIn('cat named Miso', requests[0]['messages'][1]['content'])
        memory.record_turn.assert_not_called()
        llm.finish_turn('What is my cat called?', 'Miso.')
        self.assertEqual(memory.record_turn.call_args.args[:2], ('What is my cat called?', 'Miso.'))

    def test_memory_options_are_opt_in_and_reasoning_is_validated(self):
        with patch.dict('os.environ', {}, clear=True):
            self.assertFalse(MemorySettings.from_env().enabled)
            self.assertIsNone(ConversationMemory.from_env(Mock(), 10))
        with patch.dict('os.environ', {'MEMORY_ENABLED': 'true', 'MEMORY_MODEL': 'gpt-6-luna',
                                      'MEMORY_REASONING_EFFORT': 'High'}, clear=True):
            self.assertEqual(MemorySettings.from_env().reasoning, 'high')
        for options in ({'reasoning': 'maximum'}, {'max_tokens': 512}, {'timeout': float('nan')},
                        {'soft_tokens': 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                MemorySettings(**options)

    def test_startup_with_bad_json_disables_memory_without_resetting_files_or_calling_api(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'memory.json'
            path.write_text('{invalid edit')
            decisions = Mock()
            with patch.dict('os.environ', {'MEMORY_ENABLED': 'true', 'MEMORY_DIR': directory,
                                          'LOG_PATH': directory, 'LLM_BACKEND': 'openai'}, clear=True), \
                 patch('conversation_memory.memory_logger', return_value=decisions), \
                 patch('conversation_memory.MemoryUpdater') as api:
                self.assertIsNone(ConversationMemory.from_env(Mock(), 10))
                api.assert_not_called()
            self.assertEqual(path.read_text(), '{invalid edit')
            self.assertFalse((Path(directory) / 'recent_conversation.json').exists())


class WorkerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = MemoryStore(directory.name, logger=Mock())
        self.addCleanup(self.store.close)

    def test_blocked_background_api_does_not_block_recall_or_recording_next_exchange(self):
        entered, release, applied = threading.Event(), threading.Event(), threading.Event()
        updater = Mock()
        def update(batch):
            entered.set()
            if not release.wait(3):
                raise RuntimeError('test release was not signalled')
            return changes()
        updater.update.side_effect = update
        original = self.store.apply
        def apply(batch, result):
            count = original(batch, result)
            applied.set()
            return count
        self.store.apply = apply
        memory = ConversationMemory(self.store, updater, MemorySettings(), Mock(), Mock())
        try:
            memory.record_turn('My cat is Miso.', 'Understood.')
            self.assertTrue(entered.wait(2))
            self.assertFalse(release.is_set())
            history, _ = memory.read_context()
            self.assertIn('My cat is Miso.', history[0]['content'])
            memory.record_turn('What time is it?', 'Noon.')
            self.assertEqual(len(memory.read_context()[0]), 4)
            release.set()
            self.assertTrue(applied.wait(2))
        finally:
            release.set()
            memory.close()

    def test_failed_update_preserves_pending_work_and_does_not_immediately_retry(self):
        failed = threading.Event()
        updater = Mock()
        updater.update.side_effect = RuntimeError('raw service detail')
        log = Mock()
        log.warning.side_effect = lambda *a, **kw: failed.set()
        memory = ConversationMemory(self.store, updater, MemorySettings(), log, Mock())
        try:
            memory.record_turn('My cat is Miso.', 'Understood.')
            self.assertTrue(failed.wait(2))
            self.assertEqual(updater.update.call_count, 1)
            self.assertEqual(self.store.memory['personal'], [])
            self.assertIsNotNone(self.store.next_batch())
        finally:
            memory.close()

    def test_shutdown_discards_late_result_and_keeps_pending_turn_for_restart(self):
        entered, release = threading.Event(), threading.Event()
        updater = Mock()
        def update(batch):
            entered.set()
            release.wait(3)
            return changes(operation(batch['new_turns'][0], 'Cat named Miso.'))
        updater.update.side_effect = update
        memory = ConversationMemory(self.store, updater, MemorySettings(), Mock(), Mock())
        try:
            memory.record_turn('My cat is Miso.', 'Understood.')
            self.assertTrue(entered.wait(2))
            memory.close()
            release.set()
            memory.thread.join(timeout=2)
            self.assertFalse(memory.thread.is_alive())
            restored = MemoryStore(self.store.directory, logger=Mock())
            try:
                self.assertEqual(restored.memory['personal'], [])
                self.assertIsNotNone(restored.next_batch())
            finally:
                restored.close()
        finally:
            release.set()
            memory.close()

    def test_empty_transcript_does_not_disable_memory_or_make_a_paid_call(self):
        updater = Mock()
        memory = ConversationMemory(self.store, updater, MemorySettings(), Mock(), Mock())
        try:
            memory.record_turn('', 'Can you repeat that?')
            self.assertTrue(memory.available)
            self.assertEqual(self.store.recent['turns'], [])
            updater.update.assert_not_called()
        finally:
            memory.close()

    def test_dedicated_log_keeps_decisions_separate_from_other_hal_logs(self):
        decisions = memory_logger(self.store.directory)
        for handler in decisions.handlers:
            self.addCleanup(handler.close)
            self.addCleanup(decisions.removeHandler, handler)
        decisions.info('Memory decision: test addition, supporting user quote.')
        for handler in decisions.handlers:
            handler.flush()
        text = (self.store.directory / 'memory.log').read_text()
        self.assertIn('supporting user quote', text)
        self.assertFalse(decisions.propagate)


if __name__ == '__main__':
    unittest.main()
