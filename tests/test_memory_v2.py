"""Durable attributed memories, migration and maintenance without paid calls."""
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from memory_store import MemoryStore


def change(*operations, tags=(), forget=False, forget_ids=(), forget_evidence=()):
    return {'forget': forget, 'forget_ids': list(forget_ids),
            'forget_evidence': list(forget_evidence), 'tag_updates': list(tags),
            'operations': list(operations)}


def source(turn, role='user', quote=None):
    speech = turn['user_speech'] if role == 'user' else turn['assistant_reply']
    return {'turn_id': turn['id'], 'role': role, 'quote': quote or speech}


def operation(turn, text, *, section='personal', action='add', ident='',
              retention='durable', expiry=None, tags=('memory',), evidence=None):
    role = 'assistant' if section == 'hal' else 'user'
    return {'action': action, 'section': section, 'id': ident, 'text': text,
            'basis': 'explicit', 'retention': retention, 'expires_on': expiry,
            'tags': list(tags), 'reason': 'The completed exchange supports this memory.',
            'evidence': evidence or [source(turn, role)]}


class MemoryV2Tests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.now = datetime(2026, 10, 6, 10, tzinfo=timezone.utc)
        self.logger = Mock()
        self.store = self.open_store()

    def open_store(self):
        store = MemoryStore(self.directory, max_history=2, logger=self.logger, now=lambda: self.now)
        self.addCleanup(store.close)
        return store

    def say(self, speech, reply='Understood.'):
        self.store.record_turn(speech, reply)
        return self.store.next_batch()

    def legacy_entry(self, ident='m_fan', *, text='Torgo is a Vikings fan.', expiry=None):
        return {'id': ident, 'text': text, 'basis': 'explicit',
                'created_at': '2026-09-01T10:00:00+00:00',
                'updated_at': '2026-09-02T10:00:00+00:00', 'expires_on': expiry,
                'evidence_count': 4, 'evidence_dates': ['2026-09-01', '2026-09-02'],
                'evidence': [{'turn_id': 7, 'at': '2026-09-02T10:00:00+00:00',
                              'quote': text}]}

    def install_legacy(self, personal=None, topics=None, turns=()):
        self.store.close()
        legacy = {'version': 1, 'last_processed_turn': 7, 'context_after_turn': 2,
                  'personal': personal if personal is not None else [self.legacy_entry()],
                  'topics': topics or []}
        recent = {'version': 1, 'next_turn_id': max([7] + [t['id'] for t in turns]) + 1,
                  'turns': list(turns)}
        (self.directory / 'memory.json').write_text(json.dumps(legacy))
        (self.directory / 'recent_conversation.json').write_text(json.dumps(recent))
        self.store = self.open_store()
        return legacy, recent

    @staticmethod
    def index_changes(batch):
        return change(tags=[{'section': item['section'], 'id': item['id'],
                             'tags': ['vikings', 'football', 'sports']}
                            for item in batch['tagging_entries']])

    def test_v1_migration_preserves_metadata_and_pending_turns(self):
        topic = self.legacy_entry('m_project', text='Discussing a clock project.', expiry='2026-10-30')
        pending = {'id': 8, 'at': self.now.isoformat(), 'user_speech': 'I also enjoy theater.',
                   'assistant_reply': 'There is a great deal to explore there.'}
        legacy, recent = self.install_legacy(topics=[topic], turns=[pending])
        self.assertEqual(self.store.memory['version'], 2)
        self.assertEqual(self.store.memory['hal'], [])
        self.assertEqual(self.store.memory['last_processed_turn'], 7)
        self.assertEqual(self.store.memory['context_after_turn'], 2)
        for section, retention in (('personal', 'durable'), ('topics', 'temporary')):
            migrated = copy.deepcopy(self.store.memory[section][0])
            self.assertEqual(migrated.pop('tags'), [])
            self.assertEqual(migrated.pop('retention'), retention)
            self.assertEqual(migrated['evidence'][0].pop('role'), 'user')
            self.assertEqual(migrated, legacy[section][0])
        self.assertEqual(self.store.recent, recent)
        self.assertEqual(self.store.next_batch()['new_turns'], [pending])
        backup = json.loads((self.directory / 'memory.json.bak').read_text())
        self.assertEqual(backup, legacy)

    def test_invalid_legacy_file_is_not_rewritten_by_migration(self):
        self.store.close()
        invalid = {'version': 1, 'last_processed_turn': 7, 'context_after_turn': 2,
                   'personal': [self.legacy_entry(expiry='2026-10-30')], 'topics': []}
        path = self.directory / 'memory.json'
        original = json.dumps(invalid)
        path.write_text(original)
        with self.assertRaisesRegex(ValueError, 'cannot expire'):
            self.open_store()
        self.assertEqual(path.read_text(), original)

    def test_startup_backfill_changes_only_tags_and_runs_once_across_restart(self):
        self.install_legacy()
        before = copy.deepcopy(self.store.memory)
        batch = self.store.next_batch()
        self.assertEqual(batch['new_turns'], [])
        self.assertEqual(len(batch['tagging_entries']), 1)
        self.store.apply(batch, self.index_changes(batch))
        indexed = copy.deepcopy(self.store.memory)
        self.assertEqual(indexed['personal'][0].pop('tags'), ['vikings', 'football', 'sports'])
        before['personal'][0].pop('tags')
        self.assertEqual(indexed, before)
        self.store.close()
        self.store = self.open_store()
        self.assertIsNone(self.store.next_batch())
        self.assertIn('Vikings fan', self.store.recall('football')[1])

    def test_backfill_rejects_stale_patch_even_without_cursor_advance(self):
        self.install_legacy()
        batch = self.store.next_batch()
        changes = self.index_changes(batch)
        self.store.apply(batch, changes)
        with self.assertRaisesRegex(ValueError, 'Stale tag-only'):
            self.store.apply(batch, changes)
        self.assertEqual(self.store.memory['last_processed_turn'], 7)

    def test_backfill_rejects_no_progress_and_never_advances_pending_turn(self):
        pending = {'id': 8, 'at': self.now.isoformat(), 'user_speech': 'What time is it?',
                   'assistant_reply': 'Ten oclock.'}
        self.install_legacy(turns=[pending])
        before = (self.directory / 'memory.json').read_bytes()
        batch = self.store.next_batch()
        with self.assertRaisesRegex(ValueError, 'Every pending indexing entry'):
            self.store.apply(batch, change())
        self.assertEqual((self.directory / 'memory.json').read_bytes(), before)
        self.assertEqual(self.store.next_batch()['new_turns'], [pending])
        self.store.apply(batch, self.index_changes(batch))
        self.assertEqual(self.store.memory['last_processed_turn'], 8)
        self.assertIsNone(self.store.next_batch())

    def test_backfill_only_batches_are_bounded_and_cannot_create_new_memories(self):
        self.install_legacy(personal=[self.legacy_entry(f'm_{i}', text=f'Preference number {i}.')
                                      for i in range(23)])
        batch = self.store.next_batch()
        self.assertEqual(len(batch['tagging_entries']), 20)
        fabricated = {'id': 8, 'user_speech': 'Invented.', 'assistant_reply': 'Invented.'}
        with self.assertRaisesRegex(ValueError, 'Tag-only batches'):
            self.store.apply(batch, change(operation(fabricated, 'Invented fact.')))
        self.store.apply(batch, self.index_changes(batch))
        final = self.store.next_batch()
        self.assertEqual(len(final['tagging_entries']), 3)
        self.store.apply(final, self.index_changes(final))
        self.assertIsNone(self.store.next_batch())

    def test_tag_commit_survives_crash_before_recent_cleanup_without_retagging(self):
        self.install_legacy()
        batch = self.store.next_batch()
        original_save = self.store._save
        def fail_recent(name, value):
            if name == 'recent_conversation.json':
                raise OSError('simulated disk failure')
            return original_save(name, value)
        with patch.object(self.store, '_save', side_effect=fail_recent), self.assertRaises(OSError):
            self.store.apply(batch, self.index_changes(batch))
        self.store.close()
        self.store = self.open_store()
        self.assertIsNone(self.store.next_batch())
        self.assertEqual(self.store.memory['personal'][0]['evidence_count'], 4)

    def test_topic_can_attribute_both_speakers_in_same_turn(self):
        batch = self.say('I think determinism is compatible with choice.',
                         'I am less certain; deliberation may still matter.')
        turn = batch['new_turns'][0]
        self.store.apply(batch, change(operation(turn,
            'Torgo favors compatibilism. HAL remains tentative and emphasizes deliberation.',
            section='topics', tags=['free will', 'philosophy', 'choice'],
            evidence=[source(turn), source(turn, 'assistant')])))
        note = self.store.memory['topics'][0]
        self.assertEqual([e['role'] for e in note['evidence']], ['user', 'assistant'])
        self.assertEqual(note['evidence_count'], 2)
        self.assertEqual(len(note['evidence_dates']), 1)

    def test_role_cross_contamination_is_rejected_transactionally(self):
        batch = self.say('You probably like football.', 'I think Torgo likes baseball.')
        turn = batch['new_turns'][0]
        for section, role, text in (('personal', 'assistant', 'Torgo likes baseball.'),
                                    ('hal', 'user', 'HAL likes football.')):
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, 'Evidence role'):
                self.store.apply(batch, change(operation(turn, text, section=section,
                                                         evidence=[source(turn, role)])))
        self.assertEqual(self.store.memory['last_processed_turn'], 0)
        self.assertFalse(self.store.memory['personal'] or self.store.memory['hal'])

    def test_application_action_metadata_cannot_be_hal_evidence(self):
        batch = self.say('Show me an image.', 'Here it is.\n[Application action result: image loaded.]')
        turn = batch['new_turns'][0]
        with self.assertRaisesRegex(ValueError, 'exact user speech or HAL speech'):
            self.store.apply(batch, change(operation(turn, 'HAL likes loaded images.', section='hal',
                evidence=[source(turn, 'assistant', 'image loaded.')])) )

    def test_hal_view_retains_tentative_text_reasons_and_survives_restart(self):
        reply = 'I tentatively favor compatibilism because deliberation can still shape choices.'
        batch = self.say('What is your view of free will?', reply)
        self.store.apply(batch, change(operation(batch['new_turns'][0], reply, section='hal',
                                                tags=['free will', 'compatibilism', 'philosophy'])))
        original = copy.deepcopy(self.store.memory['hal'][0])
        self.store.close()
        self.store = self.open_store()
        self.assertEqual(self.store.memory['hal'][0], original)
        self.assertIn(reply, self.store.recall('philosophy')[1])

    def test_durable_topic_survives_age_while_temporary_topic_expires(self):
        batch = self.say('We can discuss free will today and plan a trip tomorrow.')
        turn = batch['new_turns'][0]
        self.store.apply(batch, change(
            operation(turn, 'An unresolved debate about free will.', section='topics', tags=['philosophy']),
            operation(turn, 'Planning a short trip.', section='topics', retention='temporary',
                      expiry='2026-10-07', tags=['travel'])))
        self.now += timedelta(days=400)
        facts = self.store.recall('philosophy travel')[1]
        self.assertIn('free will', facts)
        self.assertNotIn('short trip', facts)
        self.store.apply(self.say('What time is it?'), change())
        self.assertEqual(len(self.store.memory['topics']), 1)

    def test_temporary_topic_can_be_promoted_but_durable_note_cannot_be_downgraded(self):
        batch = self.say('I want to discuss free will.')
        self.store.apply(batch, change(operation(batch['new_turns'][0], 'Exploring free will.',
            section='topics', retention='temporary', expiry='2026-11-05', tags=['philosophy'])))
        ident = self.store.memory['topics'][0]['id']
        batch = self.say('That debate clarified my lasting position on free will.')
        self.store.apply(batch, change(operation(batch['new_turns'][0], 'A lasting debate on free will.',
            section='topics', action='update', ident=ident, tags=['philosophy'])))
        batch = self.say('Let us revisit free will next week.')
        with self.assertRaisesRegex(ValueError, 'cannot be downgraded'):
            self.store.apply(batch, change(operation(batch['new_turns'][0], 'Revisit free will.',
                section='topics', action='update', ident=ident, retention='temporary',
                expiry='2026-11-05', tags=['philosophy'])))
        self.assertEqual(self.store.memory['topics'][0]['retention'], 'durable')

    def test_targeted_forgetting_covers_all_sections_preserves_unrelated_discussion(self):
        batch = self.say('My cat is Miso. I also believe free will requires deliberation.',
                         'I favor the name Miso. I am interested in the role of deliberation.')
        turn = batch['new_turns'][0]
        self.store.apply(batch, change(
            operation(turn, 'Torgo has a cat named Miso.', tags=['cat', 'miso']),
            operation(turn, 'HAL favors the name Miso.', section='hal', tags=['cat', 'miso']),
            operation(turn, 'Torgo and HAL discussed the name Miso.', section='topics', tags=['miso']),
            operation(turn, 'Torgo believes free will requires deliberation.', section='topics',
                      tags=['free will', 'philosophy'])))
        affected = [entry['id'] for entries in self.store.memory.values() if isinstance(entries, list)
                    for entry in entries if 'Miso' in entry['text']]
        unrelated = copy.deepcopy(self.store.memory['topics'][1])
        batch = self.say('Forget anything about my cat Miso.')
        self.store.record_turn('My new project is a clock.', 'Understood.')
        self.store.apply(batch, change(forget=True, forget_ids=affected,
                                       forget_evidence=[source(batch['new_turns'][0])]))
        self.assertEqual(self.store.memory['personal'], [])
        self.assertEqual(self.store.memory['hal'], [])
        self.assertEqual(self.store.memory['topics'], [unrelated])
        self.store.close()
        self.store = self.open_store()
        history, facts = self.store.recall('cat philosophy')
        self.assertEqual(len(history), 2)
        self.assertIn('new project is a clock', history[0]['content'])
        self.assertNotIn('Miso', facts)
        next_batch = self.store.next_batch()
        self.assertEqual(next_batch['earlier_context'], [])
        self.assertNotIn('Miso', json.dumps(next_batch))
        self.assertEqual(next_batch['new_turns'][0]['id'], 3)

    def test_forget_everything_has_no_twenty_record_limit(self):
        self.install_legacy(personal=[self.legacy_entry(f'm_{i}', text=f'Old preference {i}.')
                                      for i in range(25)])
        batch = self.say('Forget everything you remember about me.')
        ids = [item['id'] for items in batch['catalogue'].values() for item in items]
        self.store.apply(batch, change(forget=True, forget_ids=ids,
                                       forget_evidence=[source(batch['new_turns'][0])]))
        self.assertEqual(self.store.memory['personal'], [])
        self.assertIsNone(self.store.next_batch())

    def test_catalogue_keeps_outside_retrieval_records_visible_for_maintenance(self):
        self.install_legacy(personal=[self.legacy_entry('m_fan'),
                                      self.legacy_entry('m_cat', text='The cat is called Miso.')])
        indexing = self.store.next_batch()
        self.store.apply(indexing, change(tags=[
            {'section': 'personal', 'id': 'm_fan', 'tags': ['football', 'sports']},
            {'section': 'personal', 'id': 'm_cat', 'tags': ['cat', 'pet', 'miso']}]))
        batch = self.say('What do you think about football?')
        self.assertEqual([i['id'] for i in batch['memory']['personal']], ['m_fan'])
        self.assertEqual({i['id'] for i in batch['catalogue']['personal']}, {'m_fan', 'm_cat'})
        self.assertNotIn('evidence', batch['catalogue']['personal'][0])
        self.assertIn('evidence', batch['memory']['personal'][0])
        self.store.apply(batch, change(operation(batch['new_turns'][0], 'The cat is called Miso.',
            action='delete', ident='m_cat', tags=['cat', 'pet', 'miso'])))
        self.assertEqual([item['id'] for item in self.store.memory['personal']], ['m_fan'])

    def test_retrieval_log_records_query_selected_text_tags_and_match_reason(self):
        self.install_legacy()
        indexing = self.store.next_batch()
        self.store.apply(indexing, self.index_changes(indexing))
        self.logger.reset_mock()
        self.store.recall('What do you think about football?')
        call = self.logger.info.call_args
        self.assertEqual(call.args[0], 'Memory retrieval: %s')
        details = json.loads(call.args[1])
        self.assertEqual(details['purpose'], 'foreground')
        self.assertEqual(details['query'], 'What do you think about football?')
        self.assertIn('football', details['query_terms'])
        self.assertEqual(details['selected_count'], 1)
        selected = details['selected'][0]
        self.assertEqual(selected['text'], 'Torgo is a Vikings fan.')
        self.assertIn('football', selected['tags'])
        self.assertIn('football', selected['matched']['query']['tags'])


if __name__ == '__main__':
    unittest.main()
