"""Local tagged recall, continuity and bounded prompts; no models or network."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from memory_retrieval import compact_entry, retrieve


def entry(ident, text, *tags, **extra):
    return {'id': ident, 'text': text, 'tags': list(tags), **extra}


def ids(selected):
    return {item['id'] for values in selected.values() for item in values}


class RetrievalTests(unittest.TestCase):
    def test_broad_tags_recall_user_hal_and_topic_without_requiring_every_word(self):
        memories = {
            'personal': [entry('fan', 'Torgo is a Vikings fan.', 'Minnesota Vikings', 'football', 'sports'),
                         entry('music', 'Torgo writes orchestral music.', 'composition')],
            'hal': [entry('hal-sport', 'HAL enjoys comparing team strategies.', 'football', 'sports')],
            'topics': [entry('playoffs', 'We disagreed about the chances of a postseason run.',
                             'NFL playoffs', 'football')],
        }
        selected, trace = retrieve(memories, 'What do you think about football this season?', [])
        self.assertEqual(ids(selected), {'fan', 'hal-sport', 'playoffs'})
        fan = next(item for item in trace['selected'] if item['id'] == 'fan')
        self.assertEqual(fan['matched']['query']['tags'], ['football'])
        self.assertEqual(fan['matched']['query']['text'], [])
        self.assertEqual(trace['unmatched_count'], 1)
        self.assertEqual(trace['query'], 'What do you think about football this season?')
        json.dumps(trace)  # The actual log payload must be JSON-safe.

    def test_multiword_tags_possessives_case_punctuation_accents_and_plurals(self):
        memories = {'personal': [entry('team', 'A favorite team.', 'Minnesota Vikings'),
                                 entry('cats', 'Owns a cat.', 'pets'),
                                 entry('cafe', 'A favorite place.', 'café')],
                    'topics': [entry('ethics', 'A discussion.', 'moral responsibility')]}
        selected, trace = retrieve(memories, "VIKINGS’ fans, cats, CAFÉ'S and responsibilities?", [])
        self.assertEqual(ids(selected), {'team', 'cats', 'cafe', 'ethics'})
        self.assertIn('viking', trace['query_terms'])
        self.assertIn('responsibility', trace['query_terms'])
        self.assertIn('cafe', trace['query_terms'])

    def test_legacy_records_without_tags_are_found_by_text(self):
        selected, trace = retrieve({'personal': [{'id': 'legacy', 'text': 'The user owns a cat named Miso.'}]},
                                   'How is Miso?', [])
        self.assertEqual(ids(selected), {'legacy'})
        self.assertEqual(trace['selected'][0]['matched']['query']['text'], ['miso'])

    def test_followup_uses_both_user_and_assistant_recent_topic_words(self):
        memories = {'hal': [entry('mind', 'HAL leans toward compatibilism.', 'free will')],
                    'topics': [entry('choices', 'A debate about choices.', 'moral responsibility')]}
        turns = [{'id': 12, 'user_speech': 'Let us talk about free will.',
                  'assistant_reply': 'Moral responsibility seems central to this.'}]
        selected, trace = retrieve(memories, 'Do you still think that?', turns)
        self.assertEqual(ids(selected), {'mind', 'choices'})
        self.assertEqual(trace['query_terms'], [])
        self.assertIn('responsibility', trace['context_terms'])
        self.assertEqual(trace['context_turns'][0]['turn_id'], 12)

    def test_only_last_three_exchanges_supply_context_and_recent_words_rank_higher(self):
        memories = {'topics': [entry('old', 'Old subject.', 'obsolete'),
                               entry('older', 'Earlier subject.', 'astronomy'),
                               entry('newer', 'Current subject.', 'gardening')]}
        turns = [{'id': 1, 'user_speech': 'Obsolete', 'assistant_reply': 'Indeed.'},
                 {'id': 2, 'user_speech': 'Astronomy', 'assistant_reply': ''},
                 {'id': 3, 'user_speech': '', 'assistant_reply': ''},
                 {'id': 4, 'user_speech': 'Gardening', 'assistant_reply': ''}]
        selected, trace = retrieve(memories, '', turns)
        self.assertEqual(ids(selected), {'older', 'newer'})
        self.assertEqual([item['id'] for item in trace['selected']], ['newer', 'older'])

    def test_current_query_beats_verbose_context_only_match(self):
        words = 'astronomy garden economics literature architecture music philosophy physics biology'
        memories = {'personal': [entry('current', 'Likes football.')],
                    'topics': [entry('prior', words, words)]}
        turns = [{'user_speech': words, 'assistant_reply': words}]
        _, trace = retrieve(memories, 'Football?', turns)
        self.assertEqual(trace['selected'][0]['id'], 'current')

    def test_unrelated_records_and_empty_search_do_not_fill_budget(self):
        memories = {'personal': [entry('cat', 'Torgo owns a cat.')],
                    'hal': [entry('view', 'HAL favors carefully reasoned arguments.')]}
        for query in ('', 'Hello HAL, what do you think?', 'Volcanoes?'):
            with self.subTest(query=query):
                selected, trace = retrieve(memories, query, [])
                self.assertEqual(ids(selected), set())
                self.assertEqual(trace['unmatched_count'], 2)

    def test_budget_skips_oversized_record_but_still_selects_smaller_match(self):
        memories = {'personal': [entry('large', 'Football ' * 100, 'football'),
                                 entry('small', 'Football fan.')]}
        before = copy.deepcopy(memories)
        selected, trace = retrieve(memories, 'Football', [], token_budget=60)
        self.assertEqual(ids(selected), {'small'})
        self.assertLessEqual(trace['serialized_characters'], 240)
        self.assertEqual(trace['omitted_count'], 1)
        self.assertEqual(trace['omitted'][0]['id'], 'large')
        self.assertEqual(trace['omitted'][0]['reason'], 'token_budget')
        self.assertEqual(trace['omitted'][0]['text'], memories['personal'][0]['text'])
        self.assertEqual(memories, before)
        selected['personal'][0]['text'] = 'Changed by a caller.'
        self.assertEqual(memories, before)

    def test_budget_and_tie_order_are_deterministic_without_a_fixed_record_limit(self):
        memories = {'personal': [entry(f'fan-{number:02}', f'Football fact {number}.') for number in range(20)]}
        first, trace = retrieve(memories, 'football', [], token_budget=1000)
        second, trace2 = retrieve({'personal': list(reversed(memories['personal']))},
                                  'football', [], token_budget=1000)
        self.assertEqual(first, second)
        self.assertEqual(trace, trace2)
        self.assertEqual(len(ids(first)), 20)
        smaller, limited = retrieve(memories, 'football', [], token_budget=100)
        self.assertLess(len(ids(smaller)), 20)
        self.assertGreater(limited['omitted_count'], 0)
        compact = {section: [compact_entry(item) for item in items] for section, items in smaller.items()}
        self.assertEqual(len(json.dumps(compact, ensure_ascii=False)), limited['serialized_characters'])
        self.assertLessEqual(limited['serialized_characters'], 400)

    def test_compact_entry_keeps_attribution_dates_retention_and_ids_without_evidence(self):
        memory = entry('hal-1', 'HAL tentatively favors this argument.', 'philosophy',
                       basis='explicit', updated_at='2026-10-06T00:00:00+00:00', expires_on=None,
                       retention='durable', evidence=[{'quote': 'Raw transcript.'}], evidence_count=1)
        compact = compact_entry(memory)
        self.assertEqual(set(compact), {'id', 'text', 'tags', 'basis', 'updated_at', 'expires_on', 'retention'})
        compact['tags'].append('modified')
        self.assertEqual(memory['tags'], ['philosophy'])

    def test_explicit_general_recall_browses_requested_sections_without_recent_context(self):
        memories = {'personal': [entry('fan', 'Torgo follows the Vikings.', 'football')],
                    'hal': [entry('mind', 'HAL leans toward compatibilism.', 'philosophy')],
                    'topics': [entry('debate', 'We disagreed about determinism.', 'free will')]}
        cases = [('What do you remember about me?', ['personal'], {'fan'}),
                 ('Hey HAL, what do you know about me?', ['personal'], {'fan'}),
                 ('Tell me about yourself.', ['hal'], {'mind'}),
                 ('What are your interests?', ['hal'], {'mind'}),
                 ('What are your views?', ['hal'], {'mind'}),
                 ('What are your opinions?', ['hal'], {'mind'}),
                 ('What do you remember?', ['personal', 'hal', 'topics'], {'fan', 'mind', 'debate'})]
        for query, sections, expected in cases:
            with self.subTest(query=query):
                selected, trace = retrieve(memories, query, [])
                self.assertEqual(ids(selected), expected)
                self.assertEqual(trace['browse_sections'], sections)
                self.assertEqual(trace['matched_count'], 0)
                self.assertEqual(trace['browse_only_count'], len(expected))
                for result in trace['selected']:
                    self.assertEqual(result['reason'], 'browse')
                    self.assertEqual(result['score'], 0)
                    self.assertEqual(result['matched'], {'query': {'text': [], 'tags': []},
                                                         'context': {'text': [], 'tags': []}})

    def test_browse_has_same_budget_logs_omissions_and_prefers_newer_records(self):
        memories = {'personal': [entry('older', 'A favorite animal.', updated_at='2025-01-01T00:00:00+00:00'),
                                 entry('newer', 'A favorite album.', updated_at='2026-01-01T00:00:00+00:00')]}
        selected, trace = retrieve(memories, 'What do you remember about me?', [], token_budget=50)
        self.assertEqual(ids(selected), {'newer'})
        self.assertLessEqual(trace['serialized_characters'], 200)
        self.assertEqual(trace['omitted'][0]['id'], 'older')
        self.assertEqual(trace['omitted'][0]['reason'], 'token_budget')
        self.assertEqual(trace['omitted'][0]['retrieval_reason'], 'browse')
        reversed_result = retrieve({'personal': list(reversed(memories['personal']))},
                                   'What do you remember about me?', [], token_budget=50)
        self.assertEqual((selected, trace), reversed_result)

    def test_query_and_recent_context_matches_precede_browse_candidates(self):
        memories = {'hal': [entry('browse', 'Enjoys astronomy.', updated_at='2026-10-05T00:00:00+00:00'),
                            entry('context', 'Enjoys philosophy.', updated_at='2025-01-01T00:00:00+00:00'),
                            entry('query', 'Views regarding animals.', updated_at='2024-01-01T00:00:00+00:00')]}
        _, trace = retrieve(memories, 'What are your views?',
                            [{'user_speech': 'Philosophy?', 'assistant_reply': 'Certainly.'}])
        self.assertEqual([item['id'] for item in trace['selected']], ['query', 'context', 'browse'])
        self.assertEqual([item['reason'] for item in trace['selected']], ['query', 'context', 'browse'])

    def test_specific_queries_greetings_and_empty_queries_never_trigger_browse(self):
        memories = {'personal': [entry('fan', 'A Vikings fan.', 'football'),
                                 entry('music', 'Writes music.', 'composition')],
                    'hal': [entry('view', 'Favors a particular philosophical position.', 'philosophy')]}
        cases = [('What do you remember about the Vikings game?', {'fan'}),
                 ('What are your views on football?', {'fan'}),
                 ('What is football?', {'fan'}),
                 ('Remember the Vikings game.', {'fan'}),
                 ('', set()), ('Hello HAL.', set())]
        for query, expected in cases:
            with self.subTest(query=query):
                selected, trace = retrieve(memories, query, [])
                self.assertEqual(ids(selected), expected)
                self.assertEqual(trace['browse_sections'], [])
                self.assertEqual(trace['browse_only_count'], 0)


if __name__ == '__main__':
    unittest.main()
