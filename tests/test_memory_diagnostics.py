"""Rejected evidence is diagnosable without weakening validation or losing work."""
import json
import logging
import threading
from unittest.mock import Mock

import pytest

from conversation_memory import ConversationMemory, MemorySettings
from memory_store import EvidenceValidationError, MemoryStore
from test_memory_v2 import change, operation, source


@pytest.mark.parametrize('reply,quote,check', [
    ('A' * 301, 'A' * 301, 'quote_length'),
    ('A literal reply.', 'A paraphrased reply.', 'quote_mismatch'),
    ('A literal reply.', '', 'quote_empty'),
    ('A literal reply.', None, 'quote_type'),
])
def test_rejected_proposal_logs_exact_check_sources_and_preserves_pending_turn(
        tmp_path, caplog, reply, quote, check):
    decisions = logging.getLogger('memory-diagnostics-test')
    store = MemoryStore(tmp_path, logger=decisions)
    memory = None
    try:
        store.record_turn('What do you think?', reply)
        batch = store.next_batch()
        turn = batch['new_turns'][0]
        first = operation(turn, 'A discussion took place.', section='topics')
        bad_source = dict(source(turn, 'assistant'), quote=quote)
        second = operation(turn, 'A further discussion summary.', section='topics', evidence=[bad_source])
        proposal = change(first, second)
        before = {name: (tmp_path / name).read_bytes() for name in ('memory.json', 'recent_conversation.json')}
        updater, main = Mock(), Mock()
        updater.update.return_value = proposal
        completed = threading.Event()
        main.warning.side_effect = lambda *a, **kw: completed.set()
        with caplog.at_level('INFO', logger=decisions.name):
            memory = ConversationMemory(store, updater, MemorySettings(), main, decisions)
            assert completed.wait(2), 'Background rejection was not reported.'
        record = next(r for r in caplog.records if r.getMessage().startswith('Rejected memory patch details:'))
        details = json.loads(record.args[0])
        assert details['previous_cursor'] == 0 and details['new_turn_ids'] == [turn['id']]
        assert details['proposed_changes'] == proposal
        validation = details['validation']
        assert validation['check'] == check
        assert validation['operation'] == {'index': 2, 'action': 'add', 'section': 'topics', 'id': ''}
        assert validation['source_index'] == 1
        assert validation['source'] == bad_source
        assert validation['available_source_texts'] == [reply]
        assert validation['quote_length'] == (len(quote) if isinstance(quote, str) else None)
        assert validation['max_quote_length'] == 300
        if check == 'quote_length':
            assert '301 characters; maximum is 300' in details['reason']
        assert all((tmp_path / name).read_bytes() == data for name, data in before.items())
        assert store.next_batch()['new_turns'][0]['id'] == turn['id']
        updater.update.assert_called_once()
        main.warning.assert_called_once_with(
            'Background memory patch rejected; see memory.log. Pending exchanges retained.')
    finally:
        if memory is not None:
            memory.close()
        store.close()


def test_exact_300_character_evidence_is_still_accepted():
    speech = 'A' * 300
    turn = {'id': 1, 'at': '2026-10-07T12:00:00+00:00',
            'user_speech': speech, 'assistant_reply': 'Understood.'}
    result = MemoryStore._evidence([source(turn)], {1: turn})
    assert result[0]['quote'] == speech


def test_overlong_reply_can_be_saved_with_exact_excerpt_after_restart(tmp_path):
    # The 379-character answer from the reported F-106 rejection. The summary
    # can retain the full discussion even though its evidence uses an excerpt.
    reply = (
        'NASA used the F-106s for flight research and experiments, not as operational fighters. '
        'The two-seat F-106B tested supersonic engines and fighter maneuverability, and one '
        'aircraft was modified to study lightning strikes. Later, six converted QF-106 drones '
        'took part in the Eclipse project, testing whether a transport aircraft could tow and '
        'launch a reusable space-launch vehicle.'
    )
    store = MemoryStore(tmp_path, logger=Mock())
    try:
        store.record_turn('What does NASA use them for?', reply)
        batch = store.next_batch()
        turn = batch['new_turns'][0]
        proposal = change(operation(turn, reply, section='topics',
                                    evidence=[source(turn, 'assistant')]))
        with pytest.raises(EvidenceValidationError) as err:
            store.apply(batch, proposal)
        assert err.value.details['quote_length'] == 379
        assert store.memory['topics'] == []
        assert store.memory['last_processed_turn'] == 0
    finally:
        store.close()

    store = MemoryStore(tmp_path, logger=Mock())
    try:
        batch = store.next_batch()
        assert batch['new_turns'][0] == turn
        excerpt = reply.split('. ', 1)[0] + '.'
        proposal['operations'][0]['evidence'][0]['quote'] = excerpt
        store.apply(batch, proposal)
        saved = store.memory['topics'][0]
        assert saved['text'] == reply
        assert saved['evidence'][0]['quote'] == excerpt
        assert store.memory['last_processed_turn'] == turn['id']
        assert store.next_batch() is None
    finally:
        store.close()


def test_wrong_role_and_unavailable_turn_have_distinct_diagnostics():
    turn = {'id': 1, 'at': '2026-10-07T12:00:00+00:00',
            'user_speech': 'Hello.', 'assistant_reply': 'Understood.'}
    for evidence, expected in [(source(turn, 'assistant'), 'source_role'),
                               (dict(source(turn), turn_id=99), 'source_not_supplied')]:
        with pytest.raises(EvidenceValidationError) as err:
            MemoryStore._evidence([evidence], {1: turn})
        assert err.value.details['check'] == expected
        assert err.value.details['source'] == evidence


def test_source_limit_counts_old_context_and_both_new_speakers_together():
    turns = [dict(id=i, at='2026-10-07T12:00:00+00:00',
                  user_speech=f'Question {i}.', assistant_reply=f'Answer {i}.')
             for i in (60, 61, 62)]
    # Match the reported shape: the original request, both sides of its retry,
    # and both sides of the pending follow-up are five sources, not three turns.
    evidence = [source(turns[0]), source(turns[1]), source(turns[1], 'assistant'),
                source(turns[2]), source(turns[2], 'assistant')]
    with pytest.raises(EvidenceValidationError) as err:
        MemoryStore._evidence(evidence, {62: turns[2]},
                              roles=('user', 'assistant'), earlier=turns[:2])
    assert err.value.details['check'] == 'source_count'
    assert err.value.details['source_count'] == 5
    assert err.value.details['max_source_count'] == 3
    assert 'received 5' in str(err.value)
    assert err.value.details['sources'] == evidence
    # A model-selected set of three keeps the original referent and new exchange.
    selected = [evidence[0], evidence[3], evidence[4]]
    accepted = MemoryStore._evidence(selected, {62: turns[2]},
                                     roles=('user', 'assistant'), earlier=turns[:2])
    assert [s['quote'] for s in accepted] == [s['quote'] for s in selected]
    assert [s.get('context_only', False) for s in accepted] == [True, False, False]
