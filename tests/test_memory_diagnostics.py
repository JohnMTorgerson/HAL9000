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


def test_wrong_role_and_unavailable_turn_have_distinct_diagnostics():
    turn = {'id': 1, 'at': '2026-10-07T12:00:00+00:00',
            'user_speech': 'Hello.', 'assistant_reply': 'Understood.'}
    for evidence, expected in [(source(turn, 'assistant'), 'source_role'),
                               (dict(source(turn), turn_id=99), 'source_not_supplied')]:
        with pytest.raises(EvidenceValidationError) as err:
            MemoryStore._evidence([evidence], {1: turn})
        assert err.value.details['check'] == expected
        assert err.value.details['source'] == evidence
