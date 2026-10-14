"""Memory proposal repair, source counting, and pending work without paid calls."""
import copy
import json
import threading
from unittest.mock import Mock

import pytest

from conversation_memory import ConversationMemory, MemorySettings
from memory_store import EvidenceValidationError, MemoryStore
from memory_validation import MemoryValidationError
import test_memory as memory_tests
from test_memory_v2 import change, operation, source


@pytest.fixture
def store(tmp_path):
    result = MemoryStore(tmp_path, logger=Mock())
    yield result
    result.close()


def pending(store):
    store.record_turn('I sing in a choir. I rehearse on Tuesdays.', 'I enjoy choral harmony.')
    return store.next_batch()


def excerpts(turn):
    return [source(turn, quote='I sing in a choir.'), source(turn, quote='I rehearse on Tuesdays.')]


def worker(store, updater):
    # Exercise the real attempt logic synchronously; threaded queue behavior is
    # covered separately below and in the integration/restart tests.
    result = ConversationMemory.__new__(ConversationMemory)
    result.store, result.updater = store, updater
    result.logger, result.decisions = Mock(), Mock()
    result.stopping, result.wake = threading.Event(), threading.Event()
    result.available = True
    return result


def test_multiple_excerpts_count_one_source_per_memory_and_survive_restart(store):
    batch = pending(store)
    turn = batch['new_turns'][0]
    quotes = excerpts(turn)
    store.apply(batch, change(
        operation(turn, 'Sings in a choir and rehearses Tuesdays.', evidence=quotes + [quotes[0]]),
        operation(turn, 'Discussed choir rehearsals; HAL enjoys choral harmony.', section='topics',
                  evidence=quotes + [source(turn, 'assistant')]),
    ))
    assert store.memory['personal'][0]['evidence_count'] == 1
    assert len(store.memory['personal'][0]['evidence']) == 2  # Exact repeated quote removed.
    assert store.memory['topics'][0]['evidence_count'] == 2  # User and HAL are separate sources.
    assert len(store.memory['topics'][0]['evidence']) == 3
    expected = copy.deepcopy(store.memory)
    store.close()
    reopened = MemoryStore(store.directory, logger=Mock())
    try:
        assert reopened.memory == expected
    finally:
        reopened.close()


def test_context_and_multiple_new_excerpts_do_not_inflate_reinforcement_or_updates(store):
    first = pending(store)
    original = first['new_turns'][0]
    store.apply(first, change(operation(original, 'Sings in a choir.', evidence=excerpts(original))))
    ident = store.memory['personal'][0]['id']
    store.record_turn('I still sing. Rehearsals are still Tuesdays.', 'Understood.')
    batch = store.next_batch()
    turn = batch['new_turns'][0]
    store.apply(batch, change(operation(turn, 'Sings in a choir.', action='reinforce', ident=ident,
        evidence=[source(original, quote='I sing in a choir.'),
                  source(turn, quote='I still sing.'), source(turn, quote='Rehearsals are still Tuesdays.')])) )
    saved = store.memory['personal'][0]
    assert saved['evidence_count'] == 2
    assert saved['evidence'][0]['context_only']
    store.record_turn('I left the choir. I no longer rehearse.', 'Understood.')
    batch = store.next_batch()
    turn = batch['new_turns'][0]
    store.apply(batch, change(operation(turn, 'No longer sings in a choir.', action='update', ident=ident,
        evidence=[source(turn, quote='I left the choir.'), source(turn, quote='I no longer rehearse.')])) )
    assert store.memory['personal'][0]['evidence_count'] == 1


@pytest.mark.parametrize('role,quote,check', [
    ('user', 'An invented second quotation.', 'quote_mismatch'),
    ('assistant', 'I enjoy choral harmony.', 'source_role'),
    ('user', 'A' * 301, 'quote_length'),
])
def test_second_excerpt_is_still_independently_validated(store, role, quote, check):
    batch = pending(store)
    turn = batch['new_turns'][0]
    proposal = change(operation(turn, 'Sings in a choir.', evidence=[excerpts(turn)[0], source(turn, role, quote)]))
    with pytest.raises(EvidenceValidationError) as exc:
        store.apply(batch, proposal)
    assert exc.value.details['check'] == check
    assert exc.value.details['source_index'] == 2
    assert store.memory['personal'] == [] and store.memory['last_processed_turn'] == 0


@pytest.mark.parametrize('section,field,value,check,limit', [
    ('topics', 'text', 'A' * 2164, 'text_length', 2000),
    ('topics', 'text', 'A' * 2154, 'text_length', 2000),
    ('personal', 'text', 'A' * 801, 'text_length', 800),
    ('hal', 'text', 'A' * 801, 'text_length', 800),
    ('topics', 'reason', '', 'reason_empty', 400),
    ('topics', 'reason', ' \n ', 'reason_empty', 400),
    ('topics', 'reason', 'A' * 401, 'reason_length', 400),
    ('topics', 'text', None, 'text_type', 2000),
])
def test_text_errors_identify_field_and_lengths_and_leave_whole_patch_uncommitted(
        store, section, field, value, check, limit):
    batch = pending(store)
    turn = batch['new_turns'][0]
    invalid = operation(turn, 'A memory.', section=section)
    invalid[field] = value
    proposal = change(operation(turn, 'A valid first memory.'), invalid)
    before = (store.directory / 'memory.json').read_bytes()
    with pytest.raises(MemoryValidationError) as exc:
        store.apply(batch, proposal)
    details = exc.value.details
    assert details['operation'] == {'index': 2, 'section': section, 'action': 'add', 'id': ''}
    assert details['check'] == check and details['field'] == field
    assert details['actual_length'] == (len(value) if isinstance(value, str) else None)
    assert details['max_length'] == limit
    assert (store.directory / 'memory.json').read_bytes() == before
    assert store.memory['last_processed_turn'] == 0


def test_repair_receives_exact_failure_and_success_keeps_prior_topic_as_continuation(store):
    initial = pending(store)
    store.apply(initial, change(operation(initial['new_turns'][0], 'Prior history. ' * 130, section='topics')))
    old = copy.deepcopy(store.memory['topics'][0])
    store.record_turn('Our choir is performing next month.', 'That sounds interesting.')
    batch = store.next_batch()
    turn = batch['new_turns'][0]
    rejected = change(operation(turn, 'A' * 2164, section='topics', action='extend', ident=old['id']))
    continuation = change(operation(turn, 'Choir discussion continued: a performance is planned for next month.', section='topics'))
    updater = Mock()
    def update(received, *, correction=None):
        assert received == batch
        assert store.memory['topics'] == [old]  # No partial application of the rejected proposal.
        if correction is None:
            return rejected
        assert correction['proposed_changes'] == rejected
        assert correction['validation']['actual_length'] == 2164
        assert correction['validation']['max_length'] == 2000
        assert '2164 characters; maximum is 2000' in correction['reason']
        return continuation
    updater.update.side_effect = update
    memory = worker(store, updater)
    assert memory._process_batch(batch) == (1, continuation)
    assert updater.update.call_count == 2
    assert store.memory['topics'][0] == old
    assert store.memory['topics'][1]['text'] == continuation['operations'][0]['text']
    assert store.memory['last_processed_turn'] == turn['id']
    memory.logger.warning.assert_not_called()


def test_repair_payload_reaches_api_with_original_batch_and_rejected_proposal():
    helper = memory_tests.APITests()
    updater, requests = helper.client(lambda _: memory_tests.completion(change()))
    batch = {'new_turns': [{'id': 98, 'user_speech': 'An actual statement.'}]}
    correction = {'reason': 'Topic text has 2164 characters; maximum is 2000.',
                  'validation': {'check': 'text_length', 'actual_length': 2164, 'max_length': 2000},
                  'proposed_changes': change()}
    try:
        updater.update(batch, correction=correction)
        messages = requests[0]['messages']
        assert [m['role'] for m in messages] == ['system', 'user', 'assistant', 'user']
        assert json.loads(messages[1]['content']) == batch
        assert json.loads(messages[2]['content']) == correction['proposed_changes']
        error = json.loads(messages[3]['content'])['application_validation_error']
        assert error == {key: correction[key] for key in ('reason', 'validation')}
        assert 'complete replacement' in messages[3]['content']
        assert 'never the rejected proposal' in messages[0]['content']
    finally:
        helper.doCleanups()


@pytest.mark.parametrize('failure', [RuntimeError('API unavailable'), ValueError('Incomplete/refused memory decision.')])
def test_error_during_correction_stops_without_another_request(store, failure):
    batch = pending(store)
    rejected = change(operation(batch['new_turns'][0], 'A' * 2164, section='topics'))
    updater = Mock()
    updater.update.side_effect = [rejected, failure]
    memory = worker(store, updater)
    assert memory._process_batch(batch) is None
    assert updater.update.call_count == 2
    assert store.memory['last_processed_turn'] == 0
    assert store.next_batch()['new_turns'] == batch['new_turns']


def test_external_file_edit_is_not_sent_to_model_or_overwritten(store):
    batch = pending(store)
    path = store.directory / 'memory.json'
    edited = path.read_text() + '\n'
    path.write_text(edited)
    updater = Mock()
    updater.update.return_value = change(operation(batch['new_turns'][0], 'Sings in a choir.'))
    memory = worker(store, updater)
    assert memory._process_batch(batch) is None
    updater.update.assert_called_once_with(batch)
    assert not memory.available
    assert path.read_text() == edited


def test_shutdown_before_correction_does_not_make_extra_request(store):
    batch = pending(store)
    updater = Mock()
    updater.update.return_value = change(operation(batch['new_turns'][0], 'A' * 2164, section='topics'))
    memory = worker(store, updater)
    memory.logger.info.side_effect = lambda *a, **kw: memory.stopping.set()
    assert memory._process_batch(batch) is None
    updater.update.assert_called_once_with(batch)
    assert store.memory['last_processed_turn'] == 0


def test_threaded_worker_can_record_during_correction_and_then_process_next_turn(store):
    batch = pending(store)
    rejected = change(operation(batch['new_turns'][0], 'A' * 2164, section='topics'))
    repairing, release, finished = threading.Event(), threading.Event(), threading.Event()
    updater, main = Mock(), Mock()
    def update(batch, *, correction=None):
        turn = batch['new_turns'][0]
        if turn['id'] == 1 and correction is None:
            return rejected
        if correction is not None:
            repairing.set()
            assert release.wait(3)
        return change(operation(turn, turn['user_speech'], section='topics'))
    updater.update.side_effect = update
    main.info.side_effect = lambda *a, **kw: finished.set() if store.memory['last_processed_turn'] == 2 else None
    memory = ConversationMemory(store, updater, MemorySettings(), main, Mock())
    try:
        assert repairing.wait(2)
        memory.record_turn('The concert is Friday.', 'Understood.')
        assert store.memory['last_processed_turn'] == 0
        assert len(memory.read_context()[0]) == 4
        release.set()
        assert finished.wait(2)
        assert updater.update.call_count == 3
        assert store.memory['last_processed_turn'] == 2
        assert len(store.memory['topics']) == 2
    finally:
        release.set()
        memory.close()
