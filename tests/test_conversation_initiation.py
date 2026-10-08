"""Initiation timing, one-request generation and real transcript attribution; no paid calls."""
import copy
from datetime import datetime, timedelta, timezone
import json
from unittest.mock import Mock, patch

import pytest

from conversation_initiation import ConversationInitiator, InitiationSettings, InitiationRequest, InitiationDecision
from conversation_memory import ConversationMemory, MemorySettings
from followup import FollowupSettings, FollowupDecision
from llm_client import LLMClient, LLMServiceError
from memory_store import MemoryStore, EvidenceValidationError
from test_memory_v2 import operation, change
import test_followup as followup_tests


class Clock:
    def __init__(self):
        self.value = datetime(2026, 10, 8, 9, tzinfo=timezone(timedelta(hours=-5)))
        self.tick = 0.

    def advance(self, seconds):
        self.tick += seconds
        self.value += timedelta(seconds=seconds)


@pytest.fixture
def memory(tmp_path):
    store = MemoryStore(tmp_path, logger=Mock())
    facade = ConversationMemory.__new__(ConversationMemory)
    facade.store, facade.logger, facade.decisions = store, Mock(), Mock()
    facade.available, facade.settings = True, MemorySettings(soft_tokens=500)
    facade.stopping, facade.wake = Mock(), Mock()
    facade.stopping.is_set.return_value = False
    try:
        yield facade
    finally:
        store.close()


def scheduler(memory, clock, **options):
    return ConversationInitiator(InitiationSettings(enabled=True, **options), memory, Mock(),
                                 now=lambda: clock.value, clock=lambda: clock.tick,
                                 choose=lambda first, last: first)


def test_settings_are_opt_in_and_unsupported_modes_fail_gracefully(memory, monkeypatch):
    monkeypatch.delenv('INITIATION_ENABLED', raising=False)
    assert not InitiationSettings.from_env().enabled
    for options in ({'daily_attempts': 3}, {'start_hour': 21, 'end_hour': 9},
                    {'quiet_seconds': float('nan')}, {'activity_db': float('inf')}):
        with pytest.raises(ValueError):
            InitiationSettings(**options)
    monkeypatch.setenv('INITIATION_ENABLED', 'true')
    for backend, followups, usable in [('ollama', True, True), ('openai', False, True),
                                     ('openai', True, False)]:
        memory.available = usable
        log = Mock()
        assert ConversationInitiator.from_env(log, memory, backend, FollowupSettings(followups)) is None
        log.warning.assert_called_once()


def test_quiet_activity_hours_and_restart_quota(memory):
    clock = Clock()
    ctrl = scheduler(memory, clock)
    assert ctrl.observe(speech=False, sound_db=-80) is None
    clock.advance(121)
    assert ctrl.observe(speech=False, sound_db=-80) is None  # Quiet alone is not presence.
    assert isinstance(ctrl.observe(speech=False, sound_db=-30), InitiationRequest)
    assert ctrl.begin(InitiationRequest(), 'Do you have a moment?')
    assert ctrl.state['due_at'] is None
    restarted = scheduler(memory, clock)
    assert restarted.pending is None
    assert restarted.state['attempts'][0]['status'] == 'interrupted'
    clock.advance(121)
    assert restarted.observe(speech=False, sound_db=-30) is None  # No repeat after restart.
    clock.advance(24 * 3600)
    assert restarted.observe(speech=True, sound_db=-20) is None
    clock.advance(100)
    assert restarted.observe(speech=False, sound_db=-30) is None
    clock.advance(21)
    assert isinstance(restarted.observe(speech=False, sound_db=-80), InitiationRequest)
    clock.value = clock.value.replace(hour=22)
    assert restarted.observe(speech=False, sound_db=-30) is None


def test_stale_activity_does_not_authorize_a_later_attempt(memory):
    clock = Clock()
    ctrl = scheduler(memory, clock)
    ctrl.observe(speech=False, sound_db=-30)
    clock.advance(301)
    assert ctrl.observe(speech=False, sound_db=-80) is None


def test_two_attempts_are_separated_and_missed_days_do_not_accumulate(memory):
    clock = Clock()
    ctrl = scheduler(memory, clock, daily_attempts=2)
    ctrl.begin(InitiationRequest(), 'Are you available?')
    ctrl.end_window()
    assert ctrl.state['due_at'] == clock.value.timestamp() + 10800
    clock.advance(600)
    assert ctrl.observe(speech=False, sound_db=-30) is None
    clock.advance(10200)
    assert ctrl.observe(speech=False, sound_db=-30) == InitiationRequest()
    ctrl.begin(InitiationRequest(), 'Are you available?')
    assert ctrl.state['due_at'] is None
    ctrl.end_window()
    clock.advance(7 * 86400)
    ctrl.observe(speech=False, sound_db=-30)
    assert len(ctrl.state['attempts']) == 2  # Opportunities aren't a backlog of utterances.


def test_manual_test_bypasses_schedule_but_expires_and_waits_for_speech_to_stop(memory):
    clock = Clock()
    clock.value = clock.value.replace(hour=23)
    ctrl = scheduler(memory, clock)
    ctrl.request_path.write_text(str(clock.value.timestamp()))
    assert ctrl.observe(speech=True, sound_db=-20) is None
    assert ctrl.request_path.exists()
    clock.advance(2)
    assert ctrl.observe(speech=False, sound_db=-80) == InitiationRequest(manual=True)
    assert not ctrl.request_path.exists()
    ctrl.request_path.write_text(str(clock.value.timestamp() - 121))
    assert ctrl.observe(speech=False, sound_db=-30) is None
    assert not ctrl.request_path.exists()


def test_storage_conflict_prevents_speaking_and_does_not_overwrite_edits(memory):
    clock = Clock()
    ctrl = scheduler(memory, clock)
    path = memory.store.directory / 'initiation.json'
    edited = path.read_text() + '\n'
    path.write_text(edited)
    assert not ctrl.begin(InitiationRequest(), 'Hello?')
    assert not ctrl.available and ctrl.pending is None
    assert path.read_text() == edited


def test_forget_barrier_excludes_old_openings_without_resetting_attempts(memory):
    clock = Clock()
    ctrl = scheduler(memory, clock)
    ctrl.begin(InitiationRequest(), 'Available?')
    ctrl.resolved(InitiationDecision('opening', 'How is your cat?', [], 'cat', 'Past discussion.'))
    assert ctrl.recent_attempts()[0]['opening'] == 'How is your cat?'
    memory.store.memory['context_after_turn'] = 4
    assert ctrl.recent_attempts() == []
    assert ctrl.state['due_at'] is None


def test_complete_catalogue_and_recent_transcripts_preserve_speakers_and_quotes(memory):
    store = memory.store
    store.record_turn('I sing in a choir.', 'I enjoy discussing musical structure.')
    batch = store.next_batch()
    turn = batch['new_turns'][0]
    store.apply(batch, change(operation(turn, 'The user sings in a choir.', tags=['choir', 'music']),
        operation(turn, 'HAL enjoys discussing musical structure.', section='hal'),
        operation(turn, 'A discussion about choir and musical structure.', section='topics')))
    lead = {'at': '2026-10-08T09:00:00-05:00', 'text': 'Do you have a moment?'}
    ident = store.record_turn('Yes.', 'What first got you interested in singing?', assistant_lead_in=lead)
    history, facts, ids = store.initiation_context()
    archive = json.loads(facts)
    assert all(len(archive[section]) == 1 for section in ('personal', 'hal', 'topics'))
    assert len(ids) == 3 and 'evidence' not in facts
    assert [m['role'] for m in history] == ['user', 'assistant', 'assistant', 'user', 'assistant']
    assert lead['text'] in history[-3]['content'] and history[-2]['content'].endswith('Yes.')
    batch = store.next_batch()
    turn = batch['new_turns'][0]
    assert turn['id'] == ident and turn['assistant_lead_in'] == lead
    accepted = MemoryStore._evidence([{'turn_id': ident, 'role': 'assistant', 'quote': lead['text']}],
                                     {ident: turn}, roles=('assistant',))
    assert accepted[0]['at'] == lead['at']
    with pytest.raises(EvidenceValidationError):
        MemoryStore._evidence([{'turn_id': ident, 'role': 'user', 'quote': lead['text']}], {ident: turn})
    store.close()
    reopened = MemoryStore(store.directory, logger=Mock())
    try:
        assert reopened.initiation_context()[0] == history
    finally:
        reopened.close()


def api_fixture(value, memory, *, finish='stop'):
    helper = followup_tests.DecisionTests()
    client, requests = helper.fixture(lambda _: followup_tests.completion(value, finish=finish))
    client.memory = memory
    return helper, client, requests


def test_initiation_includes_retained_pending_turns_beyond_the_ordinary_history_limit(memory):
    memory.store.max_history = 2
    for i in range(5):
        memory.store.record_turn(f'Question {i}', f'Answer {i}')
    assert len(memory.store.recent['turns']) == 5  # Updates have not run yet.
    history, _, _ = memory.store.initiation_context()
    assert len(history) == 10
    assert all(f'Answer {i}' in json.dumps(history) for i in range(5))


def result(decision='opening', reply='What got you interested in singing?', ids=()):
    return dict(decision=decision, reply=reply, memory_ids=list(ids), topic='singing', reason='Curiosity.')


def test_one_request_has_all_memory_and_untrimmed_recent_history_then_pins_selection(memory):
    for i in range(4):
        memory.store.record_turn(f'I enjoy interest number {i}.', f'We discussed interest {i}.')
        batch = memory.store.next_batch()
        memory.store.apply(batch, change(operation(batch['new_turns'][0], f'The user enjoys interest {i}.')))
    ids = memory.store.initiation_context()[2]
    helper, client, requests = api_fixture(result(ids=[ids[0]]), memory)
    try:
        client.begin_turn('Yes.')
        lead = {'at': '2026-10-08T09:00:00-05:00', 'text': 'Do you have a moment?'}
        decision = client.get_initiation_response('Yes.', lead_in=lead, recent_attempts=[])
        assert decision.decision == 'opening' and len(requests) == 1
        messages = requests[0]['messages']
        assert requests[0]['response_format']['json_schema']['name'] == 'hal_initiation'
        assert 'COMPLETE compact memory catalogue' in messages[1]['content']
        assert all(ident in messages[1]['content'] for ident in ids)
        assert all(f'We discussed interest {i}.' in json.dumps(messages) for i in range(4))
        assert [m['role'] for m in messages[-2:]] == ['assistant', 'user']
        assert lead['text'] in messages[-2]['content']
        client.finish_turn('Yes.', decision.reply)
        assert memory.store.recent['turns'][-1]['assistant_lead_in'] == lead
        client.begin_turn('How so?')
        assert ids[0] in client.memory_context
        client.end_initiation()
        assert client.initiation_memory_ids == []
    finally:
        helper.doCleanups()


@pytest.mark.parametrize('value,finish', [
    (result(ids=['invented_id']), 'stop'), (result(), 'length'),
    (result(reply='[IMAGE_REQUEST] {}'), 'stop'), (result(decision='ignore'), 'stop'),
])
def test_invalid_initiation_is_not_retried_or_committed(memory, value, finish):
    helper, client, requests = api_fixture(value, memory, finish=finish)
    before = copy.deepcopy(client.chat_history)
    try:
        with pytest.raises(LLMServiceError):
            client.get_initiation_response('Yes.', lead_in={'at': 'today', 'text': 'Available?'}, recent_attempts=[])
        assert len(requests) == 1 and client.chat_history == before
        assert client.assistant_lead_in is None
    finally:
        helper.doCleanups()


def test_idle_capture_closes_mic_and_never_transcribes_an_initiation_signal():
    voice, detector, clock, state = followup_tests.CaptureTests().fixture(speech=None)
    detector.analyze.return_value = {'matched': False, 'seconds': 0}
    ctrl, stream, trigger = Mock(), Mock(), Mock()
    ctrl.observe.return_value = InitiationRequest(True)
    with patch('voice_input.time', clock):
        assert voice.read_command(trigger, audio_stream=stream, initiation=ctrl) == InitiationRequest(True)
    assert state['capture_closed'] and state['keys_closed']
    trigger.assert_not_called()
    stream.start.assert_not_called()
    stream.cancel.assert_called_once()


def test_speech_during_slow_wake_decoding_is_not_mistaken_for_quiet():
    voice, detector, clock, _ = followup_tests.CaptureTests().fixture(speech=(1.2, 1.8))
    def decode(audio):
        clock.sleep(3.)  # The speech has ended before decoding returns.
        return {'matched': False, 'seconds': 3}
    detector.analyze.side_effect = decode
    ctrl = Mock()
    ctrl.observe.return_value = InitiationRequest(True)
    with patch('voice_input.time', clock):
        voice.read_command(initiation=ctrl)
    assert ctrl.observe.call_args.kwargs['speech'] is True


def test_active_followup_and_wake_commands_take_priority_over_initiation():
    for followup in (True, False):
        voice, detector, clock, state = followup_tests.CaptureTests().fixture()
        detector.analyze.return_value = {'matched': True, 'models': ['fixture'], 'seconds': 0,
                                        'transcripts': ['Hey HAL']}
        ctrl = Mock()
        with patch('voice_input.time', clock):
            captured = voice.read_command(initiation=ctrl, **({'followup_deadline': 5} if followup else {}))
        assert isinstance(captured, tuple)
        ctrl.observe.assert_not_called()


@pytest.mark.parametrize('outcome', ['opening', 'decline', 'ignore', 'silence', 'empty', 'error'])
def test_main_flow_checks_availability_before_any_llm_and_handles_outcomes(memory, outcome):
    ns, run, clock = followup_tests.MainLoopTests().fixture()
    ctrl = scheduler(memory, Clock())
    ns['initiator'] = ctrl
    reply = 'What got you interested in singing?' if outcome == 'opening' else 'Another time, then.'
    ns['llm'].get_initiation_response.return_value = InitiationDecision(
        outcome if outcome in ('opening', 'decline', 'ignore') else 'opening',
        '' if outcome == 'ignore' else reply, [], 'music', 'Curiosity.')
    if outcome == 'error':
        ns['llm'].get_initiation_response.side_effect = LLMServiceError('Service unavailable.')
    ns['stt'].transcribe.return_value = '' if outcome == 'empty' else 'Yes.'
    reads = []

    def read(on_trigger, **options):
        reads.append(options)
        if len(reads) == 1:
            assert options['initiation'] is ctrl
            return InitiationRequest(True)
        if len(reads) == 2:
            assert 'followup_deadline' in options and 'initiation' not in options
            ns['llm'].get_response.assert_not_called()
            ns['llm'].get_initiation_response.assert_not_called()
            ns['stt'].transcribe.assert_not_called()
            assert ns['play_audio'].call_args.kwargs['label'] == 'availability'
            if outcome == 'silence':
                return None
            on_trigger('followup')
            return [.1], 16000
        if outcome in ('decline', 'silence', 'error'):
            assert 'followup_deadline' not in options
        else:
            assert 'followup_deadline' in options
        raise KeyboardInterrupt

    ns['voice_input'].read_command.side_effect = read
    with pytest.raises(SystemExit):
        run()
    expected = 0 if outcome in ('silence', 'empty') else 1
    assert ns['llm'].get_initiation_response.call_count == expected
    ns['llm'].get_response.assert_not_called()
    ns['llm'].get_followup_response.assert_not_called()
    assert ns['llm'].finish_turn.call_count == (1 if outcome in ('opening', 'decline') else 0)
    assert len(ctrl.state['attempts']) == 1
    if outcome in ('silence', 'decline', 'error'):
        assert ctrl.state['attempts'][0]['status'] == {
            'silence': 'unanswered', 'decline': 'decline', 'error': 'failed'}[outcome]


def test_ignored_availability_answer_keeps_the_original_deadline(memory):
    ns, run, clock = followup_tests.MainLoopTests().fixture()
    ns['initiator'] = scheduler(memory, Clock())
    ns['llm'].get_initiation_response.side_effect = [
        InitiationDecision('ignore', '', [], '', 'Background speech.'),
        InitiationDecision('opening', 'How is choir going?', [], 'choir', 'Curiosity.')]
    ns['stt'].transcribe.side_effect = ['Bob, pass the remote.', 'Yes, HAL.']
    reads = []
    def read(on_trigger, **options):
        reads.append(options)
        if len(reads) == 1:
            return InitiationRequest(True)
        if len(reads) == 3:
            assert options['followup_deadline'] == reads[1]['followup_deadline']
        if len(reads) == 4:
            assert options['followup_deadline'] > reads[1]['followup_deadline']
            raise KeyboardInterrupt
        clock.advance(1.)
        on_trigger('followup')
        return [.1], 16000
    ns['voice_input'].read_command.side_effect = read
    with pytest.raises(SystemExit):
        run()
    assert ns['llm'].get_initiation_response.call_count == 2
    ns['llm'].finish_turn.assert_called_once_with('Yes, HAL.', 'How is choir going?')


def test_answering_the_opening_returns_to_normal_followups_and_records_engagement(memory):
    ns, run, clock = followup_tests.MainLoopTests().fixture()
    ctrl = ns['initiator'] = scheduler(memory, Clock())
    ns['llm'].get_initiation_response.return_value = InitiationDecision(
        'opening', 'What got you interested in singing?', [], 'music', 'Curiosity.')
    ns['llm'].get_followup_response.return_value = FollowupDecision('respond', 'That sounds like a fond memory.')
    ns['stt'].transcribe.side_effect = ['Yes.', 'Singing with my father.']
    reads = []
    def read(on_trigger, **options):
        reads.append(options)
        if len(reads) == 1:
            return InitiationRequest(True)
        if len(reads) == 4:
            assert ctrl.active is None and ctrl.state['attempts'][-1]['status'] == 'engaged'
            return None
        if len(reads) == 5:
            assert 'initiation' in options and 'followup_deadline' not in options
            raise KeyboardInterrupt
        on_trigger('followup')
        return [.1], 16000
    ns['voice_input'].read_command.side_effect = read
    with pytest.raises(SystemExit):
        run()
    ns['llm'].get_initiation_response.assert_called_once()
    ns['llm'].get_followup_response.assert_called_once_with('Singing with my father.', explicitly_addressed=False)
    assert [call.args for call in ns['llm'].finish_turn.call_args_list] == [
        ('Yes.', 'What got you interested in singing?'),
        ('Singing with my father.', 'That sounds like a fond memory.')]


def test_direct_request_in_place_of_consent_can_use_the_existing_action_flow(memory):
    ns, run, clock = followup_tests.MainLoopTests().fixture()
    ctrl = ns['initiator'] = scheduler(memory, Clock())
    ns['llm'].get_initiation_response.return_value = InitiationDecision(
        'respond', '[EXTERNAL_API_CALL] weather Minneapolis', [], '', 'User asked about weather.')
    ns['llm'].get_response.return_value = 'It is sunny.'
    ns['stt'].transcribe.return_value = 'Hey HAL, what is the weather?'
    reads = []
    def read(on_trigger, **options):
        reads.append(options)
        if len(reads) == 1:
            return InitiationRequest(True)
        if len(reads) == 3:
            assert ctrl.pending is None and ctrl.active is None
            assert ctrl.state['attempts'][-1]['status'] == 'respond'
            raise KeyboardInterrupt
        on_trigger('followup')
        return [.1], 16000
    ns['voice_input'].read_command.side_effect = read
    with pytest.raises(SystemExit):
        run()
    assert ns['llm'].get_initiation_response.call_args.kwargs['explicitly_addressed'] is True
    ns['handle_api_call'].assert_called_once_with('weather', ['Minneapolis'], 'Hey HAL, what is the weather?')
    ns['llm'].finish_turn.assert_called_once_with('Hey HAL, what is the weather?', 'It is sunny.')
