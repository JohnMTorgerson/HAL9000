"""Follow-up boundaries and application behavior; no paid API or microphone."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import httpx
import numpy as np
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from audio_capture import AudioHistory, AudioOverflowError
from followup import FollowupDecision, FollowupSession, FollowupSettings, explicitly_addresses_hal
from llm_client import LLMClient, LLMServiceError
from voice_input import VoiceInput, VoiceSettings, CommandTooLongError
import test_hal_recovery as hal_tests


class SessionTests(unittest.TestCase):
    def test_ignore_preserves_remaining_time_but_end_closes_immediately(self):
        clock = hal_tests.SimulatedClock()
        session = FollowupSession(FollowupSettings(enabled=True), clock.perf_counter)
        self.assertIsNone(session.deadline())
        clock.advance(10)
        session.after_response(explicit=True)
        self.assertEqual(session.deadline(), 18)
        clock.advance(3)  # Capturing, transcribing and ignoring uses real time.
        self.assertEqual(session.deadline(), 18)
        self.assertEqual(session.deadline() - clock.now, 5)
        session.close()  # end, rather than ignore
        self.assertIsNone(session.deadline())

    def test_accepted_turns_do_not_move_session_limit_and_explicit_wake_can_reset_it(self):
        clock = hal_tests.SimulatedClock()
        session = FollowupSession(FollowupSettings(enabled=True), clock.perf_counter)
        session.after_response(explicit=True)
        for _ in range(17):
            clock.advance(7)
            session.after_response(explicit=False)
            self.assertEqual(session.session_end, 120)
            self.assertLessEqual(session.deadline(), 120)
        self.assertEqual(session.deadline(), 120)
        clock.advance(3)  # A turn can finish after its start deadline.
        session.after_response(explicit=False)
        self.assertIsNone(session.deadline())
        session.after_response(explicit=True)
        self.assertEqual(session.session_end, 242)
        self.assertEqual(session.deadline(), 130)

    def test_timeout_and_disabled_mode_never_open_a_window(self):
        clock = hal_tests.SimulatedClock()
        off = FollowupSession(FollowupSettings(), clock.perf_counter)
        off.after_response(explicit=True)
        self.assertIsNone(off.deadline())
        on = FollowupSession(FollowupSettings(enabled=True), clock.perf_counter)
        on.after_response(explicit=True)
        clock.advance(8)
        self.assertIsNone(on.deadline())
        on.after_response(explicit=False)
        self.assertIsNone(on.deadline())

    def test_configuration_and_explicit_address_do_not_treat_hey_how_as_an_override(self):
        with patch.dict('os.environ', {}, clear=True):
            self.assertEqual(FollowupSettings.from_env(), FollowupSettings())
        with patch.dict('os.environ', {'FOLLOWUP_ENABLED': 'True', 'FOLLOWUP_WINDOW_SECONDS': '6',
                                      'FOLLOWUP_SESSION_SECONDS': '90'}, clear=True):
            self.assertEqual(FollowupSettings.from_env(), FollowupSettings(True, 6, 90))
        for values in ({'window': 0}, {'window': 31}, {'window': float('nan')},
                       {'session_limit': 3}, {'session_limit': float('inf')}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                FollowupSettings(**values)
        with patch.dict('os.environ', {'FOLLOWUP_ENABLED': 'maybe'}), self.assertRaises(ValueError):
            FollowupSettings.from_env()
        for text in ('Hey, HAL!', 'Hey Al, what time is it?', 'Hey howl, what next?'):
            self.assertTrue(explicitly_addresses_hal(text))
        for text in ('Hey, how many tablespoons?', 'Hey Halley', 'Tell Hal about it'):
            self.assertFalse(explicitly_addresses_hal(text))


def completion(value, *, finish='stop', refusal=None):
    return httpx.Response(200, json={'id': 'fixture', 'object': 'chat.completion',
        'created': 0, 'model': 'gpt-6-luna', 'service_tier': 'priority', 'choices': [
            {'index': 0, 'finish_reason': finish, 'message': {'role': 'assistant',
             'content': json.dumps(value) if isinstance(value, dict) else value, 'refusal': refusal}}]})


class DecisionTests(unittest.TestCase):
    def fixture(self, handler):
        requests = []
        def handle(request):
            requests.append(json.loads(request.content))
            return handler(len(requests))
        transport = httpx.Client(transport=httpx.MockTransport(handle))
        api = OpenAI(api_key='fixture-not-a-key', http_client=transport, max_retries=0)
        self.addCleanup(api.close)
        with patch('llm_client.OpenAI', return_value=api):
            client = LLMClient('openai', 'gpt-6-luna', max_history=2,
                               openai_api_key='fixture-not-a-key', service_tier='fast', logger=Mock())
        client.chat_history = [{'role': 'user', 'content': 'What is the weather?'},
                               {'role': 'assistant', 'content': 'It will rain today.'}]
        return client, requests

    def test_one_request_contains_context_schema_and_fast_settings_then_commits_plain_reply(self):
        client, requests = self.fixture(lambda _: completion({'decision': 'respond', 'reply': 'Sunny tomorrow.'}))
        result = client.get_followup_response('And tomorrow?')
        self.assertEqual(result, FollowupDecision('respond', 'Sunny tomorrow.'))
        self.assertEqual(len(requests), 1)
        body = requests[0]
        self.assertEqual(body['response_format']['type'], 'json_schema')
        self.assertTrue(body['response_format']['json_schema']['strict'])
        self.assertEqual(body['service_tier'], 'priority')
        self.assertEqual(body['reasoning_effort'], 'none')
        self.assertEqual(body['max_completion_tokens'], 512)
        self.assertIn('It will rain today.', json.dumps(body['messages']))
        self.assertIn('explicitly_addressed=false', body['messages'][0]['content'])
        self.assertEqual(client.chat_history[-1], {'role': 'assistant', 'content': 'Sunny tomorrow.'})
        self.assertIn('And tomorrow?', client.chat_history[-2]['content'])

    def test_ignore_and_end_neither_trim_nor_pollute_history(self):
        for decision in ('ignore', 'end'):
            with self.subTest(decision=decision):
                client, requests = self.fixture(lambda n: completion(
                    {'decision': decision, 'reply': ''} if n == 1 else {'decision': 'respond', 'reply': 'Ready.'}))
                client.chat_history *= 3  # Verify rejected turns don't even trim old history.
                before = copy.deepcopy(client.chat_history)
                self.assertEqual(client.get_followup_response('Private background fragment').decision, decision)
                self.assertEqual(client.chat_history, before)
                client.get_followup_response('Hey HAL, are you there?', explicitly_addressed=True)
                self.assertNotIn('Private background fragment', json.dumps(requests[1]['messages']))
                self.assertIn('explicitly_addressed=true', requests[1]['messages'][0]['content'])

    def test_external_command_is_unwrapped_and_subsequent_external_response_uses_normal_protocol(self):
        command = '[EXTERNAL_API_CALL] calendar_next_event'
        client, requests = self.fixture(lambda n: completion(
            {'decision': 'respond', 'reply': command} if n == 1 else 'Your event is tomorrow.'))
        self.assertEqual(client.get_followup_response('What is next on my calendar?').reply, command)
        self.assertEqual(client.get_response('[EXTERNAL_API_RESPONSE] fixture'), 'Your event is tomorrow.')
        self.assertNotIn('response_format', requests[1])
        self.assertNotIn('FOLLOW-UP CONTROL', requests[1]['messages'][0]['content'])
        self.assertEqual(client.chat_history[-3]['content'], command)

    def test_invalid_truncated_or_refused_decisions_are_never_retried_or_committed(self):
        cases = [
            completion('[EXTERNAL_API_CALL] weather default'),
            completion({'decision': 'respond', 'reply': 'partial'}, finish='length'),
            completion(None, refusal='fixture refusal'),
            completion({'decision': 'maybe', 'reply': ''}),
            completion({'decision': 'ignore', 'reply': '[EXTERNAL_API_CALL] calendar_next_event'}),
            completion({'decision': 'end', 'reply': 'Goodbye.'}),
            completion({'decision': 'respond', 'reply': ' '}),
            completion({'decision': 'respond', 'reply': None}),
            completion({'decision': 'ignore'}),
            completion({'decision': 'ignore', 'reply': '', 'unexpected': True}),
        ]
        for response in cases:
            with self.subTest(response=response.json()):
                client, requests = self.fixture(lambda _: response)
                before = copy.deepcopy(client.chat_history)
                with self.assertRaises(LLMServiceError):
                    client.get_followup_response('Unconfirmed speech')
                self.assertEqual(client.chat_history, before)
                self.assertEqual(len(requests), 1)

    def test_service_failure_and_unsupported_backend_leave_history_unchanged(self):
        client, requests = self.fixture(lambda _: httpx.Response(429, json={'error': {
            'message': 'fixture', 'code': 'credit_balance_exhausted', 'type': 'insufficient_quota'}}))
        before = copy.deepcopy(client.chat_history)
        with self.assertRaisesRegex(LLMServiceError, 'credits'):
            client.get_followup_response('Unconfirmed speech')
        self.assertEqual(client.chat_history, before)
        self.assertEqual(len(requests), 1)
        with self.assertRaisesRegex(LLMServiceError, 'openai'):
            LLMClient('ollama', 'fixture').get_followup_response('hello')


class CaptureClock(hal_tests.SimulatedClock):
    capture = None
    def sleep(self, seconds):
        self.advance(seconds)
        self.capture.feed()


class CaptureTests(unittest.TestCase):
    def fixture(self, speech=(.3, .9), press=None):
        clock, logger = CaptureClock(), Mock()
        signal = np.zeros(12 * 16000, np.int16)
        if speech:
            signal[round(speech[0] * 16000):round(speech[1] * 16000)] = 1234
        state = {}
        class Capture:
            def __init__(self, **kwargs):
                self.rate = 16000
                self.latency = kwargs['latency']
                self.device_name = 'fixture'
                self.history = AudioHistory(self.rate)
                clock.capture = self
            def start(self):
                pass
            def feed(self):
                total, _ = self.history.position()
                delivered = max(0, round((clock.now - self.latency) * self.rate))
                if delivered > total:
                    self.history.append(signal[total:delivered], wall=clock.now)
            def check(self):
                self.feed()
            def close(self):
                state['capture_closed'] = True
        class Keys:
            available = True
            def __init__(self, _):
                pass
            def start(self):
                pass
            def snapshot(self):
                if press and clock.now >= press[0]:
                    return press[0], press[1] if clock.now >= press[1] else None
                return None, None
            def close(self):
                state['keys_closed'] = True
        def bounds(audio):
            voiced = np.flatnonzero(audio)
            return (int(voiced[0]), int(voiced[-1] + 1)) if len(voiced) >= 1600 else None
        detector = Mock()
        detector.speech_bounds.side_effect = bounds
        detector.last_speech_sample.side_effect = lambda audio: bounds(audio)[1] if bounds(audio) else None
        voice = VoiceInput(VoiceSettings(silence=.5, maximum=8), logger, detector,
                           capture_factory=Capture, keyboard_factory=Keys)
        return voice, detector, clock, state

    def test_followup_uses_local_vad_and_retains_complete_utterance_without_wake_decode(self):
        voice, detector, clock, state = self.fixture()
        triggered = []
        with patch('voice_input.time', clock):
            audio, rate = voice.read_command(triggered.append, followup_deadline=2)
        self.assertEqual(triggered, ['followup'])
        self.assertEqual(rate, 16000)
        self.assertEqual(np.count_nonzero(audio), round(.6 * 16000))
        self.assertAlmostEqual(voice.last_speech_end_at, .9)
        detector.analyze.assert_not_called()
        self.assertTrue(state['capture_closed'] and state['keys_closed'])

    def test_silence_and_speech_starting_after_deadline_expire_without_upload(self):
        for speech in (None, (1.05, 2)):
            with self.subTest(speech=speech):
                voice, detector, clock, state = self.fixture(speech)
                stream, trigger = Mock(), Mock()
                with patch('voice_input.time', clock):
                    self.assertIsNone(voice.read_command(trigger, audio_stream=stream, followup_deadline=1))
                self.assertLess(clock.now, 1.5)
                trigger.assert_not_called()
                stream.start.assert_not_called()
                stream.append.assert_not_called()
                stream.end_audio.assert_not_called()
                detector.analyze.assert_not_called()
                self.assertTrue(state['capture_closed'] and state['keys_closed'])

    def test_speech_just_before_deadline_is_allowed_to_finish(self):
        voice, detector, clock, state = self.fixture((.95, 1.6))
        triggered = []
        with patch('voice_input.time', clock):
            audio, rate = voice.read_command(triggered.append, followup_deadline=1)
        self.assertGreater(clock.now, 2)
        self.assertEqual(triggered, ['followup'])
        self.assertEqual(np.count_nonzero(audio), round(.65 * 16000))

    def test_spacebar_overrides_an_automatic_followup_and_is_not_filtered(self):
        voice, detector, clock, state = self.fixture((.2, .7), press=(.65, 1.))
        triggered = []
        with patch('voice_input.time', clock):
            audio, _ = voice.read_command(triggered.append, followup_deadline=2)
        self.assertEqual(triggered, ['followup', 'spacebar'])
        self.assertEqual(np.count_nonzero(audio), 8000)

    def test_discarded_followup_cancels_live_turn_and_closes_capture(self):
        for failure in ('overflow', 'too_long'):
            with self.subTest(failure=failure):
                voice, detector, clock, state = self.fixture((.3, 11) if failure == 'too_long' else (.3, .9))
                stream = Mock()
                def trigger(kind):
                    if failure == 'overflow':
                        clock.capture.check = Mock(side_effect=AudioOverflowError('lost audio'))
                with patch('voice_input.time', clock), self.assertRaises(
                        AudioOverflowError if failure == 'overflow' else CommandTooLongError):
                    voice.read_command(trigger, audio_stream=stream, followup_deadline=2)
                stream.end_audio.assert_not_called()
                stream.cancel.assert_called_once()
                self.assertTrue(state['capture_closed'] and state['keys_closed'])

    def test_followup_vad_can_load_without_wake_model_and_missing_vad_exits_safely(self):
        speech = Mock()
        with patch.dict('os.environ', {'WAKE_ENABLED': 'false'}, clear=True), \
             patch('wake_detector.SpeechActivityDetector', return_value=speech):
            voice = VoiceInput.from_env(Mock(), followup_enabled=True)
        self.assertIsNone(voice.detector)
        self.assertIs(voice.speech_detector, speech)
        voice.speech_detector = None
        voice.capture_factory = Mock(side_effect=AssertionError('No capture should open'))
        self.assertIsNone(voice.read_command(followup_deadline=float('inf')))


class MainLoopTests(unittest.TestCase):
    def fixture(self):
        ns, run = hal_tests.MainLoopTests().fixture()
        clock = hal_tests.SimulatedClock()
        ns.update(time=clock, followup_settings=FollowupSettings(enabled=True),
                  wave=MagicMock(), voice=Mock(), syn_config=object(), USER='fixture',
                  strip_name_at_sentence_end=lambda text, name: text, sf=Mock())
        ns['sf'].read.return_value = ([.1], 16000)
        ns['voice'].synthesize_wav.side_effect = lambda *a, **kw: clock.advance(.2)
        ns['play_audio'].side_effect = lambda *a, **kw: clock.advance(1.)
        return ns, run, clock

    def test_ignored_speech_keeps_deadline_accepted_external_reply_renews_it_and_end_closes(self):
        ns, run, clock = self.fixture()
        ns['DEBUG_PLAYBACK'] = True  # Unconfirmed speech must still remain silent.
        reads, deadlines, playback_ends = [], [], []
        texts = iter(['What is the weather?', 'Bob, pass the remote.', 'What about tomorrow?', 'That is all, HAL.'])
        ns['stt'].transcribe.side_effect = lambda *a: next(texts)
        ns['llm'].get_response.side_effect = ['Rain today.', 'Sun tomorrow.']
        decisions = iter([FollowupDecision('ignore'),
                          FollowupDecision('respond', '[EXTERNAL_API_CALL] weather tomorrow'),
                          FollowupDecision('end')])
        ns['llm'].get_followup_response.side_effect = lambda *a, **kw: clock.advance(1., next(decisions))
        def play(*args, **kwargs):
            clock.advance(1.)
            if kwargs['label'] == 'reply':
                playback_ends.append(clock.now)
        ns['play_audio'].side_effect = play
        def read(on_trigger, **kwargs):
            reads.append(kwargs)
            if len(reads) == 1:
                self.assertNotIn('followup_deadline', kwargs)
                clock.advance(100.)  # Idle before explicit wake doesn't consume the session.
                on_trigger('wakeword')
            elif len(reads) <= 4:
                deadline = kwargs['followup_deadline']
                deadlines.append(deadline)
                if len(reads) == 3:
                    self.assertEqual(deadline, deadlines[0])  # ignore didn't renew
                else:
                    self.assertEqual(deadline, playback_ends[-1] + 8)
                clock.advance(.2)
                on_trigger('followup')
            else:
                self.assertNotIn('followup_deadline', kwargs)  # end closes immediately
                raise KeyboardInterrupt
            clock.advance(.5)
            return [.1], 16000
        ns['voice_input'].read_command.side_effect = read
        with self.assertRaises(SystemExit):
            run()
        ns['handle_api_call'].assert_called_once_with('weather', ['tomorrow'], 'What about tomorrow?')
        self.assertEqual(ns['voice'].synthesize_wav.call_count, 2)
        self.assertEqual([call.kwargs['label'] for call in ns['play_audio'].call_args_list],
                         ['debug query', 'reply', 'acknowledgment', 'reply'])
        self.assertEqual(ns['llm'].get_response.call_count, 2)  # initial and external follow-through
        self.assertEqual(ns['llm'].get_followup_response.call_count, 3)
        displayed = [call.args[0] for call in ns['logger'].display.call_args_list]
        self.assertFalse(any('Bob' in text or 'That is all' in text for text in displayed))
        self.assertFalse(any('decision' in text or 'BACKGROUND' in text for text in displayed))
        self.assertIn('USER: What about tomorrow?', displayed)
        heard = [call.args[1] for call in ns['logger'].info.call_args_list
                 if call.args[0] == 'FOLLOWUP heard: %s']
        self.assertEqual(heard, ['Bob, pass the remote.', 'What about tomorrow?', 'That is all, HAL.'])
        for call in ns['logger'].info.call_args_list:
            if call.args[0] == 'FOLLOWUP heard: %s':
                self.assertEqual(call.kwargs['extra'], {'speech_role': 'user'})

    def test_direct_wake_in_followup_is_signalled_to_luna_and_resets_session_after_reply(self):
        ns, run, clock = self.fixture()
        ns['followup_settings'] = FollowupSettings(True, 8, 8)
        ns['stt'].transcribe.side_effect = ['First request.', 'Hey HAL, a new question.']
        ns['llm'].get_response.return_value = 'Ready.'
        ns['llm'].get_followup_response.return_value = FollowupDecision('respond', 'A new answer.')
        reads = []
        def read(on_trigger, **kwargs):
            reads.append(kwargs)
            if len(reads) == 1:
                on_trigger('wakeword')
            elif len(reads) == 2:
                clock.advance(7)
                on_trigger('followup')
                clock.advance(2)  # Explicitly addressing HAL can start a new session.
            else:
                self.assertGreater(kwargs['followup_deadline'], reads[1]['followup_deadline'])
                self.assertEqual(kwargs['followup_deadline'], clock.now + 8)
                raise KeyboardInterrupt
            return [.1], 16000
        ns['voice_input'].read_command.side_effect = read
        with self.assertRaises(SystemExit):
            run()
        ns['llm'].get_followup_response.assert_called_once_with(
            'Hey HAL, a new question.', explicitly_addressed=True)

    def test_bad_followup_decision_closes_window_without_tts_or_external_requests(self):
        ns, run, clock = self.fixture()
        ns['llm'].get_response.return_value = 'Ready.'
        ns['llm'].get_followup_response.side_effect = LLMServiceError('Invalid follow-up decision.')
        reads = []
        def read(on_trigger, **kwargs):
            reads.append(kwargs)
            if len(reads) == 3:
                self.assertNotIn('followup_deadline', kwargs)
                raise KeyboardInterrupt
            on_trigger('wakeword' if len(reads) == 1 else 'followup')
            return [.1], 16000
        ns['voice_input'].read_command.side_effect = read
        with self.assertRaises(SystemExit):
            run()
        self.assertEqual(ns['voice'].synthesize_wav.call_count, 1)
        ns['handle_api_call'].assert_not_called()
        self.assertEqual(ns['llm'].get_response.call_count, 1)

    def test_silent_live_followup_timeout_closes_unused_stream_without_transcription(self):
        ns, run, clock = self.fixture()
        ns['stt'].mode = 'live'
        ns['llm'].get_response.return_value = 'Ready.'
        streams = [Mock(), Mock(), Mock()]
        ns['stt'].create_live_stream.side_effect = streams
        reads = []
        def read(on_trigger, **kwargs):
            reads.append(kwargs)
            if len(reads) == 1:
                on_trigger('spacebar')
                return [.1], 16000
            if len(reads) == 2:
                self.assertIn('followup_deadline', kwargs)
                return None
            streams[1].close.assert_called_once()
            self.assertNotIn('followup_deadline', kwargs)
            raise KeyboardInterrupt
        ns['voice_input'].read_command.side_effect = read
        with self.assertRaises(SystemExit):
            run()
        self.assertEqual(ns['stt'].transcribe.call_count, 1)
        ns['llm'].get_followup_response.assert_not_called()
        for stream in streams:
            stream.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
