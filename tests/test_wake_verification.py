"""Wake verification at the real main-loop boundary; no hardware or paid API."""
import io
import logging
from unittest.mock import patch

import pytest

import test_followup as followup_tests
import test_voice_input as voice_tests
from followup import FollowupDecision
from live_transcription import NoSpeechError


@pytest.fixture
def app():
    ns, run, _ = followup_tests.MainLoopTests().fixture()
    ns['stt'].backend = 'api'
    ns['llm'].get_response.return_value = 'Ready.'
    return ns, run


def info_messages(ns):
    return [call.args[0] % call.args[1:] for call in ns['logger'].info.call_args_list]


@pytest.mark.parametrize('text', [
    "Hey, let's go.", 'Hey now, wait a minute.', 'Apparently no one does.',
    'Hey Halley!', 'Hey Howard!', 'Hey howling!', '', ' \n\t', None,
])
def test_rejection_returns_to_wake_listening_without_llm_display_or_memory(app, text):
    ns, run = app
    if text is None:
        ns['stt'].transcribe.side_effect = NoSpeechError('Transcription returned no speech.')
    else:
        ns['stt'].transcribe.return_value = text
    reads = []

    def read(on_trigger, **options):
        reads.append(options)
        assert 'followup_deadline' not in options
        if len(reads) == 2:
            assert ns['led'].method_calls[-1][0] == 'off'
            ns['DisplayServerManager'].return_value.stop.assert_not_called()
            raise KeyboardInterrupt
        on_trigger('wakeword')
        ns['led'].on.assert_called_once()
        return [.1], 16000

    ns['voice_input'].read_command.side_effect = read
    with pytest.raises(SystemExit):
        run()
    ns['stt'].transcribe.assert_called_once()
    ns['images'].context.assert_not_called()
    for method in ('begin_turn', 'get_response', 'get_followup_response',
                   'get_initiation_response', 'finish_turn'):
        getattr(ns['llm'], method).assert_not_called()
    ns['handle_api_call'].assert_not_called()
    ns['display_handler'].begin_response.assert_not_called()
    ns['logger'].display.assert_not_called()
    ns['logger'].warning.assert_not_called()
    ns['logger'].error.assert_not_called()
    ns['play_audio'].assert_not_called()
    ns['voice'].synthesize_wav.assert_not_called()
    rejections = [msg for msg in info_messages(ns) if msg.startswith('Wake verification rejected')]
    assert len(rejections) == 1
    assert ('[empty transcription]' if text is None else repr(text)) in rejections[0]


@pytest.mark.parametrize('alias', ['Hal', 'Hall', 'Hell', 'How', 'Al', 'Howl'])
@pytest.mark.parametrize('prefix', ['', 'Some buffered speech. Um, '])
def test_same_lax_aliases_survive_case_punctuation_and_leading_speech(app, alias, prefix):
    ns, run = app
    text = f'{prefix}HEY, {alias}! Are you there?'
    ns['stt'].transcribe.return_value = text
    reads = []

    def read(on_trigger, **options):
        reads.append(options)
        if len(reads) == 2:
            assert 'followup_deadline' in options
            raise KeyboardInterrupt
        on_trigger('wakeword')
        return [.1], 16000

    ns['voice_input'].read_command.side_effect = read
    with pytest.raises(SystemExit):
        run()
    ns['stt'].transcribe.assert_called_once()
    ns['llm'].get_response.assert_called_once_with(text)
    ns['llm'].finish_turn.assert_called_once_with(text, 'Ready.')
    assert any(msg.startswith('Wake verification accepted:') for msg in info_messages(ns))


@pytest.mark.parametrize('kind,backend,mode', [
    ('spacebar', 'api', 'static'), ('spacebar', 'api', 'live'),
    ('wakeword', 'api', 'live'), ('wakeword', 'local', 'static'),
])
def test_other_capture_and_transcription_paths_do_not_require_a_wake_phrase(app, kind, backend, mode):
    ns, run = app
    ns['stt'].backend, ns['stt'].mode = backend, mode
    ns['stt'].transcribe.return_value = 'How are you?'
    reads = []

    def read(on_trigger, **options):
        reads.append(options)
        if len(reads) == 2:
            assert 'followup_deadline' in options
            raise KeyboardInterrupt
        on_trigger(kind)
        return [.1], 16000

    ns['voice_input'].read_command.side_effect = read
    with pytest.raises(SystemExit):
        run()
    ns['llm'].get_response.assert_called_once_with('How are you?')
    ns['llm'].finish_turn.assert_called_once_with('How are you?', 'Ready.')
    assert not any(msg.startswith('Wake verification') for msg in info_messages(ns))


def test_real_wake_after_rejection_and_unaddressed_followup_work_normally(app):
    ns, run = app
    ns['stt'].transcribe.side_effect = ['Hey, now wait.', 'Hey HAL, are you there?', 'How are you?']
    ns['llm'].get_followup_response.return_value = FollowupDecision('respond', 'Functioning perfectly.')
    reads = []

    def read(on_trigger, **options):
        reads.append(options)
        if len(reads) <= 2:
            assert 'followup_deadline' not in options
        else:
            assert 'followup_deadline' in options
        if len(reads) == 4:
            assert options['followup_deadline'] > reads[2]['followup_deadline']
            raise KeyboardInterrupt
        on_trigger('followup' if len(reads) == 3 else 'wakeword')
        return [.1], 16000

    ns['voice_input'].read_command.side_effect = read
    with pytest.raises(SystemExit):
        run()
    assert ns['stt'].transcribe.call_count == 3
    ns['llm'].get_response.assert_called_once_with('Hey HAL, are you there?')
    ns['llm'].get_followup_response.assert_called_once_with('How are you?', explicitly_addressed=False)
    assert [call.args for call in ns['llm'].finish_turn.call_args_list] == [
        ('Hey HAL, are you there?', 'Ready.'), ('How are you?', 'Functioning perfectly.')]
    assert len([msg for msg in info_messages(ns) if msg.startswith('Wake verification')]) == 2


def test_info_log_keeps_local_evidence_once_even_when_final_transcript_is_rejected(app):
    ns, run = app
    clock, _, _, voice = voice_tests.FlowTests().fixture()
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setLevel(logging.INFO)  # Match the main log's threshold.
    logger = logging.Logger('wake-verification-test', level=logging.DEBUG)
    logger.addHandler(handler)
    voice.logger = ns['logger'] = logger
    original_analyze = voice.detector.analyze
    scans = []

    def analyze(audio):
        scans.append(True)
        if len(scans) == 1:
            return {'matched': False, 'models': [], 'seconds': 0,
                    'transcripts': {'base.en': 'Unmatched rolling scan'}}
        return original_analyze(audio)

    voice.detector.analyze = analyze
    ns['stt'].transcribe.return_value = 'Hey now, wait a minute.'
    reads = []

    def read(on_trigger, **options):
        reads.append(options)
        if len(reads) == 2:
            assert 'followup_deadline' not in options
            raise KeyboardInterrupt
        return voice.read_command(on_trigger)

    ns['voice_input'].read_command.side_effect = read
    with patch('voice_input.time', clock), pytest.raises(SystemExit):
        run()
    messages = output.getvalue()
    assert len(scans) == 2
    assert 'Unmatched rolling scan' not in messages
    assert messages.count('Wake transcript:') == 1
    assert "Wake transcript: {'base.en': 'Hey Hal, how many'}" in messages
    assert messages.count('Wake verification rejected') == 1
    assert "final transcript='Hey now, wait a minute.'" in messages
    assert messages.index('Wake transcript:') < messages.index('Wake verification rejected')
    ns['llm'].begin_turn.assert_not_called()
