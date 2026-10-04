"""Exact image recall across restart, bounded storage, and display retries."""
import io
import json
from unittest.mock import Mock, patch

from PIL import Image
import pytest

from display_client import DisplayClient
from image_history import ImageHistory
from image_lookup import ImageCandidate, ImageResult, ImageWorkflow, UnsupportedImageProvider


def command(action, query=''):
    return '[IMAGE_REQUEST] ' + json.dumps({'action': action, 'query': query})


def picture(color='yellow'):
    raw = io.BytesIO()
    Image.new('RGB', (300, 200), color).save(raw, format='JPEG')
    return ImageCandidate(color, f'https://example.com/{color}.jpg',
                          f'https://example.com/cars/{color}', data=raw.getvalue(),
                          reply=f'Here is the {color} Elise.',
                          description=f'A {color} Lotus Elise in profile.',
                          citations=[{'url': 'https://example.com/credit', 'label': 'Photographer'}])


def workflow(directory, candidates=None, statuses=None):
    provider, display = Mock(supported=True), Mock()
    provider.lookup.return_value = ImageResult('ok', images=candidates or [picture()])
    display.present_image.side_effect = statuses
    display.present_image.return_value = 'loaded'
    return ImageWorkflow(provider, display, history=ImageHistory(directory)), provider, display


def test_restart_preserves_exact_bytes_and_sources_without_search_or_auto_display(tmp_path):
    first, provider, display = workflow(tmp_path)
    result = first.handle_reply(command('search', 'Lotus Elise'))
    saved_id = first.history.catalogue()[0]['id']
    assert saved_id in result.action_result
    assert picture().image_url in result.action_result
    assert first.handle_reply(command('close')).status == 'closed'

    # This is a new controller and a new archive instance, like restarting HAL.
    second, provider, display = workflow(tmp_path)
    context = json.loads(second.context())
    assert context['cached_images'] == 0
    assert context['saved_images'][0]['description'] == picture().description
    display.present_image.assert_not_called()
    result = second.handle_reply(command('recall', saved_id))
    assert result.status == 'ok' and 'again' in result.reply
    provider.lookup.assert_not_called()
    assert display.present_image.call_args.args[0] == picture().data
    assert display.present_image.call_args.kwargs['citations'] == [
        {'url': picture().source_url, 'label': 'example.com'}, *picture().citations]


def test_named_recall_chooses_most_recent_matching_subject_not_latest_other_image(tmp_path):
    history = ImageHistory(tmp_path)
    old_id = history.remember(picture('yellow'), 'Lotus Elise')
    latest_id = history.remember(picture('red'), 'Lotus Elise')
    phone = picture('blue')
    phone.description = 'A blue phone.'
    history.remember(phone, 'latest phone')
    restored = ImageHistory(tmp_path)
    assert restored.recall('Lotus Elise')[1].id == latest_id
    assert restored.recall(old_id)[1].data == picture('yellow').data
    assert restored.recall('yellow Lotus Elise')[1].id == old_id
    assert restored.recall()[1].data == phone.data
    assert restored.recall('Mustang') is None
    assert restored.recall('img_not_a_saved_id') is None


def test_archive_is_bounded_and_recalls_refresh_recency_without_duplicating_files(tmp_path):
    history = ImageHistory(tmp_path, limit=2)
    yellow_id = history.remember(picture('yellow'), 'Elise')
    red_id = history.remember(picture('red'), 'Elise')
    assert history.remember(picture('yellow'), 'Elise') == yellow_id
    blue_id = history.remember(picture('blue'), 'Elise')
    history = ImageHistory(tmp_path, limit=2)
    assert [r['id'] for r in history.catalogue()] == [blue_id, yellow_id]
    assert history.recall(red_id) is None
    assert len(list(tmp_path.glob('*.jpg'))) == 2


@pytest.mark.parametrize('damage', ['missing', 'changed'])
def test_missing_or_damaged_exact_picture_never_substitutes_a_different_one(tmp_path, damage):
    history = ImageHistory(tmp_path)
    saved_id = history.remember(picture(), 'Elise')
    file = next(tmp_path.glob('*.jpg'))
    if damage == 'missing':
        file.unlink()
    else:
        file.write_bytes(picture('red').data)
    history.remember(picture('blue'), 'Elise')
    controller, provider, display = workflow(tmp_path)
    result = controller.handle_reply(command('recall', saved_id))
    assert result.status == 'no_results' and 'exact image' in result.reply
    provider.lookup.assert_not_called()
    display.present_image.assert_not_called()


def test_failed_display_retries_same_verified_image_and_archives_only_after_success(tmp_path):
    controller, provider, display = workflow(tmp_path, [picture(), picture('red')],
                                             ['unavailable', 'loaded'])
    assert controller.handle_reply(command('search', 'Elise')).status == 'error'
    state = json.loads(controller.context())
    assert state['retry_available'] and state['pending_image_description'] == picture().description
    assert state['saved_images'] == []
    assert controller.handle_reply(command('retry')).status == 'ok'
    assert provider.lookup.call_count == 1
    assert [c.args[0] for c in display.present_image.call_args_list] == [picture().data] * 2
    assert len(controller.history.catalogue()) == 1
    assert not json.loads(controller.context())['retry_available']


def test_retry_decode_failure_does_not_switch_to_an_alternative(tmp_path):
    controller, provider, display = workflow(tmp_path, [picture(), picture('red')],
                                             ['unavailable', 'error'])
    controller.handle_reply(command('search', 'Elise'))
    assert controller.handle_reply(command('retry')).status == 'error'
    assert provider.lookup.call_count == 1
    assert [c.args[0] for c in display.present_image.call_args_list] == [picture().data] * 2


def test_expired_display_retry_does_not_start_a_new_search(tmp_path):
    controller, provider, display = workflow(tmp_path, statuses=['unavailable'])
    clock = Mock(return_value=0)
    controller.clock = clock
    controller.handle_reply(command('search', 'Elise'))
    clock.return_value = 1201
    assert controller.handle_reply(command('retry')).status == 'no_results'
    assert provider.lookup.call_count == 1
    assert display.present_image.call_count == 1


def test_saved_recall_works_with_ollama_without_calling_any_image_provider(tmp_path):
    history = ImageHistory(tmp_path)
    saved_id = history.remember(picture(), 'Elise')
    display = Mock()
    display.present_image.return_value = 'loaded'
    provider = UnsupportedImageProvider('ollama')
    provider.lookup = Mock(side_effect=AssertionError('No search should run'))
    controller = ImageWorkflow(provider, display, history=history)
    assert controller.handle_reply(command('recall', saved_id)).status == 'ok'
    provider.lookup.assert_not_called()


def test_archive_write_failure_does_not_turn_a_successful_display_into_a_failure(tmp_path):
    controller, provider, display = workflow(tmp_path)
    controller.history.remember = Mock(side_effect=OSError('Disk full'))
    result = controller.handle_reply(command('search', 'Elise'))
    assert result.status == 'ok' and result.reply.startswith('Here is')
    assert "couldn't save" in result.reply and 'Saved image ID:' not in result.action_result


def test_corrupt_archive_does_not_prevent_startup(tmp_path):
    (tmp_path / 'history.json').write_text('{broken', encoding='utf-8')
    history = ImageHistory(tmp_path)
    assert history.catalogue() == [] and history.recall() is None


def test_display_timeout_logs_browser_progress_and_cleans_up(caplog):
    display = DisplayClient()
    display._s = Mock()
    display._s.post.return_value.json.return_value = {'token': 'test-token'}
    display._s.get.return_value.json.return_value = {
        'status': 'received', 'connected_browsers': 1, 'document_visibility': 'visible'}
    with caplog.at_level('INFO', logger='HAL'), \
            patch('display_client.time.monotonic', side_effect=[0, 0, 9]), \
            patch('display_client.time.sleep'):
        assert display.present_image(picture().data, citations=[]) == 'unavailable'
    assert 'timed out after 8.0s' in caplog.text
    assert "'status': 'received'" in caplog.text and "'connected_browsers': 1" in caplog.text
    assert display._s.post.call_args.args[0].endswith('/api/clear')
