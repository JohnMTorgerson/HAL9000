"""Image search/vision contracts and voice integration, without paid API calls."""
import base64
import io
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from openai import OpenAI
from PIL import Image
import pytest

from image_files import normalize_image, public_url, web_url, download_image
from image_lookup import (ImageCandidate, ImageProviderError, ImageResult, ImageWorkflow,
                          UnsupportedImageProvider, make_image_provider, parse_image_request,
                          repair_image_reply)
from image_provider_openai import OpenAIImageProvider
from llm_client import LLMClient
from followup import FollowupDecision
from test_song_request import loop_fixture

COMMAND = '[IMAGE_REQUEST] {"action":"search","query":"the latest example phone"}'


def jpeg(color='blue'):
    out = io.BytesIO()
    Image.new('RGB', (300, 200), color).save(out, format='JPEG')
    return out.getvalue()


def image(number='1'):
    return ImageCandidate(number, f'https://images.example/{number}.jpg',
                          f'https://example.com/product/{number}', data=jpeg(),
                          reply=f'Here is picture {number}.', description=f'Phone view {number}.')


def response_body(answer, *, results=None, refusal=None, status='completed', reason=None):
    contents = ([{'type': 'refusal', 'refusal': refusal}] if refusal is not None else
                [{'type': 'output_text', 'text': json.dumps(answer), 'annotations': []}])
    output = []
    if results is not None:
        output.append({'type': 'web_search_call', 'id': 'ws-fixture', 'status': 'completed',
                       'action': {'type': 'search', 'query': 'fixture'}, 'results': results})
    output.append({'id': 'msg-fixture', 'type': 'message', 'role': 'assistant',
                   'status': 'completed', 'content': contents})
    return {'id': 'resp-fixture', 'object': 'response', 'created_at': 1, 'model': 'gpt-6-luna',
            'status': status, 'error': None, 'incomplete_details': {'reason': reason} if reason else None,
            'output': output}


def search_body():
    return response_body({'status': 'ok', 'subject': 'Example Phone 2',
                          'evidence': 'Official release page identifies Phone 2.', 'reply': '',
                          'preferred_image_urls': []},
        results=[{'type': 'image_result', 'image_url': f'https://images.example/{n}.jpg',
                  'source_website_url': f'https://example.com/phone2/{n}', 'caption': 'Phone 2'}
                 for n in ('1', '2')])


def vision_body():
    return response_body({'status': 'ok', 'reply': '', 'rejected_images': [], 'images': [
        {'id': n, 'description': f'Phone 2 view {n}.', 'spoken_reply': f'Here is view {n}.'}
        for n in ('2', '1')]})


@pytest.fixture
def provider_factory():
    clients = []
    def make(bodies, download=None):
        requests, queued = [], iter(bodies)
        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=next(queued))
        http = httpx.Client(transport=httpx.MockTransport(handler))
        client = OpenAI(api_key='fixture-not-a-key', http_client=http, max_retries=0)
        clients.append(client)
        download = download or (lambda url: jpeg('red' if '/1.jpg' in url else 'blue'))
        return OpenAIImageProvider(client, 'gpt-6-luna', 'fast', download=download), requests
    yield make
    for client in clients:
        client.close()


def test_real_pinned_sdk_serializes_search_and_batch_vision_in_two_requests(provider_factory):
    provider, calls = provider_factory([search_body(), vision_body()])
    result = provider.lookup('What does the new phone look like?')
    assert result.status == 'ok'
    assert [i.id for i in result.images] == ['2', '1']
    assert result.images[0].source_url == 'https://example.com/phone2/2'
    assert len(calls) == 2
    assert calls[0]['tools'][0]['search_content_types'] == ['image', 'text']
    assert calls[0]['include'] == ['web_search_call.results']
    assert calls[0]['tool_choice'] == 'required' and calls[0]['max_tool_calls'] == 3
    assert calls[0]['store'] is False
    assert all(c['service_tier'] == 'priority' and c['reasoning']['effort'] == 'low' for c in calls)
    inputs = calls[1]['input'][0]['content']
    pictures = [c for c in inputs if c['type'] == 'input_image']
    assert len(pictures) == 2
    assert base64.b64decode(pictures[0]['image_url'].split(',')[1]) == jpeg('red')
    assert 'tools' not in calls[1]
    assert json.loads(inputs[0]['text'])['date'] == json.loads(calls[0]['input'])['date']


@pytest.mark.parametrize('stage', ['search', 'verify'])
@pytest.mark.parametrize('kind', ['native', 'schema', 'filter'])
def test_refusal_at_each_stage_stops_without_retry_or_selection(provider_factory, stage, kind):
    if kind == 'native':
        refused = response_body({}, refusal="I can't provide those images.")
    elif kind == 'schema':
        refused = response_body({'status': 'refused', 'reply': 'I cannot fulfill that request.'})
    else:
        refused = response_body({}, status='incomplete', reason='content_filter')
    provider, calls = provider_factory(([search_body()] if stage == 'verify' else []) + [refused])
    with pytest.raises(ImageProviderError) as err:
        provider.lookup('fixture')
    assert err.value.status == 'refused'
    assert len(calls) == (2 if stage == 'verify' else 1)


def test_unknown_refusal_reason_is_not_fabricated(provider_factory):
    provider, _ = provider_factory([response_body({}, refusal='')])
    with pytest.raises(ImageProviderError, match='without giving a specific reason'):
        provider.lookup('fixture')


def test_missing_raw_image_results_does_not_trust_model_written_urls(provider_factory):
    body = search_body()
    body['output'] = body['output'][1:]
    parsed = json.loads(body['output'][0]['content'][0]['text'])
    parsed['image_url'] = 'https://invented.example/photo.jpg'
    body['output'][0]['content'][0]['text'] = json.dumps(parsed)
    download = Mock()
    provider, calls = provider_factory([body], download)
    assert provider.lookup('fixture').status == 'error'
    assert len(calls) == 1
    download.assert_not_called()


def test_unrecognized_selected_id_never_reaches_display(provider_factory):
    bad = response_body({'status': 'ok', 'reply': '', 'images': [
        {'id': 'invented', 'description': 'unverified', 'spoken_reply': 'Here it is.'}]})
    provider, _ = provider_factory([search_body(), bad])
    with pytest.raises(ImageProviderError, match='invalid selection'):
        provider.lookup('fixture')


def test_duplicate_downloads_compared_once(provider_factory):
    selected = response_body({'status': 'ok', 'reply': '', 'images': [
        {'id': '1', 'description': 'Phone.', 'spoken_reply': 'Here it is.'}]})
    provider, calls = provider_factory([search_body(), selected], lambda url: jpeg())
    assert len(provider.lookup('fixture').images) == 1
    assert len([x for x in calls[1]['input'][0]['content'] if x['type'] == 'input_image']) == 1


@pytest.mark.parametrize('ranked', [False, True])
def test_refined_search_candidates_are_not_crowded_out_by_early_results(provider_factory, ranked):
    body = search_body()
    # Enough early results to fill the old download budget before the targeted call.
    early = body['output'][0]['results']
    early.extend(dict(early[0], image_url=f'https://images.example/old-{n}.jpg') for n in range(5))
    refined = {'type': 'image_result', 'image_url': 'https://images.example/right.jpg',
               'source_website_url': 'https://example.com/current', 'caption': 'Current model'}
    body['output'].insert(1, {'type': 'web_search_call', 'id': 'ws-refined',
                             'status': 'completed', 'results': [refined]})
    if ranked:
        answer = json.loads(body['output'][-1]['content'][0]['text'])
        answer['preferred_image_urls'] = ['https://invented.example/fake.jpg', refined['image_url']]
        body['output'][-1]['content'][0]['text'] = json.dumps(answer)
    chosen = response_body({'status': 'ok', 'reply': '', 'images': [
        {'id': '1', 'description': 'Current model.', 'spoken_reply': 'Here it is.'}]})
    downloader = Mock(return_value=jpeg())
    provider, calls = provider_factory([body, chosen], downloader)
    result = provider.lookup('new phone')
    assert result.images[0].image_url == refined['image_url']
    assert downloader.call_args_list[0].args[0] == refined['image_url']
    assert len(downloader.call_args_list) == 5
    assert all('invented.example' not in call.args[0] for call in downloader.call_args_list)
    assert 'current' in calls[1]['input'][0]['content'][1]['text']


def test_grounded_preference_changes_the_vision_candidate_order(provider_factory):
    body = search_body()
    answer = json.loads(body['output'][-1]['content'][0]['text'])
    answer['preferred_image_urls'] = ['https://images.example/2.jpg']
    body['output'][-1]['content'][0]['text'] = json.dumps(answer)
    provider, calls = provider_factory([body, vision_body()])
    assert provider.lookup('current phone').status == 'ok'
    contents = calls[1]['input'][0]['content']
    assert json.loads(contents[1]['text'])['id'] == '2'
    assert base64.b64decode(contents[2]['image_url'].split(',')[1]) == jpeg('blue')


def test_verification_failure_gets_one_targeted_recovery_with_reasons_logged(provider_factory, caplog):
    failure = {'status': 'no_results', 'reply': 'These pictures show an older model.',
               'images': [], 'rejected_images': [{'id': '1', 'reason': 'Older model.'},
                                                {'id': '2', 'reason': 'Accessory only.'}]}
    provider, calls = provider_factory([search_body(), response_body(failure),
                                       search_body(), vision_body()])
    with caplog.at_level('INFO', logger='HAL'):
        result = provider.lookup('the new iPhone')
    assert result.status == 'ok' and len(calls) == 4
    retry = json.loads(calls[2]['input'])
    assert retry['request'] == 'the new iPhone'
    assert retry['previous_attempt']['stage'] == 'verify'
    assert retry['previous_attempt']['reason'] == failure['reply']
    assert retry['previous_attempt']['checked_images'][0]['id'] == '1'
    assert retry['previous_attempt']['rejected_images'] == failure['rejected_images']
    for expected in ('Older model.', 'Accessory only.', 'Image verification inputs',
                     'https://images.example/1.jpg', 'Official release page', 'Image lookup recovery'):
        assert expected in caplog.text


def test_search_no_results_recovery_is_bounded_and_preserves_specific_explanation(provider_factory):
    failure = response_body({'status': 'no_results', 'subject': 'Example Phone',
                             'evidence': 'Only older photos found.',
                             'reply': 'I found older models, but no confirmed picture of that model.'})
    provider, calls = provider_factory([failure, failure])
    result = provider.lookup('an iPhone 18')
    assert result.status == 'no_results' and len(calls) == 2
    assert result.reply == 'I found older models, but no confirmed picture of that model.'
    assert json.loads(calls[1]['input'])['previous_attempt']['stage'] == 'search'


def test_refusal_during_recovery_stops_immediately(provider_factory):
    provider, calls = provider_factory([
        response_body({'status': 'no_results', 'reply': 'No useful results.'}),
        response_body({}, refusal='The provider declined this request.')])
    with pytest.raises(ImageProviderError) as err:
        provider.lookup('fixture')
    assert err.value.status == 'refused' and len(calls) == 2


@pytest.mark.parametrize('user_input, reply', [
    ('Hey Hal, can you show me that Lotus Elise again?', 'Certainly. Here it is again.'),
    ('Hey Hal, show me a picture of an X-29.',
     'Here it is, the experimental Grumman X-29 with its distinctive forward-swept wings.'),
    ('Could you please show me a photo of a blue bird?', 'Here you go.'),
])
def test_false_image_success_is_repaired_to_an_actual_command(user_input, reply):
    repaired = parse_image_request(repair_image_reply(user_input, reply))
    assert repaired['action'] == 'search'
    assert any(subject in repaired['query'] for subject in ('Lotus Elise', 'X-29', 'blue bird'))


@pytest.mark.parametrize('user_input, reply', [
    ('Do not show me a picture of a bird.', 'Here it is.'),
    ('Explain the phrase "show me a picture of a bird".', 'Here it is.'),
    ('Show me a picture of a bird.', 'Which bird would you like to see?'),
    ('Show me a picture of a bird.', "I can't provide that image."),
    ('Show me that calendar again.', 'Here it is.'),
    ('Show me a picture of a bird.', '[EXTERNAL_API_CALL] wikipedia search bird'),
    ('Show me a picture of a bird.', COMMAND),
    ('What is an X-29?', 'Here it is, the answer.'),
])
def test_repair_leaves_other_intents_clarifications_and_refusals_alone(user_input, reply):
    assert repair_image_reply(user_input, reply) == reply


@pytest.mark.parametrize('failure', ['download', 'no_results', 'verify_no_results', 'truncated'])
def test_ordinary_failures_are_not_reported_as_content_restrictions(provider_factory, failure):
    bodies, downloader = [search_body()], None
    if failure == 'download':
        downloader = Mock(side_effect=ValueError('fixture broken image'))
    elif failure == 'no_results':
        bodies = [response_body({'status': 'no_results', 'reply': ''})] * 2
    elif failure == 'verify_no_results':
        bodies += [response_body({'status': 'no_results', 'reply': '', 'images': []})]
        bodies *= 2
    else:
        bodies = [response_body({}, status='incomplete', reason='max_output_tokens')]
    provider, _ = provider_factory(bodies, downloader)
    try:
        result = provider.lookup('fixture')
        assert result.status in ('error', 'no_results')
    except ImageProviderError as exc:
        assert exc.status == 'error'


@pytest.mark.parametrize('raw', ['{}', 'null', '{"action":"play","query":""}',
                                '{"action":"search","query":""}',
                                '{"action":"next","query":"phone"}',
                                '{"action":"search","query":"phone","url":"x"}'])
def test_malformed_commands_never_become_spoken_control_text(raw):
    with pytest.raises(ValueError):
        parse_image_request('[IMAGE_REQUEST] ' + raw)
    workflow = ImageWorkflow(UnsupportedImageProvider('ollama'), Mock())
    result = workflow.handle_reply('[IMAGE_REQUEST] ' + raw)
    assert result.status == 'error' and '[IMAGE_REQUEST]' not in result.reply


def test_unsupported_backend_never_constructs_openai_provider_or_calls_display():
    llm = SimpleNamespace(backend='ollama')
    with patch('image_provider_openai.OpenAIImageProvider') as constructor:
        provider = make_image_provider(llm)
        constructor.assert_not_called()
    display, acknowledge = Mock(), Mock()
    workflow = ImageWorkflow(provider, display)
    result = workflow.handle_reply(COMMAND, on_search=acknowledge)
    assert result.status == 'unsupported' and 'Ollama' in result.reply
    acknowledge.assert_not_called()
    display.present_image.assert_not_called()


def test_show_next_previous_close_use_verified_cache_and_record_actual_image():
    provider, display, clock = Mock(supported=True), Mock(), Mock(return_value=0)
    provider.lookup.return_value = ImageResult('ok', images=[image('1'), image('2')])
    display.present_image.return_value = 'loaded'
    workflow = ImageWorkflow(provider, display, clock=clock)
    first = workflow.handle_reply(COMMAND)
    assert first.reply == 'Here is picture 1.'
    assert 'Showed image 1 of 2' in first.action_result
    assert display.present_image.call_args.kwargs['citations'][0]['url'] == image('1').source_url
    assert workflow.handle_reply('[IMAGE_REQUEST] {"action":"next","query":""}').reply == 'Here is picture 2.'
    assert workflow.handle_reply('[IMAGE_REQUEST] {"action":"previous","query":""}').reply == 'Here is picture 1.'
    assert provider.lookup.call_count == 1
    clock.return_value = 121
    assert json.loads(workflow.context())['display_timeout_elapsed']
    assert workflow.handle_reply('[IMAGE_REQUEST] {"action":"close","query":""}').status == 'closed'
    display.clear.assert_called_once_with(key='image-lookup')
    assert json.loads(workflow.context())['cached_images'] == 0


def test_expired_cache_does_not_silently_start_another_search():
    provider, display, clock = Mock(supported=True), Mock(), Mock(return_value=0)
    provider.lookup.return_value = ImageResult('ok', images=[image()])
    display.present_image.return_value = 'loaded'
    workflow = ImageWorkflow(provider, display, clock=clock)
    workflow.handle_reply(COMMAND)
    clock.return_value = 1201
    assert workflow.handle_reply('[IMAGE_REQUEST] {"action":"next","query":""}').status == 'no_results'
    assert provider.lookup.call_count == 1


def test_failed_image_tries_checked_alternative_and_uses_its_own_description():
    provider, display = Mock(supported=True), Mock()
    provider.lookup.return_value = ImageResult('ok', images=[image('1'), image('2')])
    display.present_image.side_effect = ['error', 'loaded']
    result = ImageWorkflow(provider, display).handle_reply(COMMAND)
    assert result.reply == 'Here is picture 2.'
    assert 'Phone view 2.' in result.action_result


@pytest.mark.parametrize('status', ['hidden', 'unavailable'])
def test_display_failure_does_not_claim_success_or_repeat_wait_for_every_image(status):
    provider, display = Mock(supported=True), Mock()
    provider.lookup.return_value = ImageResult('ok', images=[image('1'), image('2')])
    display.present_image.return_value = status
    workflow = ImageWorkflow(provider, display)
    result = workflow.handle_reply(COMMAND)
    assert result.status == 'error' and "couldn't show" in result.reply
    assert 'Showed image' not in result.action_result
    assert display.present_image.call_count == 1


@pytest.mark.parametrize('followup', [False, True])
@pytest.mark.parametrize('routed_reply', [COMMAND, 'Here it is, the Grumman X-29.'])
def test_real_voice_loop_speaks_image_result_without_fourth_llm_request(followup, routed_reply):
    ns, run, clock = loop_fixture()
    image_query = 'Hey Hal, show me a picture of an X-29.'
    ns['stt'].transcribe.side_effect = ['Hello.', image_query] if followup else [image_query]
    ns['llm'].get_response.return_value = 'Hello.' if followup else routed_reply
    ns['llm'].get_followup_response.return_value = FollowupDecision('respond', routed_reply)
    provider, display = Mock(supported=True), Mock()
    provider.lookup.return_value = ImageResult('ok', images=[image()])
    display.present_image.return_value = 'loaded'
    ns['images'] = ImageWorkflow(provider, display, logger=ns['logger'])
    count = 0
    def capture(on_trigger, **kwargs):
        nonlocal count
        count += 1
        if count > (2 if followup else 1):
            raise KeyboardInterrupt
        on_trigger('followup' if count == 2 else 'wakeword')
        return [.1], 16000
    ns['voice_input'].read_command.side_effect = capture
    with pytest.raises(SystemExit):
        run()
    assert ns['llm'].get_response.call_count == 1
    assert ns['llm'].get_followup_response.call_count == int(followup)
    ns['handle_api_call'].assert_not_called()
    assert ns['voice'].synthesize_wav.call_args.args[0] == image().reply
    assert 'Showed image' in ns['llm'].finish_turn.call_args.kwargs['action_result']
    assert [c.kwargs['label'] for c in ns['play_audio'].call_args_list] == (
        ['reply', 'acknowledgment', 'reply'] if followup else ['acknowledgment', 'reply'])


@pytest.mark.parametrize('followup', [False, True])
@pytest.mark.parametrize('http_block', [False, True])
def test_initial_model_refusal_is_spoken_and_kept_in_history(followup, http_block):
    body = {'id': 'chat-fixture', 'object': 'chat.completion', 'created': 1,
            'model': 'fixture', 'choices': [{'index': 0, 'finish_reason': 'stop',
            'message': {'role': 'assistant', 'content': None, 'refusal': "I can't provide that image."}}]}
    if http_block:
        body = {'error': {'message': 'raw provider error', 'type': 'invalid_request_error',
                          'code': 'content_policy_violation'}}
    http = httpx.Client(transport=httpx.MockTransport(lambda request:
        httpx.Response(400 if http_block else 200, json=body)))
    client = LLMClient('ollama', 'fixture')
    client.backend, client.service_tier = 'openai', None
    client.client = OpenAI(api_key='fixture-not-a-key', http_client=http)
    try:
        answer = client.get_followup_response('fixture').reply if followup else client.get_response('fixture')
        assert client.last_refusal
        assert answer == ("The provider's content filter blocked that request." if http_block
                          else "I can't provide that image.")
        assert client.chat_history[-1]['content'] == answer
    finally:
        client.client.close()


def test_decoder_rejects_html_and_bounds_large_images():
    with pytest.raises(Exception):
        normalize_image(b'<html>not a picture</html>')
    raw = io.BytesIO()
    Image.new('RGB', (2400, 1800)).save(raw, format='PNG')
    converted = Image.open(io.BytesIO(normalize_image(raw.getvalue())))
    assert converted.size == (1600, 1200) and converted.format == 'JPEG'


@pytest.mark.parametrize('url', ['file:///etc/passwd', 'javascript:alert(1)',
                               'https://user:secret@example.com/a.jpg', 'http://example.com:8000/a'])
def test_unsafe_url_schemes_and_credentials_are_not_accepted(url):
    assert not web_url(url)


def test_private_download_targets_are_rejected():
    with patch('image_files.socket.getaddrinfo', return_value=[(2, 1, 6, '', ('127.0.0.1', 80))]):
        with pytest.raises(ValueError, match='public internet'):
            public_url('http://example.com/image.jpg')


def test_native_refusal_in_real_voice_loop_never_triggers_wikipedia_or_another_action():
    ns, run, clock = loop_fixture()
    ns['llm'].get_response.return_value = "I'm sorry, I can't do that."
    ns['llm'].last_refusal = True
    ns['looks_factual'] = Mock(return_value=True)
    ns['extract_named_entities'] = Mock(return_value=['fixture person'])
    def capture(on_trigger, **kwargs):
        if ns['voice_input'].read_command.call_count > 1:
            raise KeyboardInterrupt
        on_trigger('wakeword')
        return [.1], 16000
    ns['voice_input'].read_command.side_effect = capture
    with pytest.raises(SystemExit):
        run()
    ns['handle_api_call'].assert_not_called()
    ns['images'].handle_reply.assert_not_called()
    ns['extract_named_entities'].assert_not_called()
    assert ns['voice'].synthesize_wav.call_args.args[0] == ns['llm'].get_response.return_value
