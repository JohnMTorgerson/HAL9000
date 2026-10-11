"""News discovery, actual HTML extraction, failure recovery and HAL integration."""
import json
import logging
import socket
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

import article_reader
from article_reader import ArticleReadError, extract_article, read_article, MAX_HTML_BYTES, MAX_TEXT_CHARS
from news_api import NewsService, CACHE_SECONDS, CATALOG_SECONDS
from followup import FollowupDecision
from llm_client import LLMClient
import test_followup as followup_tests
import test_hal_recovery as hal_tests

HTML = '''<!doctype html><html lang="en"><head>
<title>Harbor trains add evening service</title>
<meta property="article:published_time" content="2026-10-11T09:00:00Z">
</head><body><nav>Unrelated navigation coupon</nav><article>
<h1>Harbor trains add evening service</h1>
<p>The transit board approved an expanded evening schedule at its meeting on Friday.
Passengers travelling home after concerts will have more departures to choose from.
The board said the new schedule will begin next month after operator training is complete.</p>
<p>Service planners studied passenger counts from the past twelve months and spoke with
neighborhood groups about gaps in the current timetable. Weekend departures remain unchanged.</p>
<table><tr><th>Route</th><th>Last departure</th></tr>
<tr><td>Harbor line</td><td>11:45 PM</td></tr></table>
<p>The final approved budget includes twelve additional evening departures each week.
Tickets will continue to cost four dollars, and existing passes remain valid.</p>
</article><footer>Unrelated subscription coupon</footer></body></html>'''


def search_articles(count=5):
    return [{'title': f'Transit news {i}', 'description': 'A report about evening trains.',
             'content': 'The board approved a schedule... [1200 chars]',
             'url': f'https://publisher.example/story/{i}',
             'publishedAt': '2026-10-11T09:00:00Z', 'source': {'name': 'Fixture Gazette'}}
            for i in range(count)]


def article(url):
    return {**extract_article(HTML, url), 'url': url, 'retrieved_at': '2026-10-11T10:00:00+00:00'}


@pytest.fixture
def search_http():
    response = Mock()
    response.json.return_value = {'articles': search_articles()}
    with patch('news_api.requests.get', return_value=response) as get:
        yield get, response


def test_search_returns_readable_ids_dates_urls_and_searches_content(search_http, caplog):
    get, _ = search_http
    service = NewsService(api_key='fixture-secret')
    with caplog.at_level(logging.INFO, logger='HAL'):
        result = service.dispatch(['search', 'transit', 'schedule'])
    assert len(result['articles']) == 5
    first = result['articles'][0]
    assert first['published_at'] == '2026-10-11T09:00:00Z'
    assert first['excerpt_truncated']
    assert first['id'] in service.context()
    assert 'excerpt' not in service.context()  # Compact catalogue, not article text.
    params = get.call_args.kwargs['params']
    assert params['q'] == 'transit schedule'
    assert params['in'] == 'title,description,content'
    assert params['sortby'] == 'relevance'
    assert params['apikey'] == 'fixture-secret'
    assert first['url'] in caplog.text and first['id'] in caplog.text
    assert 'fixture-secret' not in caplog.text


def test_legacy_queries_headlines_budget_and_no_results_are_distinct(search_http):
    get, response = search_http
    service = NewsService(api_key='fixture-secret')
    assert service.dispatch([])['status'] == 'ok'
    assert get.call_args.args[0].endswith('/top-headlines')
    response.json.return_value = {'articles': []}
    assert service.dispatch(['transit', 'strike'])['status'] == 'no_results'
    assert get.call_args.kwargs['params']['q'] == 'transit strike'
    assert service.dispatch(['search', 'another'])['status'] == 'limit_reached'
    assert get.call_count == 2
    service.begin_turn()
    service.search('another')
    assert get.call_count == 3


def test_service_errors_do_not_expose_api_key_or_claim_no_results(search_http, caplog):
    get, _ = search_http
    get.side_effect = requests.HTTPError('URL with apikey=fixture-secret', response=Mock(status_code=403))
    service = NewsService(api_key='fixture-secret')
    with caplog.at_level(logging.INFO, logger='HAL'):
        result = service.search('trains')
    assert result['status'] == 'unavailable'
    assert result['http_status'] == 403
    assert 'fixture-secret' not in json.dumps(result) + caplog.text


def test_reads_are_limited_retry_different_sources_and_reset_each_turn(search_http, caplog):
    reader = Mock(side_effect=ArticleReadError('blocked', 'Publisher requires payment.'))
    service = NewsService(api_key='fixture', reader=reader)
    ids = [a['id'] for a in service.search('trains')['articles']]
    with caplog.at_level(logging.INFO, logger='HAL'):
        assert service.read(ids[0])['status'] == 'blocked'
        assert service.read(ids[0])['already_attempted']
        assert service.read(ids[1])['status'] == 'blocked'
        assert service.read(ids[2])['status'] == 'blocked'
        assert service.read(ids[3])['status'] == 'limit_reached'
    assert reader.call_count == 3
    assert 'Publisher requires payment' in caplog.text
    assert search_articles()[0]['url'] in caplog.text
    service.begin_turn()
    reader.side_effect = article
    assert service.read(ids[3])['status'] == 'ok'


def test_cache_reuses_article_for_followups_then_refetches_and_expires(search_http, caplog):
    now = [0]
    reader = Mock(side_effect=article)
    service = NewsService(api_key='fixture', reader=reader, clock=lambda: now[0])
    article_id = service.search('trains')['articles'][0]['id']
    first = service.read(article_id)
    service.begin_turn()
    with caplog.at_level(logging.INFO, logger='HAL'):
        cached = service.read(article_id)
    assert cached['cached'] and cached['retrieved_at'] == first['retrieved_at']
    assert cached['text'] == first['text']
    assert 'News read cache hit' in caplog.text
    reader.assert_called_once()
    now[0] = CACHE_SECONDS + 1
    service.begin_turn()
    assert not service.read(article_id)['cached']
    assert reader.call_count == 2
    now[0] = CATALOG_SECONDS + 1
    assert service.read(article_id)['status'] == 'invalid_request'
    assert service.context() == ''


def test_unknown_ids_and_arbitrary_urls_cannot_trigger_downloads(search_http):
    reader = Mock()
    service = NewsService(api_key='fixture', reader=reader)
    service.search('trains')
    for value in ('1', 'n_invented', 'https://publisher.example/other', 'http://127.0.0.1:8000'):
        assert service.dispatch(['read', value])['status'] == 'invalid_request'
    reader.assert_not_called()


def test_real_extractor_retains_article_tables_and_removes_page_clutter():
    result = extract_article(HTML, 'https://publisher.example/story')
    assert '11:45 PM' in result['text']
    assert 'twelve additional evening departures' in result['text']
    assert 'Unrelated' not in result['text']
    assert result['published_at'] == '2026-10-11'
    assert not result['text_truncated']
    assert result['completeness'] == 'extracted_not_independently_verified'


@pytest.mark.parametrize('html', [
    '<html><head><title>Site Unavailable</title></head><body>Please return later.</body></html>',
    '<html><article><h1>Access denied</h1><p>Please enable JavaScript and cookies.</p></article></html>',
    HTML.replace('</head>', '<script type="application/ld+json">{"@type":"NewsArticle","isAccessibleForFree":false}</script></head>'),
])
def test_unavailable_and_paywalled_pages_are_not_returned_as_articles(html):
    with pytest.raises(ArticleReadError) as exc:
        extract_article(html, 'https://publisher.example/story')
    assert exc.value.status in ('blocked', 'unreadable')


def test_long_article_has_explicit_truncation_flag():
    long = HTML.replace('</article>', '<p>' + ('This additional passage describes service changes in detail. ' * 500) + '</p></article>')
    result = extract_article(long, 'https://publisher.example/story')
    assert len(result['text']) == MAX_TEXT_CHARS
    assert result['text_truncated']
    assert result['completeness'] == 'truncated_by_hal'


def http_response(status=200, *, body=HTML.encode(), headers=None):
    response = MagicMock()
    response.__enter__.return_value = response
    response.status_code = status
    response.is_redirect = 300 <= status < 400
    response.headers = {'Content-Type': 'text/html; charset=utf-8', **(headers or {})}
    response.iter_content.return_value = [body]
    if status >= 400:
        response.raise_for_status.side_effect = requests.HTTPError('fixture')
    return response


@pytest.fixture
def article_http():
    session = MagicMock()
    session.__enter__.return_value = session
    session.get.return_value = http_response()
    with patch('article_reader.requests.Session', return_value=session), \
         patch('image_files.socket.getaddrinfo', return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))]):
        yield session


def test_public_download_preserves_final_url_and_uses_actual_extraction(article_http):
    article_http.get.side_effect = [http_response(302, headers={'Location': '/article'}), http_response()]
    result = read_article('https://publisher.example/link')
    assert result['url'] == 'https://publisher.example/article'
    assert '11:45 PM' in result['text']
    assert result['retrieved_at']
    assert article_http.trust_env is False
    assert article_http.get.call_args.kwargs['allow_redirects'] is False


@pytest.mark.parametrize('status,expected', [(401, 'blocked'), (402, 'blocked'), (403, 'blocked'),
                                           (429, 'rate_limited'), (500, 'unavailable')])
def test_http_failures_are_read_outcomes_not_crashes(article_http, status, expected):
    article_http.get.return_value = http_response(status)
    with pytest.raises(ArticleReadError) as exc:
        read_article('https://publisher.example/story')
    assert exc.value.status == expected
    article_http.get.assert_called_once()


@pytest.mark.parametrize('headers,body,expected', [
    ({'Content-Type': 'application/pdf'}, b'PDF', 'unreadable'),
    ({'Content-Length': str(MAX_HTML_BYTES + 1)}, b'', 'too_large'),
    ({}, b'x' * (MAX_HTML_BYTES + 1), 'too_large'),
])
def test_article_type_and_size_are_bounded(article_http, headers, body, expected):
    article_http.get.return_value = http_response(headers=headers, body=body)
    with pytest.raises(ArticleReadError) as exc:
        read_article('https://publisher.example/story')
    assert exc.value.status == expected


def test_redirect_to_private_address_is_rejected_before_request(article_http):
    article_http.get.return_value = http_response(302, headers={'Location': 'http://127.0.0.1/private'})
    answers = [[(socket.AF_INET, socket.SOCK_STREAM, 6, '', (ip, 443))]
               for ip in ('93.184.216.34', '127.0.0.1')]
    with patch('image_files.socket.getaddrinfo', side_effect=answers), pytest.raises(ArticleReadError) as exc:
        read_article('https://publisher.example/story')
    assert exc.value.status == 'invalid_url'
    article_http.get.assert_called_once()


def test_dns_failure_is_distinct_from_a_rejected_private_url(article_http):
    with patch('image_files.socket.getaddrinfo', side_effect=socket.gaierror('fixture')), \
         pytest.raises(ArticleReadError) as exc:
        read_article('https://publisher.example/story')
    assert exc.value.status == 'unavailable'
    assert 'hostname could not be resolved' in str(exc.value)
    article_http.get.assert_not_called()


def test_missing_optional_dependency_reports_installation_without_network(article_http):
    with patch.dict(sys.modules, {'trafilatura.utils': None}), pytest.raises(ArticleReadError) as exc:
        read_article('https://publisher.example/story')
    assert 'requirements-news.txt' in str(exc.value)
    article_http.get.assert_not_called()


def test_main_loop_search_reads_alternative_and_reuses_cache_on_followup(search_http):
    ns, run, _ = followup_tests.MainLoopTests().fixture()
    reader = Mock(side_effect=[ArticleReadError('blocked', 'Publisher requires payment.'),
                               {**article(search_articles()[1]['url']), 'text': 'Context. ' * 200 + 'The last train leaves at 11:45 PM.'}])
    news = ns['news'] = NewsService(api_key='fixture', reader=reader)
    ns['handle_api_call'] = hal_tests.load_hal_function('handle_api_call', ns)
    texts = ['Tell me more about the transit story.', 'What was the last departure again?']
    ns['stt'].transcribe.side_effect = texts
    responses = []
    def respond(text):
        responses.append(text)
        if len(responses) == 1:
            return '[EXTERNAL_API_CALL] news search transit service'
        if len(responses) == 2:
            ids = list(news.articles)
            ns['llm'].get_followup_response.return_value = FollowupDecision('respond', f'[EXTERNAL_API_CALL] news read {ids[1]}')
            return f'[EXTERNAL_API_CALL] news read {ids[0]}'
        if len(responses) == 3:
            assert 'Publisher requires payment.' in text
            return f'[EXTERNAL_API_CALL] news read {list(news.articles)[0]}'
        assert 'The last train leaves at 11:45 PM.' in text  # Well beyond the old 800-character cutoff.
        assert 'TRUNCATED (for logging only)' not in text
        assert 'Original request:' in text
        return 'The last train leaves at 11:45 PM, according to the Fixture Gazette.'
    ns['llm'].get_response.side_effect = respond
    reads = []
    def capture(on_trigger, **options):
        reads.append(options)
        if len(reads) == 3:
            assert 'followup_deadline' in options
            raise KeyboardInterrupt
        on_trigger('spacebar' if len(reads) == 1 else 'followup')
        return [.1], 16000
    ns['voice_input'].read_command.side_effect = capture
    with pytest.raises(SystemExit):
        run()
    assert reader.call_count == 2  # Blocked first source, successful second; followup reads cache.
    assert search_http[0].call_count == 1
    assert ns['llm'].finish_turn.call_count == 2
    assert ns['voice'].synthesize_wav.call_count == 2
    assert len(responses) == 5
    assert 'Fixture Gazette' in ns['llm'].news_context
    previews = [call.args[1] for call in ns['logger'].info.call_args_list if call.args[0] == 'Enriched prompt for HAL: %s']
    assert any('TRUNCATED (for logging only)' in preview for preview in previews)


def test_repeated_tool_requests_end_without_speaking_commands():
    ns, run, _ = followup_tests.MainLoopTests().fixture()
    ns['llm'].get_response.return_value = '[EXTERNAL_API_CALL] news search unavailable'
    ns['voice_input'].read_command.side_effect = [([.1], 16000), KeyboardInterrupt]
    with pytest.raises(SystemExit):
        run()
    assert ns['handle_api_call'].call_count == 8
    assert ns['llm'].get_response.call_count == 10
    reply = ns['voice'].synthesize_wav.call_args.args[0]
    assert 'could not complete' in reply and '[EXTERNAL_API_CALL]' not in reply


def test_ollama_receives_catalogue_without_openai_dependency():
    client = LLMClient('ollama', 'fixture')
    client.news_context = '{"recent_articles":[{"id":"n_fixture","title":"Transit story"}]}'
    with patch('llm_client.subprocess.run', return_value=Mock(stdout='Ready.')) as run:
        assert client.get_response('Tell me more about that story.') == 'Ready.'
    prompt = run.call_args.args[0][-1]
    assert 'n_fixture' in prompt and 'untrusted source metadata' in prompt


def test_openai_receives_the_same_catalogue_including_on_followups():
    helper = followup_tests.DecisionTests()
    client, requests = helper.fixture(lambda _: followup_tests.completion(
        {'decision': 'respond', 'reply': '[EXTERNAL_API_CALL] news read n_fixture'}))
    try:
        client.news_context = '{"recent_articles":[{"id":"n_fixture","title":"Transit story"}]}'
        reply = client.get_followup_response('Tell me more about that story.')
        assert reply.reply == '[EXTERNAL_API_CALL] news read n_fixture'
        assert 'n_fixture' in requests[0]['messages'][0]['content']
        assert 'untrusted source metadata' in requests[0]['messages'][0]['content']
    finally:
        helper.doCleanups()
