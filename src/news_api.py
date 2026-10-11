"""GNews discovery and bounded article reading, independent of the LLM backend."""
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import re
import time

from dotenv import load_dotenv
import requests

from article_reader import ArticleReadError, read_article
from image_files import web_url

load_dotenv()
BASE_URL = 'https://gnews.io/api/v4'
CACHE_SECONDS = 30 * 60
CATALOG_SECONDS = 60 * 60
MAX_ARTICLES = 20
MAX_SEARCHES = 2
MAX_READS = 3


class NewsService:
    def __init__(self, logger=None, *, api_key=None, reader=read_article, clock=time.monotonic):
        self.logger = logger or logging.getLogger('HAL')
        self.api_key = os.getenv('GNEWS_API_KEY') if api_key is None else api_key
        self.reader = reader
        self.clock = clock
        self.articles = OrderedDict()
        self.begin_turn()

    def begin_turn(self):
        self.searches = 0
        self.reads = 0
        self.attempted = {}
        self._prune()

    def _prune(self):
        for article_id, entry in list(self.articles.items()):
            if self.clock() - entry['seen_at'] >= CATALOG_SECONDS:
                del self.articles[article_id]

    def context(self):
        self._prune()
        items = [entry['metadata'] for entry in list(self.articles.values())[-5:]]
        return json.dumps({'recent_articles': items,
            'note': 'Metadata only. Use news read <id> for article details; dates and retrieval times matter.'},
            ensure_ascii=False) if items else ''

    def dispatch(self, params):
        if params and params[0].lower() == 'read':
            if len(params) != 2:
                return {'status': 'invalid_request', 'message': 'Use news read <id> from a news search result.'}
            return self.read(params[1])
        if params and params[0].lower() == 'search':
            params = params[1:]
        # The original news <topic> and bare news commands remain supported.
        return self.search(' '.join(params))

    def search(self, query='', max_results=5, country='us'):
        if self.searches >= MAX_SEARCHES:
            self.logger.info('News search limit reached: %s searches this turn.', self.searches)
            return {'status': 'limit_reached', 'message': 'No more news searches this turn. Read an existing result or answer from the evidence available.'}
        self.searches += 1
        query = query.strip()
        self.logger.info('News search: query=%r; attempt=%s/%s.', query or '[top headlines]', self.searches, MAX_SEARCHES)
        if not self.api_key:
            return {'status': 'unavailable', 'message': 'GNews API key is not set.'}
        if len(query) > 200:
            return {'status': 'invalid_request', 'message': 'News query is too long. Use a few essential keywords (maximum 200 characters).'}
        params = {'apikey': self.api_key, 'lang': 'en', 'max': min(5, max(1, max_results))}
        if query:
            params.update(q=query, sortby='relevance', **{'in': 'title,description,content'})
        else:
            params['country'] = country
        try:
            response = requests.get(f'{BASE_URL}/{"search" if query else "top-headlines"}',
                                    params=params, timeout=5)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            if not isinstance(articles, list):
                raise ValueError('Invalid articles')
        except (requests.RequestException, ValueError, TypeError, AttributeError) as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            self.logger.info('News search failed: query=%r; HTTP=%s; error=%s.', query, status, type(exc).__name__)
            return {'status': 'unavailable', 'message': 'News search failed. Do not interpret this as no news existing.', 'http_status': status}
        now = self.clock()
        retrieved = datetime.now(timezone.utc).isoformat()
        results = []
        for article in articles[:5]:
            if not isinstance(article, dict) or not web_url(article.get('url')):
                continue
            url = article['url']
            article_id = 'n_' + hashlib.sha256(url.encode()).hexdigest()[:12]
            source = article.get('source') or {}
            if not isinstance(source, dict):
                source = {}
            metadata = {'id': article_id, 'title': str(article.get('title') or 'Untitled')[:300],
                        'source': str(source.get('name') or 'Unknown source')[:120],
                        'url': url, 'published_at': article.get('publishedAt'),
                        'search_retrieved_at': retrieved}
            entry = self.articles.get(article_id, {})
            entry.update(metadata=metadata, seen_at=now)
            self.articles[article_id] = entry
            self.articles.move_to_end(article_id)
            while len(self.articles) > MAX_ARTICLES:
                self.articles.popitem(last=False)
            excerpt = str(article.get('content') or '')
            results.append({**metadata, 'description': str(article.get('description') or '')[:600],
                            'excerpt': excerpt[:1000],
                            'excerpt_truncated': len(excerpt) > 1000 or bool(re.search(r'\[\+?\d+ chars\]', excerpt))})
            self.logger.info('News result: id=%s; source=%s; published=%s; title=%r; URL=%s.',
                             article_id, metadata['source'], metadata['published_at'], metadata['title'], url)
        self.logger.info('News search complete: query=%r; results=%s.', query or '[top headlines]', len(results))
        return {'status': 'ok' if results else 'no_results', 'kind': 'news_search', 'query': query,
                'retrieved_at': retrieved, 'articles': results,
                'message': 'Search results contain excerpts, not verified full articles. For missing details, use news read <id>. GNews free results can be delayed by 12 hours.'}

    def read(self, article_id):
        self._prune()
        entry = self.articles.get(article_id)
        if entry is None:
            self.logger.info('News read rejected: unknown or expired article id=%r.', article_id)
            return {'status': 'invalid_request', 'message': 'Unknown or expired article ID. Select an ID from current news search results; do not invent URLs.'}
        metadata = entry['metadata']
        self.articles.move_to_end(article_id)
        self.logger.info('News read selected: id=%s; URL=%s.', article_id, metadata['url'])
        if entry.get('cached') and self.clock() - entry['cached_at'] < CACHE_SECONDS:
            self.logger.info('News read cache hit: id=%s; retrieved=%s; URL=%s.',
                             article_id, entry['cached']['retrieved_at'], entry['cached']['url'])
            return {**entry['cached'], 'cached': True}
        if article_id in self.attempted:
            return {**self.attempted[article_id], 'already_attempted': True}
        if self.reads >= MAX_READS:
            self.logger.info('News read limit reached: %s article attempts this turn.', self.reads)
            return {'status': 'limit_reached', 'message': 'Article reading limit reached for this turn. Answer from available evidence and explain any missing information.'}
        self.reads += 1
        try:
            article = self.reader(metadata['url'])
        except ArticleReadError as exc:
            result = {**metadata, 'kind': 'news_article', 'status': exc.status, 'message': str(exc)}
            self.logger.info('News read failed: id=%s; status=%s; reason=%s; URL=%s.',
                             article_id, exc.status, exc, metadata['url'])
        else:
            result = {**metadata, **article, 'title': article.get('title') or metadata['title'],
                      'published_at': metadata['published_at'] or article.get('published_at'),
                      'page_date': article.get('published_at'),
                      'kind': 'news_article', 'status': 'ok', 'cached': False}
            entry.update(cached=result, cached_at=self.clock())
            self.logger.info('News read succeeded: id=%s; source=%s; published=%s; retrieved=%s; characters=%s; truncated=%s; URL=%s.',
                             article_id, metadata['source'], result['published_at'], result['retrieved_at'],
                             len(result['text']), result['text_truncated'], result['url'])
        self.attempted[article_id] = result
        return result
