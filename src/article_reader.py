"""Read public article HTML without a browser, login, or paywall bypass."""
import json
import re
import time
from datetime import datetime, timezone
from urllib.parse import urljoin

import requests

from image_files import public_url

MAX_HTML_BYTES = 2 * 1024 * 1024
MAX_TEXT_CHARS = 20_000


class ArticleReadError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def extract_article(html, url):
    # Optional dependency: HAL and news search still start without the reader.
    try:
        from trafilatura import extract
    except ImportError:
        raise ArticleReadError('unavailable',
            'Article reader is not installed. Install src/requirements-news.txt.') from None

    if re.search(r'''["']isAccessibleForFree["']\s*:\s*["']?false\b''', html, re.I):
        raise ArticleReadError('blocked', 'The page marks its content as requiring paid access.')
    raw = extract(html, url=url, output_format='json', with_metadata=True,
                  include_comments=False, include_tables=True,
                  date_extraction_params={'extensive_search': False})
    document = json.loads(raw) if raw else {}
    text = (document.get('text') or '').strip()
    title = (document.get('title') or '').strip()
    # A 200 response can still be an access screen, not an article.
    blocked = re.compile(r'(site unavailable|access denied|verify (?:that )?you are (?:a )?human|'
                         r'verify (?:that )?you.re not a robot|checking your browser|'
                         r'subscribe to (?:continue|read)|sign in to (?:continue|read)|'
                         r'enable javascript and cookies)', re.I)
    if blocked.search(title) or (len(text) < 1500 and blocked.search(text)):
        raise ArticleReadError('blocked', 'The page returned an access screen rather than article text.')
    if len(text) < 100:
        raise ArticleReadError('unreadable', 'No usable article text could be extracted.')
    truncated = len(text) > MAX_TEXT_CHARS
    return {'title': title, 'published_at': document.get('date'),
            'text': text[:MAX_TEXT_CHARS], 'text_truncated': truncated,
            'completeness': 'truncated_by_hal' if truncated else 'extracted_not_independently_verified'}


def read_article(url):
    deadline = time.monotonic() + 15
    try:
        from trafilatura.utils import decode_file
        with requests.Session() as session:
            # Public reads must not inherit local proxy credentials or netrc auth.
            session.trust_env = False
            for _ in range(4):
                if time.monotonic() >= deadline:
                    raise ArticleReadError('timeout', 'Article download exceeded its time budget.')
                try:
                    public_url(url)
                except ValueError:
                    raise ArticleReadError('invalid_url', 'Article URL must resolve to the public internet.') from None
                except OSError:
                    raise ArticleReadError('unavailable', 'Article hostname could not be resolved; try another source.') from None
                with session.get(url, stream=True, allow_redirects=False, timeout=(4, 6),
                                 headers={'User-Agent': 'HAL9000 article reader',
                                          'Accept': 'text/html,application/xhtml+xml'}) as response:
                    if response.is_redirect:
                        location = response.headers.get('Location')
                        if not location:
                            raise ArticleReadError('unreadable', 'Article redirect has no destination.')
                        url = urljoin(url, location)
                        continue
                    if response.status_code in (401, 402, 403):
                        raise ArticleReadError('blocked',
                            f'Publisher refused access (HTTP {response.status_code}); try another source.')
                    if response.status_code == 429:
                        raise ArticleReadError('rate_limited', 'Publisher rate limited the request; try another source.')
                    response.raise_for_status()
                    mime = response.headers.get('Content-Type', '').split(';')[0].strip().lower()
                    if mime not in ('text/html', 'application/xhtml+xml'):
                        raise ArticleReadError('unreadable', 'The URL did not return an HTML article.')
                    if int(response.headers.get('Content-Length', '0')) > MAX_HTML_BYTES:
                        raise ArticleReadError('too_large', 'Article HTML exceeds the download limit.')
                    data = bytearray()
                    for chunk in response.iter_content(65536):
                        data.extend(chunk)
                        if len(data) > MAX_HTML_BYTES:
                            raise ArticleReadError('too_large', 'Article HTML exceeds the download limit.')
                        if time.monotonic() >= deadline:
                            raise ArticleReadError('timeout', 'Article download exceeded its time budget.')
                    # Detect HTML encoding, avoiding requests' ISO-8859-1 default.
                    article = extract_article(decode_file(bytes(data)), url)
                    return {**article, 'url': url, 'retrieved_at': datetime.now(timezone.utc).isoformat()}
        raise ArticleReadError('unreadable', 'Too many article redirects.')
    except ArticleReadError:
        raise
    except ImportError:
        raise ArticleReadError('unavailable',
            'Article reader is not installed. Install src/requirements-news.txt.') from None
    except requests.Timeout:
        raise ArticleReadError('timeout', 'Article download timed out; try another source.') from None
    except requests.RequestException:
        # Raw request exceptions can contain credentials or proxy details.
        raise ArticleReadError('unavailable', 'Article download failed; try another source.') from None
    except (ValueError, TypeError):
        raise ArticleReadError('unreadable', 'Article HTML or metadata could not be read.') from None
