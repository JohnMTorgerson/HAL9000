"""Small on-disk archive of exact, successfully displayed image files."""
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import tempfile

from image_files import MAX_IMAGE_BYTES, web_url
from image_lookup import ImageCandidate


class ImageHistory:
    def __init__(self, directory=None, *, limit=20, logger=None):
        self.directory = (Path(directory) if directory is not None else
                          Path(__file__).resolve().parent.parent / 'data' / 'image-history')
        self.limit = max(1, limit)
        self.logger = logger or logging.getLogger('HAL')
        self.records = []
        try:
            saved = json.loads((self.directory / 'history.json').read_text(encoding='utf-8'))
            if saved.get('version') != 1 or not isinstance(saved.get('images'), list):
                raise ValueError('Unrecognized image history')
            self.records = [r for r in saved['images'] if self._valid(r)][:self.limit]
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            self.logger.warning('Image history could not be read: %s', type(exc).__name__)

    @staticmethod
    def _valid(record):
        if not isinstance(record, dict):
            return False
        digest = record.get('sha256', '')
        return (isinstance(digest, str) and re.fullmatch(r'[a-f0-9]{64}', digest) is not None
                and record.get('id') == 'img_' + digest[:16]
                and all(isinstance(record.get(k), str) and len(record[k]) <= 1000
                        for k in ('query', 'description', 'shown_at'))
                and web_url(record.get('source_url')) and web_url(record.get('image_url'))
                and isinstance(record.get('citations'), list) and len(record['citations']) <= 6
                and all(isinstance(c, dict) and web_url(c.get('url'))
                        and isinstance(c.get('label'), str) and len(c['label']) <= 200
                        for c in record['citations']))

    def catalogue(self):
        """Descriptions and durable IDs go to the LLM, not image bytes."""
        return [{key: r[key] for key in ('id', 'query', 'description', 'shown_at', 'source_url')}
                for r in self.records]

    def _write(self, name, data):
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix='.image-', delete=False) as f:
                temporary = Path(f.name)
                f.write(data)
            temporary.replace(self.directory / name)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def remember(self, candidate, query):
        """Call only after a browser confirms loading; keep the original bytes."""
        if not candidate.data or len(candidate.data) > MAX_IMAGE_BYTES:
            raise ValueError('Invalid image size')
        digest = hashlib.sha256(candidate.data).hexdigest()
        record = {
            'id': 'img_' + digest[:16], 'sha256': digest, 'query': query[:1000],
            'description': candidate.description[:1000],
            'source_url': candidate.source_url, 'image_url': candidate.image_url,
            'citations': [{'url': c['url'], 'label': c['label'][:200]}
                          for c in candidate.citations[:6]],
            'shown_at': datetime.now(timezone.utc).isoformat(),
        }
        if not self._valid(record):
            raise ValueError('Invalid image history metadata')
        records = [record] + [r for r in self.records if r['id'] != record['id']]
        records = records[:self.limit]
        self._write(digest + '.jpg', candidate.data)
        self._write('history.json', json.dumps({'version': 1, 'images': records},
                                               ensure_ascii=False, indent=2).encode('utf-8'))
        self.records = records
        # Delete only archive-owned files, after the new manifest is committed.
        retained = {r['sha256'] + '.jpg' for r in records}
        try:
            for path in self.directory.glob('*.jpg'):
                if re.fullmatch(r'[a-f0-9]{64}\.jpg', path.name) and path.name not in retained:
                    path.unlink(missing_ok=True)
        except OSError as exc:
            self.logger.warning('Image history cleanup failed: %s', type(exc).__name__)
        return record['id']

    def recall(self, selector=''):
        """Select a durable ID, the latest image, or a conservatively matched subject.

        Subject matching is only a fallback for repaired spoken display claims;
        the conversation model normally supplies an ID from the catalogue.
        """
        if not selector:
            record = next(iter(self.records), None)
        elif selector.startswith('img_'):
            record = next((r for r in self.records if r['id'] == selector), None)
        else:
            words = set(re.findall(r'\w+', selector.casefold())) - {
                'a', 'an', 'the', 'of', 'picture', 'photo', 'image'}
            record = next((r for r in self.records if words and words <= set(
                re.findall(r'\w+', (r['query'] + ' ' + r['description']).casefold()))), None)
        if record is None:
            return None
        path = self.directory / (record['sha256'] + '.jpg')
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError('Saved image is too large')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('Saved image checksum mismatch')
        candidate = ImageCandidate(record['id'], record['image_url'], record['source_url'],
                                   data=data, description=record['description'],
                                   citations=record['citations'],
                                   reply='Here is that image again. ' + record['description'])
        return record['query'], candidate
