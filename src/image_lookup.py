"""Provider-independent image commands, cached alternatives, and display actions."""
from dataclasses import dataclass, field
import json
import logging
import re
import time
from urllib.parse import urlsplit

IMAGE_MARKER = '[IMAGE_REQUEST]'
IMAGE_FAILURE_REPLY = "I'm sorry, I couldn't complete that image request."


def repair_image_reply(user_input, reply):
    """Recover a false display claim for an unambiguous direct image request.

    This is a narrow guard after the LLM's routing decision, not a replacement
    for its handling of context, quoted speech, questions, or provider refusals.
    The caller must also exclude native provider refusals and ignored speech.
    """
    if reply.lstrip().startswith(('[IMAGE_REQUEST]', '[EXTERNAL_API_CALL]', '[PLAY_SONG]')):
        return reply
    # Do not reinterpret explanations, clarifying questions or plain refusals.
    if not re.search(r'\bhere (?:it is|they are|you (?:are|go))\b', reply, re.I):
        return reply
    if re.search(r"\b(?:can(?:not|'t|’t)|unable|won(?:'t|’t)|sorry)\b", reply, re.I):
        return reply
    request = ' '.join(user_input.split())
    request = re.sub(r'^(?:hey[, ]+)?(?:hal|al|hall|howl)[, .:!?]+', '', request, flags=re.I)
    request = re.sub(r'^(?:(?:can|could|would|will) you\s+)?(?:please\s+)?', '', request, flags=re.I)
    picture = re.fullmatch(
        r'(?:show|find|display)(?: me)? (?:an? |some |the )?'
        r'(?:picture|photo|image)s? of (.+?)[?.!]*', request, re.I)
    repeat = re.fullmatch(r'(?:show|display) me (?:that|the same) (.+?) again[?.!]*', request, re.I)
    match = picture or repeat
    if not match or not match[1].strip() or len(request) > 1000:
        return reply
    # These have dedicated display APIs, and an unnamed "that again" needs
    # context-aware routing rather than a guessed image subject.
    if re.search(r'\b(?:map|calendar|schedule|video)s?\b', match[1], re.I):
        return reply
    return IMAGE_MARKER + ' ' + json.dumps({'action': 'search', 'query': request})


def refusal_reply(reason=''):
    # A provider's explicit explanation is preferable to an invented category.
    reason = ' '.join(str(reason or '').split())
    return reason[:600] if reason else (
        'The provider declined that request without giving a specific reason.')


@dataclass
class ImageCandidate:
    id: str
    image_url: str
    source_url: str
    caption: str = ''
    data: bytes = b''
    reply: str = ''
    description: str = ''
    citations: list = field(default_factory=list)


@dataclass
class ImageResult:
    status: str
    reply: str = ''
    images: list = field(default_factory=list)
    action_result: str = ''


class ImageProviderError(Exception):
    def __init__(self, status, reply):
        super().__init__(reply)
        self.status, self.reply = status, reply


def parse_image_request(reply):
    if not reply.strip().startswith(IMAGE_MARKER):
        return None
    try:
        command = json.loads(reply.strip()[len(IMAGE_MARKER):])
    except (ValueError, TypeError):
        raise ValueError('Invalid image command JSON') from None
    if (not isinstance(command, dict) or set(command) != {'action', 'query'}
            or command['action'] not in ('search', 'next', 'previous', 'close')
            or not isinstance(command['query'], str) or len(command['query']) > 1000):
        raise ValueError('Invalid image command')
    command['query'] = command['query'].strip()
    if command['action'] == 'search' and not command['query']:
        raise ValueError('Missing image search query')
    if command['action'] != 'search' and command['query']:
        raise ValueError('Navigation does not accept a search query')
    return command


class UnsupportedImageProvider:
    supported = False

    def __init__(self, backend):
        self.backend = backend

    def lookup(self, query):
        label = 'Ollama' if self.backend == 'ollama' else 'the selected backend'
        return ImageResult('unsupported', f"Image lookup isn't available with {label} yet.")


def make_image_provider(llm, logger=None):
    if llm.backend != 'openai':
        return UnsupportedImageProvider(llm.backend)
    # All OpenAI search/vision details live behind this import and interface.
    from image_provider_openai import OpenAIImageProvider
    return OpenAIImageProvider(llm.client, llm.model_name, llm.service_tier, logger=logger)


class ImageWorkflow:
    def __init__(self, provider, display, *, logger=None, clock=time.monotonic):
        self.provider, self.display = provider, display
        self.logger = logger or logging.getLogger('HAL')
        self.clock = clock
        self.images, self.index, self.query = [], -1, ''
        self.cached_until = self.visible_until = 0

    def context(self):
        if self.clock() >= self.cached_until:
            self.images, self.index = [], -1
        return json.dumps({
            'image_lookup_available': self.provider.supported,
            'last_image_query': self.query if self.images else None,
            'last_image_description': self.images[self.index].description if self.index >= 0 else None,
            'cached_images': len(self.images),
            'last_image_number': self.index + 1,
            'display_timeout_elapsed': self.clock() >= self.visible_until,
        })

    def handle_reply(self, reply, *, on_search=None):
        try:
            command = parse_image_request(reply)
        except ValueError:
            return self._finish(ImageResult('error', IMAGE_FAILURE_REPLY))
        if command is None:
            return None
        self.context()  # Expire in-memory results after twenty minutes.
        action, query = command['action'], command['query']
        if action == 'close':
            try:
                self.display.clear(key='image-lookup')
            except Exception as exc:
                self.logger.info('Image display close failed: %s', type(exc).__name__)
                return self._finish(ImageResult('error', "I couldn't close the image display."))
            self.images, self.index, self.query = [], -1, ''
            self.visible_until = 0
            return self._finish(ImageResult('closed', 'Certainly.'))
        if not self.provider.supported:
            return self._finish(self.provider.lookup(query))
        if action == 'search':
            if on_search:
                on_search()
            self.logger.info('Image search requested: %s', query)
            try:
                result = self.provider.lookup(query)
            except ImageProviderError as exc:
                result = ImageResult(exc.status, exc.reply)
            except Exception as exc:
                # Image capability failures must not tear down the voice loop.
                self.logger.info('Image lookup failed: %s', type(exc).__name__)
                result = ImageResult('error', IMAGE_FAILURE_REPLY)
            if result.status != 'ok':
                return self._finish(result)
            self.images, self.index, self.query = result.images, -1, query
            self.cached_until = self.clock() + 1200
            target, direction = 0, 1
        else:
            direction = -1 if action == 'previous' else 1
            target = self.index + direction
            if not self.images:
                return self._finish(ImageResult('no_results', "I don't have any images to go back to. Please ask me to find one."))
            if not 0 <= target < len(self.images):
                return self._finish(ImageResult('no_results', "I don't have another image in that direction."))
        while 0 <= target < len(self.images):
            candidate = self.images[target]
            citations = [{'url': candidate.source_url,
                          'label': urlsplit(candidate.source_url).hostname}]
            citations += [c for c in candidate.citations if c['url'] != candidate.source_url]
            try:
                status = self.display.present_image(candidate.data, citations=citations[:6], ttl=120)
            except Exception as exc:
                self.logger.info('Image display failed: %s', type(exc).__name__)
                status = 'unavailable'
            if status == 'loaded':
                self.index = target
                self.visible_until = self.clock() + 120
                self.logger.info('IMAGE SHOWN: query=%s; description=%s; source=%s; image=%s',
                                 self.query, candidate.description, candidate.source_url, candidate.image_url)
                for citation in citations:
                    self.logger.info('IMAGE CITATION: %s — %s', citation['label'], citation['url'])
                return self._finish(ImageResult('ok', candidate.reply, action_result=(
                    f'Showed image {target + 1} of {len(self.images)} for {self.query}. '
                    f'{candidate.description} Source: {candidate.source_url}. '
                    'Display expires after two minutes; alternatives cached for twenty minutes.')))
            if status != 'error':  # No browser/hidden display: do not repeat the wait for each candidate.
                break
            target += direction
        return self._finish(ImageResult('error', "I found an image, but couldn't show it on the display."))

    def _finish(self, result):
        if not result.action_result:
            result.action_result = f'Image request outcome: {result.status}. {result.reply}'
        self.logger.info('Image request outcome: %s; %s', result.status, result.reply)
        return result
