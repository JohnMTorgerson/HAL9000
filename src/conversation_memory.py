"""Optional persistent memory; one background worker, no foreground API wait."""
from dataclasses import dataclass
from collections import Counter
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import threading
import time

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
from memory_prompts import MEMORY_FORMAT, recall_instructions, update_instructions
from memory_store import MemoryStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class MemorySettings:
    enabled: bool = False
    directory: Path = PROJECT_ROOT / 'data' / 'memory'
    model: str = 'gpt-6-luna'
    reasoning: str = 'medium'
    max_tokens: int = 8192
    timeout: float = 120.
    soft_tokens: int = 3000

    def __post_init__(self):
        if not self.model.strip():
            raise ValueError('MEMORY_MODEL must not be empty.')
        if self.reasoning not in ('none', 'low', 'medium', 'high', 'xhigh', 'max'):
            raise ValueError('MEMORY_REASONING_EFFORT must be none, low, medium, high, xhigh, or max.')
        if not 1024 <= self.max_tokens <= 32768:
            raise ValueError('MEMORY_MAX_COMPLETION_TOKENS must be between 1024 and 32768 (includes reasoning).')
        if not math.isfinite(self.timeout) or not 5 <= self.timeout <= 300:
            raise ValueError('MEMORY_TIMEOUT_SECONDS must be between 5 and 300.')
        if not 500 <= self.soft_tokens <= 32000:
            raise ValueError('MEMORY_SOFT_TOKEN_BUDGET must be between 500 and 32000.')

    @classmethod
    def from_env(cls):
        flag = os.getenv('MEMORY_ENABLED', 'false').strip().lower()
        if flag not in ('true', 'false', '1', '0', 'yes', 'no', 'on', 'off'):
            raise ValueError('MEMORY_ENABLED must be true or false.')
        if flag in ('false', '0', 'no', 'off'):
            return cls()
        directory = Path(os.getenv('MEMORY_DIR', 'data/memory')).expanduser()
        if not directory.is_absolute():
            directory = PROJECT_ROOT / directory
        return cls(True, directory, os.getenv('MEMORY_MODEL') or os.getenv('LLM_MODEL') or 'gpt-6-luna',
                   os.getenv('MEMORY_REASONING_EFFORT', 'medium').strip().lower(),
                   int(os.getenv('MEMORY_MAX_COMPLETION_TOKENS', '8192')),
                   float(os.getenv('MEMORY_TIMEOUT_SECONDS', '120')),
                   int(os.getenv('MEMORY_SOFT_TOKEN_BUDGET', '3000')))


def memory_logger(directory):
    """Detailed decisions have a separate file, with no propagation to the display."""
    path = Path(directory).expanduser().resolve() / 'memory.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('HAL.memory.decisions')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(getattr(h, 'baseFilename', None) == str(path) for h in logger.handlers):
        handler = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s :: %(message)s'))
        logger.addHandler(handler)
    return logger


def _change_summary(batch, changes):
    """Describe a successfully applied patch without copying private memory text."""
    labels = {'personal': 'User memory', 'hal': 'HAL memory', 'topics': 'Topic'}
    verbs = {'add': 'added', 'update': 'updated', 'extend': 'extended',
             'reinforce': 'reinforced', 'delete': 'removed', 'tag': 'tags updated'}
    counts = Counter()

    def add(section, action):
        verb = 'created' if action == 'add' and section != 'topics' else verbs[action]
        counts[f'{labels[section]} {verb}'] += 1

    for operation in changes['operations']:
        add(operation['section'], operation['action'])
    for entry in changes['tag_updates']:
        add(entry['section'], 'tag')
    forgotten = set(changes['forget_ids'])
    for section, entries in batch['catalogue'].items():
        for entry in entries:
            if entry['id'] in forgotten:
                add(section, 'delete')
    return '; '.join(label if count == 1 else f'{label} ({count})'
                     for label, count in counts.items())


class MemoryUpdater:
    """Separate client/settings: foreground fast mode is never reused here."""

    def __init__(self, settings, api_key, logger):
        self.settings, self.logger = settings, logger
        if OpenAI is None:
            raise ValueError('Persistent memory requires the OpenAI Python package.')
        self.client = OpenAI(api_key=api_key, max_retries=0, timeout=settings.timeout)

    def update(self, batch):
        settings = self.settings
        response = self.client.chat.completions.create(
            model=settings.model, reasoning_effort=settings.reasoning,
            max_completion_tokens=settings.max_tokens, service_tier='default',
            response_format=MEMORY_FORMAT,
            messages=[{'role': 'system', 'content': update_instructions()},
                      {'role': 'user', 'content': json.dumps(batch, ensure_ascii=False)}],
        )
        usage = getattr(response, 'usage', None)
        self.logger.info('Memory API usage: %s',
                         usage.model_dump_json() if usage is not None else 'not reported')
        if not response.choices:
            raise ValueError('No memory decision received.')
        choice = response.choices[0]
        if choice.finish_reason != 'stop' or getattr(choice.message, 'refusal', None):
            raise ValueError('Incomplete/refused memory decision. If truncated, increase MEMORY_MAX_COMPLETION_TOKENS.')
        try:
            return json.loads(choice.message.content)
        except (TypeError, ValueError) as exc:
            raise ValueError('Invalid JSON memory decision.') from exc

    def close(self):
        self.client.close()


class ConversationMemory:
    def __init__(self, store, updater, settings, logger, decisions):
        self.store, self.updater, self.settings = store, updater, settings
        self.logger, self.decisions = logger, decisions
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.available = True
        self.thread = threading.Thread(target=self._run, name='HAL-memory', daemon=True)
        self.thread.start()
        self.wake.set()  # Resume saved pending work after a restart.

    @classmethod
    def from_env(cls, logger, max_history):
        settings = MemorySettings.from_env()
        if not settings.enabled:
            logger.info('Persistent memory: disabled (MEMORY_ENABLED=false).')
            return None
        if os.getenv('LLM_BACKEND', 'openai') != 'openai':
            raise ValueError('MEMORY_ENABLED currently requires LLM_BACKEND=openai.')
        store = None
        decisions = None
        try:
            decisions = memory_logger(os.environ['LOG_PATH'])
            store = MemoryStore(settings.directory, max_history, logger=decisions)
            updater = MemoryUpdater(settings, os.getenv('OPENAI_API_KEY'), decisions)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if store is not None:
                store.close()
            # Do not echo a malformed JSON document, API key, or raw SDK error.
            message = ('Persistent memory unavailable (%s). Check the JSON files/backups, '
                       'MEMORY settings, permissions, and whether another HAL instance is running. '
                       'Existing files were not reset; continuing without persistent memory.')
            logger.error(message, type(exc).__name__)
            if decisions is not None:
                decisions.error(message, type(exc).__name__)
                if isinstance(exc, (ValueError, OSError)):
                    decisions.error('Storage/configuration detail: %s', exc)
            return None
        logger.info('Persistent memory ready: %s; background model %s; reasoning %s.',
                    settings.directory, settings.model, settings.reasoning)
        decisions.info('Memory started: directory=%s model=%s reasoning=%s max_completion_tokens=%s tier=default.',
                       settings.directory, settings.model, settings.reasoning, settings.max_tokens)
        return cls(store, updater, settings, logger, decisions)

    def read_context(self, query=''):
        if not self.available:
            return None
        history, facts = self.store.recall(query=query, token_budget=self.settings.soft_tokens)
        # Selection is bounded by a character estimate, never by deleting data.
        # Detailed queries, matching terms, and selected entries stay in memory.log.
        estimated = (len(facts) + 3) // 4
        self.logger.info('Memory recall: about %s tokens selected (character estimate); details in memory.log.',
                         estimated)
        return history, recall_instructions() + '\n' + facts

    def record_turn(self, user_speech, assistant_reply, at=None):
        if not self.available or self.stopping.is_set():
            return
        if (not isinstance(user_speech, str) or not user_speech.strip()
                or not isinstance(assistant_reply, str) or not assistant_reply.strip()):
            self.decisions.info('Memory skipped: empty or non-text exchange.')
            return
        try:
            ident = self.store.record_turn(user_speech, assistant_reply, at)
            self.decisions.info('Queued accepted exchange %s: %s', ident, json.dumps({
                'user_speech': user_speech, 'assistant_reply': assistant_reply,
            }, ensure_ascii=False))
            self.wake.set()
        except (OSError, ValueError) as exc:
            self._storage_failed(exc)

    def _storage_failed(self, exc):
        self.available = False
        self.stopping.set()
        self.wake.set()
        self.logger.error('Persistent memory paused after a storage error (%s); HAL can keep responding. '
                          'Inspect memory.log and the JSON files before restarting.', type(exc).__name__)
        self.decisions.error('Memory storage error: %s: %s', type(exc).__name__, str(exc))

    def _run(self):
        try:
            while not self.stopping.is_set():
                self.wake.wait()
                self.wake.clear()
                while not self.stopping.is_set() and (
                        batch := self.store.next_batch(token_budget=self.settings.soft_tokens)) is not None:
                    started = time.perf_counter()
                    self.decisions.info('Memory update started: turns=%s tagging_entries=%s previous_cursor=%s.',
                                        [t['id'] for t in batch['new_turns']],
                                        len(batch.get('tagging_entries', [])), batch['cursor'])
                    try:
                        changes = self.updater.update(batch)
                    except Exception as exc:
                        # No automatic retry loop: queued work is retained. Try
                        # again on a later completed exchange or next startup.
                        self.decisions.error('Memory API/decision failure: type=%s status=%s code=%s. '
                                             'Pending exchanges retained; no immediate retry.',
                                             type(exc).__name__, getattr(exc, 'status_code', None),
                                             getattr(exc, 'code', None))
                        if isinstance(exc, ValueError):
                            self.decisions.error('%s', exc)
                        elif getattr(exc, 'status_code', None) in (400, 403, 404):
                            self.decisions.error('Check MEMORY_MODEL access, MEMORY_REASONING_EFFORT, '
                                                 'and structured-output support for that model.')
                        elif getattr(exc, 'status_code', None) in (401, 429):
                            self.decisions.error('Check API credentials, credits, and usage/rate limits.')
                        self.logger.warning('Background memory update failed; response unaffected. See memory.log.')
                        self.wake.clear()
                        break
                    if self.stopping.is_set():
                        break
                    try:
                        count = self.store.apply(batch, changes)
                    except OSError as exc:
                        self._storage_failed(exc)
                        break
                    except ValueError as exc:
                        self.decisions.error('Rejected memory patch: %s. Existing memories retained.', exc)
                        self.decisions.error('Rejected memory patch details: %s', json.dumps({
                            'previous_cursor': batch['cursor'],
                            'new_turn_ids': [t['id'] for t in batch['new_turns']],
                            'reason': str(exc),
                            'validation': getattr(exc, 'details', None),
                            'proposed_changes': changes,
                        }, ensure_ascii=False))
                        self.logger.warning('Background memory patch rejected; see memory.log. Pending exchanges retained.')
                        self.wake.clear()
                        break
                    elapsed = time.perf_counter() - started
                    self.decisions.info('Memory update finished: %.3fs; %s changes.', elapsed, count)
                    if count:
                        self.logger.info('Background memory: %s. %s changes in %.2fs; details in memory.log.',
                                         _change_summary(batch, changes), count, elapsed,
                                         extra={'memory_change': True})
                    else:
                        self.logger.info('Background memory: no changes in %.2fs; details in memory.log.', elapsed)
        finally:
            self.updater.close()

    def close(self):
        self.stopping.set()
        self.wake.set()
        self.thread.join(timeout=1.)
        # Pending exchanges are already on disk. Do not keep HAL running to
        # finish a paid request; a late result cannot write to a closed store.
        self.store.close()
