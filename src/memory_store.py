"""Human-editable JSON memory, with atomic writes and a durable update cursor."""
import copy
from datetime import date, datetime
import fcntl
import hashlib
import json
from pathlib import Path
import os
import tempfile
import threading
import uuid

from memory_retrieval import compact_entry, retrieve

SECTIONS = ('personal', 'hal', 'topics')
TAG_BATCH_SIZE = 20


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _date(value):
    _require(isinstance(value, str), 'Expected an ISO date.')
    parsed = date.fromisoformat(value)
    _require(parsed.isoformat() == value, 'Expected YYYY-MM-DD.')
    return parsed


def _text(value, limit):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= limit


def _tags(value, *, allow_empty=False):
    _require(isinstance(value, list) and (0 if allow_empty else 1) <= len(value) <= 24,
             'Memory needs 1–24 search tags.')
    _require(all(_text(tag, 60) for tag in value), 'Search tags must be 1–60 characters.')
    return list(dict.fromkeys(' '.join(tag.casefold().split()) for tag in value))


def _retention(item, section, *, allow_legacy=False):
    if allow_legacy and section == 'topics' and item.get('retention') == 'temporary':
        _date(item.get('expires_on'))
        return
    _require(item.get('retention') == 'durable' and item.get('expires_on') is None,
             'All memories are durable and cannot expire.')


def _replace(path, data):
    """A crash exposes either the old complete file or the new complete file."""
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class MemoryStore:
    """One writer per directory; never hold the lock during an API request."""

    def __init__(self, directory, max_history=10, *, logger, now=None):
        _require(1 <= max_history <= 100, 'Memory needs LLM_MAX_HISTORY between 1 and 100.')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_history, self.logger = max_history, logger
        self.now = now or (lambda: datetime.now().astimezone())
        self.lock = threading.RLock()
        self.closed = False
        self.signatures = {}
        self.file_lock = (self.directory / '.writer.lock').open('a')
        try:
            fcntl.flock(self.file_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.memory = self._load('memory.json', {
                'version': 2, 'last_processed_turn': 0, 'context_after_turn': 0,
                'personal': [], 'hal': [], 'topics': [],
            })
            self.recent = self._load('recent_conversation.json', {
                'version': 1, 'next_turn_id': self.memory['last_processed_turn'] + 1, 'turns': [],
            })
            loaded_memory = self.memory
            self.memory = self._migrate(self.memory)
            self._validate_memory(self.memory)
            self._validate_recent(self.recent)
            old_topics = {i['id']: i.get('expires_on') for i in loaded_memory['topics']}
            # Missing memory alongside an existing recent file starts fresh; do
            # not silently regenerate deleted memories from older conversation.
            if self.signatures['memory.json'] is None:
                self.memory['last_processed_turn'] = self.recent['next_turn_id'] - 1
            self.recent['next_turn_id'] = max(self.recent['next_turn_id'],
                                               self.memory['last_processed_turn'] + 1)
            self._save('memory.json', self.memory)
            self._save('recent_conversation.json', self.recent)
            promoted = {ident: expiry for ident, expiry in old_topics.items() if expiry is not None}
            if promoted:
                self.logger.info('Memory retention migration: %s', json.dumps({
                    'promoted_topics': promoted, 'retention': 'durable',
                    'reason': 'Conversation history does not expire when an event ends.'}))
        except Exception:
            self.close()
            raise

    @staticmethod
    def _migrate(value):
        """Validate legacy records, then retain all surviving topics permanently."""
        if not isinstance(value, dict) or value.get('version') not in (1, 2):
            return value
        value = copy.deepcopy(value)
        if value['version'] == 1:
            _require('hal' not in value, 'Version 1 memory unexpectedly contains a HAL section.')
            for section in ('personal', 'topics'):
                _require(isinstance(value.get(section), list), 'Missing legacy memory section.')
                for item in value[section]:
                    _require(isinstance(item, dict) and 'tags' not in item and 'retention' not in item,
                             'Invalid legacy memory entry.')
                    item['tags'] = []
                    item['retention'] = 'durable' if section == 'personal' else 'temporary'
                    _require(isinstance(item.get('evidence'), list), 'Invalid legacy evidence.')
                    for source in item['evidence']:
                        _require(isinstance(source, dict) and 'role' not in source, 'Invalid legacy source.')
                        source['role'] = 'user'
            value.update(version=2, hal=[])
        MemoryStore._validate_memory(value, allow_legacy=True)
        for item in value['topics']:
            item.update(retention='durable', expires_on=None)
        return value

    def _load(self, name, default):
        path = self.directory / name
        if not path.exists():
            self.signatures[name] = None
            return default
        _require(path.stat().st_size <= 2_000_000, name + ' is larger than 2 MB; inspect it before restarting.')
        data = path.read_bytes()
        self.signatures[name] = hashlib.sha256(data).digest()
        return json.loads(data)

    def _save(self, name, value):
        _require(not self.closed, 'Memory store is closed.')
        path = self.directory / name
        old = path.read_bytes() if path.exists() else None
        signature = hashlib.sha256(old).digest() if old is not None else None
        _require(signature == self.signatures.get(name),
                 name + ' changed outside HAL. Stop HAL, finish editing, and restart.')
        data = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
        _require(len(data) <= 2_000_000, name + ' would exceed 2 MB; memory updates paused.')
        if data == old:
            return
        if old is not None:
            _replace(path.with_suffix(path.suffix + '.bak'), old)
        _replace(path, data)
        self.signatures[name] = hashlib.sha256(data).digest()

    @staticmethod
    def _validate_memory(value, *, allow_legacy=False):
        _require(isinstance(value, dict) and value.get('version') == 2,
                 'Unsupported memory.json format.')
        for key in ('last_processed_turn', 'context_after_turn'):
            _require(type(value.get(key)) is int and value[key] >= 0, 'Invalid memory cursor.')
        _require(value['context_after_turn'] <= value['last_processed_turn'], 'Invalid forgotten-context cursor.')
        ids = set()
        for section in SECTIONS:
            _require(isinstance(value.get(section), list), 'Missing memory section: ' + section)
            for item in value[section]:
                _require(isinstance(item, dict) and _text(item.get('id'), 80)
                         and item['id'] not in ids, 'Invalid or duplicate memory id.')
                ids.add(item['id'])
                _require(_text(item.get('text'), 2000 if section == 'topics' else 800),
                         'Memory text exceeds the section limit.')
                _require(item.get('basis') in ('explicit', 'inferred'), 'Invalid memory basis.')
                _require(section != 'hal' or item['basis'] == 'explicit', 'HAL views need explicit evidence.')
                _tags(item.get('tags'), allow_empty=True)  # Migrated notes await background indexing.
                for field in ('created_at', 'updated_at'):
                    datetime.fromisoformat(item[field])
                _retention(item, section, allow_legacy=allow_legacy)
                _require(type(item.get('evidence_count')) is int and item['evidence_count'] >= 1,
                         'Invalid memory evidence count.')
                _require(isinstance(item.get('evidence_dates'), list)
                         and len(item['evidence_dates']) <= 12, 'Invalid evidence dates.')
                for day in item['evidence_dates']:
                    _date(day)
                _require(isinstance(item.get('evidence'), list) and len(item['evidence']) <= 3,
                         'Invalid memory evidence.')
                for source in item['evidence']:
                    _require(isinstance(source, dict) and type(source.get('turn_id')) is int
                             and source.get('role') in ('user', 'assistant')
                             and _text(source.get('quote'), 300), 'Invalid memory source.')
                    _require(section != 'personal' or source['role'] == 'user',
                             'Personal memory evidence must come from the user.')
                    _require(section != 'hal' or source['role'] == 'assistant',
                             'HAL memory evidence must come from HAL.')
                    _require(type(source.get('context_only', False)) is bool,
                             'Invalid evidence context marker.')
                    datetime.fromisoformat(source['at'])

    @staticmethod
    def _validate_recent(value):
        _require(isinstance(value, dict) and value.get('version') == 1
                 and type(value.get('next_turn_id')) is int and value['next_turn_id'] > 0
                 and isinstance(value.get('turns'), list), 'Invalid recent_conversation.json.')
        previous = 0
        for turn in value['turns']:
            _require(isinstance(turn, dict) and type(turn.get('id')) is int
                     and previous < turn['id'] < value['next_turn_id'], 'Invalid turn order.')
            previous = turn['id']
            _require(_text(turn.get('user_speech'), 20_000)
                     and _text(turn.get('assistant_reply'), 20_000), 'Invalid saved exchange.')
            datetime.fromisoformat(turn['at'])

    def _active_memory(self):
        return {section: copy.deepcopy(self.memory[section]) for section in SECTIONS}

    def recall(self, query='', *, token_budget=3000):
        with self.lock:
            active = self._active_memory()
            recent = [turn for turn in self.recent['turns'][-self.max_history:]
                      if turn['id'] > self.memory['context_after_turn']]
            selected, diagnostics = retrieve(active, query, recent, token_budget=token_budget)
            self.logger.info('Memory retrieval: %s', json.dumps(
                {'purpose': 'foreground', **diagnostics}, ensure_ascii=False))
            facts = {section: [compact_entry(item) for item in items]
                     for section, items in selected.items()}
            history = []
            for turn in recent:
                history.extend([
                    {'role': 'user', 'content': f"[{turn['at']}] {turn['user_speech']}"},
                    {'role': 'assistant', 'content': turn['assistant_reply']},
                ])
            return history, json.dumps(facts, ensure_ascii=False)

    def record_turn(self, user_speech, assistant_reply, at=None):
        _require(_text(user_speech, 20_000) and _text(assistant_reply, 20_000), 'Invalid completed exchange.')
        with self.lock:
            updated = copy.deepcopy(self.recent)
            turn = {'id': updated['next_turn_id'], 'at': at or self.now().isoformat(),
                    'user_speech': user_speech, 'assistant_reply': assistant_reply}
            updated['next_turn_id'] += 1
            updated['turns'].append(turn)
            self._trim(updated)
            self._save('recent_conversation.json', updated)
            self.recent = updated
            return turn['id']

    def _trim(self, recent):
        keep = {turn['id'] for turn in recent['turns'][-self.max_history:]}
        # Unprocessed turns survive trimming and restarts, even after an API error.
        recent['turns'] = [t for t in recent['turns']
                           if t['id'] > self.memory['context_after_turn']
                           and (t['id'] > self.memory['last_processed_turn'] or t['id'] in keep)]

    def next_batch(self, *, token_budget=3000):
        with self.lock:
            cursor = self.memory['last_processed_turn']
            # Process one exchange at a time, in order. In particular, a forget
            # request must not consume newer statements queued behind it.
            turns = [t for t in self.recent['turns'] if t['id'] > cursor][:1]
            active = self._active_memory()
            tagging = [{'section': section, **compact_entry(item)}
                       for section, items in active.items() for item in items if not item['tags']][:TAG_BATCH_SIZE]
            if not turns and not tagging:
                return None
            prior = [t for t in self.recent['turns']
                     if self.memory['context_after_turn'] < t['id'] < (turns[0]['id'] if turns else cursor + 1)][-4:]
            query = '\n'.join(t['user_speech'] + '\n' + t['assistant_reply'] for t in turns)
            selected, diagnostics = retrieve(active, query, prior, token_budget=token_budget)
            self.logger.info('Memory retrieval: %s', json.dumps(
                {'purpose': 'background', **diagnostics}, ensure_ascii=False))
            return copy.deepcopy({
                'cursor': cursor, 'today': self.now().date().isoformat(),
                'memory': selected,
                # All semantic content remains visible to maintenance, so a
                # correction/forget cannot miss a detail outside retrieval.
                'catalogue': {section: [{**compact_entry(item),
                                        'evidence_count': item['evidence_count'],
                                        'evidence_dates': item['evidence_dates']} for item in items]
                              for section, items in active.items()},
                'tagging_entries': tagging, 'earlier_context': prior, 'new_turns': turns,
            })

    @staticmethod
    def _evidence(sources, turns, *, roles=('user',), earlier=(), retained=()):
        _require(isinstance(sources, list) and 1 <= len(sources) <= 3, 'Each change needs 1–3 sources.')
        context = {turn['id']: turn for turn in earlier}
        result, seen = [], set()
        for source in sources:
            _require(isinstance(source, dict) and set(source) == {'turn_id', 'role', 'quote'},
                     'Invalid source fields.')
            ident, quote, role = source['turn_id'], source['quote'], source['role']
            _require(role in roles, 'Evidence role is not allowed for this memory change.')
            _require(type(ident) is int and (ident, role) not in seen,
                     'Evidence must cite distinct speaker sources.')
            turn = turns.get(ident) or context.get(ident)
            original = None
            if turn is not None:
                speech = (turn['user_speech'] if role == 'user' else
                          turn['assistant_reply'].split('\n[Application action result:', 1)[0])
                at = turn['at']
            else:
                original = next((s for s in retained if s['turn_id'] == ident and s['role'] == role
                                 and isinstance(quote, str) and quote in s['quote']), None)
                _require(original is not None,
                         'Evidence must cite NEW accepted turns or supplied supporting context.')
                speech, at = original['quote'], original['at']
            _require(_text(quote, 300) and quote in speech,
                     'Evidence quote must be exact user speech or HAL speech from its declared role.')
            seen.add((ident, role))
            item = {'turn_id': ident, 'at': at, 'role': role, 'quote': quote}
            if ident not in turns:
                item['context_only'] = True
            result.append(item)
        _require(any(s['turn_id'] in turns for s in result),
                 'Each change requires evidence from a NEW accepted turn; context cannot reinforce itself.')
        return result

    def apply(self, batch, changes):
        """Validate the whole patch before committing entries, tags or queue cursor."""
        with self.lock:
            _require(batch['cursor'] == self.memory['last_processed_turn'], 'Stale memory update.')
            _require(isinstance(changes, dict) and set(changes) == {
                'forget', 'forget_ids', 'forget_evidence', 'tag_updates', 'operations'}
                and type(changes['forget']) is bool and isinstance(changes['operations'], list)
                and len(changes['operations']) <= 20 and isinstance(changes['tag_updates'], list)
                and len(changes['tag_updates']) <= TAG_BATCH_SIZE, 'Invalid memory changes.')
            turns = {t['id']: t for t in batch['new_turns']}
            forget = changes['forget']
            forget_ids = changes['forget_ids']
            _require(isinstance(forget_ids, list) and all(isinstance(i, str) for i in forget_ids)
                     and len(set(forget_ids)) == len(forget_ids), 'Invalid forget IDs.')
            forgotten_sources = []
            if forget:
                forgotten_sources = self._evidence(changes['forget_evidence'], turns)
                _require(not changes['operations'] and not changes['tag_updates'],
                         'Forget batches only use forget_ids; no other mutations.')
            else:
                _require(not forget_ids and changes['forget_evidence'] == [], 'Unexpected forget evidence/IDs.')
            _require(bool(turns) or (not changes['operations'] and not forget),
                     'Tag-only batches cannot change facts or forget.')
            updated, events, touched = copy.deepcopy(self.memory), [], set()
            catalogue_ids = {item['id'] for items in batch['catalogue'].values() for item in items}
            # An inferred personal interest can cite older USER quotes from a
            # retrieved topic. Never expose arbitrary archive evidence or bypass
            # the per-section role checks / requirement for a relevant new turn.
            supplied_sources = [s for items in batch['memory'].values()
                                for item in items for s in item['evidence']
                                if s['turn_id'] > self.memory['context_after_turn']]
            _require(set(forget_ids) <= catalogue_ids, 'Unknown forgotten memory ID.')
            for section in SECTIONS:
                for item in list(updated[section]):
                    if item['id'] in forget_ids:
                        updated[section].remove(item)
                        touched.add(item['id'])
                        events.append({'action': 'delete', 'section': section, 'before': item,
                                       'after': None, 'reason': 'Explicit request to forget.',
                                       'evidence': forgotten_sources})
            for op in changes['operations']:
                _require(isinstance(op, dict) and set(op) == {
                    'action', 'section', 'id', 'text', 'basis', 'retention', 'tags',
                    'expires_on', 'reason', 'evidence'}, 'Invalid operation fields.')
                action, section, ident = op['action'], op['section'], op['id']
                _require(action in ('add', 'update', 'extend', 'reinforce', 'delete') and
                         section in SECTIONS and isinstance(ident, str), 'Invalid operation.')
                _require(action != 'extend' or section == 'topics',
                         'Only discussion topics can be extended.')
                _require(_text(op['text'], 2000 if section == 'topics' else 800)
                         and _text(op['reason'], 400) and op['basis'] in ('explicit', 'inferred'),
                         'Invalid operation text or basis.')
                _require(section != 'hal' or op['basis'] == 'explicit', 'HAL views need explicit evidence.')
                roles = (('user',) if section == 'personal' else ('assistant',)
                         if section == 'hal' and action != 'delete' else ('user', 'assistant'))
                sources = self._evidence(op['evidence'], turns, roles=roles,
                    earlier=batch['earlier_context'], retained=supplied_sources)
                fresh = [s for s in sources if not s.get('context_only', False)]
                _retention(op, section)
                tags = _tags(op['tags'], allow_empty=action == 'delete')
                existing = next((i for i in updated[section] if i['id'] == ident), None)
                if action == 'add':
                    _require(ident == '', 'New memories cannot choose their own id.')
                    _require(not any(i['text'].casefold() == op['text'].casefold()
                                     for i in updated[section]), 'Duplicate memory; update the existing entry.')
                else:
                    _require(existing is not None and ident in catalogue_ids and ident not in touched,
                             'Unknown or repeated memory id.')
                    touched.add(ident)
                before = copy.deepcopy(existing)
                if action == 'delete':
                    updated[section].remove(existing)
                    after = None
                else:
                    if action == 'reinforce':
                        _require(op['text'] == existing['text'] and op['basis'] == existing['basis']
                                 and op['retention'] == existing['retention'],
                                 'Reinforcement cannot rewrite a fact or retention.')
                    at = self.now().isoformat()
                    if existing is None:
                        existing = {'id': 'm_' + uuid.uuid4().hex[:12], 'created_at': at,
                                    'evidence_count': 0, 'evidence_dates': [], 'evidence': []}
                        updated[section].append(existing)
                    existing.update(text=op['text'], basis=op['basis'], expires_on=op['expires_on'],
                                    retention=op['retention'], tags=tags, updated_at=at)
                    if action == 'update':
                        existing['evidence_count'] = 0
                        existing['evidence_dates'] = []
                    existing['evidence_count'] += len(fresh)
                    dates = existing['evidence_dates'] + [s['at'][:10] for s in fresh]
                    existing['evidence_dates'] = sorted(set(dates))[-12:]
                    evidence = [] if action == 'update' else existing['evidence']
                    # A re-cited old quote replaces its prior representation,
                    # without duplicating or counting it as new corroboration.
                    cited = {(s['turn_id'], s['role']) for s in sources}
                    evidence = [s for s in evidence if (s['turn_id'], s['role']) not in cited]
                    existing['evidence'] = (evidence + sources)[-3:]
                    after = copy.deepcopy(existing)
                events.append({'action': action, 'section': section, 'before': before,
                               'after': after, 'reason': op['reason'], 'evidence': sources})

            expected_tags = {(i['section'], i['id']) for i in batch['tagging_entries']}
            indexed = set()
            for tagging in changes['tag_updates']:
                _require(isinstance(tagging, dict) and set(tagging) == {'section', 'id', 'tags'},
                         'Invalid tag update fields.')
                section, ident = tagging['section'], tagging['id']
                _require(isinstance(section, str) and isinstance(ident, str), 'Invalid tag update target.')
                key = (section, ident)
                _require(key in expected_tags and key not in indexed and ident not in touched,
                         'Unknown or repeated indexing target.')
                item = next((i for i in updated[section] if i['id'] == ident), None)
                _require(item is not None and not item['tags'], 'Stale tag-only update.')
                before = copy.deepcopy(item)
                item['tags'] = _tags(tagging['tags'])
                indexed.add(key)
                events.append({'action': 'tag', 'section': section, 'before': before,
                               'after': copy.deepcopy(item), 'reason': 'Indexed existing memory; facts unchanged.'})
            if not forget:
                _require(all(key in indexed or key[1] in touched for key in expected_tags),
                         'Every pending indexing entry must receive tags or be updated/deleted.')
            if turns:
                updated['last_processed_turn'] = batch['new_turns'][-1]['id']
            if forget:
                updated['context_after_turn'] = updated['last_processed_turn']
            self._validate_memory(updated)
            # Entries and cursor commit together. Tagging never advances the
            # dialogue cursor or creates reinforcement from old material.
            self._save('memory.json', updated)
            self.memory = updated
            self.logger.info('Memory decision: %s', json.dumps({
                'through_turn': updated['last_processed_turn'], 'forget': forget,
                'forget_evidence': changes['forget_evidence'], 'changes': events,
            }, ensure_ascii=False))
            recent = copy.deepcopy(self.recent)
            self._trim(recent)
            self._save('recent_conversation.json', recent)
            self.recent = recent
            return len(events)

    def close(self):
        with self.lock:
            self.closed = True
            if not self.file_lock.closed:
                self.file_lock.close()
