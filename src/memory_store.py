"""Human-editable JSON memory, with atomic writes and a durable update cursor."""
import copy
from datetime import date, datetime, timedelta
import fcntl
import hashlib
import json
from pathlib import Path
import os
import tempfile
import threading
import uuid


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
                'version': 1, 'last_processed_turn': 0, 'context_after_turn': 0,
                'personal': [], 'topics': [],
            })
            self.recent = self._load('recent_conversation.json', {
                'version': 1, 'next_turn_id': self.memory['last_processed_turn'] + 1, 'turns': [],
            })
            self._validate_memory(self.memory)
            self._validate_recent(self.recent)
            # Missing memory alongside an existing recent file starts fresh; do
            # not silently regenerate deleted memories from older conversation.
            if self.signatures['memory.json'] is None:
                self.memory['last_processed_turn'] = self.recent['next_turn_id'] - 1
            self.recent['next_turn_id'] = max(self.recent['next_turn_id'],
                                               self.memory['last_processed_turn'] + 1)
            self._save('memory.json', self.memory)
            self._save('recent_conversation.json', self.recent)
        except Exception:
            self.close()
            raise

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
    def _validate_memory(value):
        _require(isinstance(value, dict) and value.get('version') == 1,
                 'Unsupported memory.json format.')
        for key in ('last_processed_turn', 'context_after_turn'):
            _require(type(value.get(key)) is int and value[key] >= 0, 'Invalid memory cursor.')
        _require(value['context_after_turn'] <= value['last_processed_turn'], 'Invalid forgotten-context cursor.')
        ids = set()
        for section in ('personal', 'topics'):
            _require(isinstance(value.get(section), list), 'Missing memory section: ' + section)
            for item in value[section]:
                _require(isinstance(item, dict) and _text(item.get('id'), 80)
                         and item['id'] not in ids, 'Invalid or duplicate memory id.')
                ids.add(item['id'])
                _require(_text(item.get('text'), 800), 'Memory text must be 1–800 characters.')
                _require(item.get('basis') in ('explicit', 'inferred'), 'Invalid memory basis.')
                for field in ('created_at', 'updated_at'):
                    datetime.fromisoformat(item[field])
                if section == 'personal':
                    _require(item.get('expires_on') is None, 'Personal memories cannot expire.')
                else:
                    _date(item.get('expires_on'))
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
                             and _text(source.get('quote'), 300), 'Invalid memory source.')
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
        today = self.now().date()
        return {section: [copy.deepcopy(item) for item in self.memory[section]
                          if section == 'personal' or _date(item['expires_on']) >= today]
                for section in ('personal', 'topics')}

    def recall(self):
        with self.lock:
            active = self._active_memory()
            facts = {section: [{key: item[key] for key in ('text', 'basis', 'updated_at', 'expires_on')}
                               for item in items] for section, items in active.items()}
            history = []
            for turn in self.recent['turns'][-self.max_history:]:
                if turn['id'] > self.memory['context_after_turn']:
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

    def next_batch(self):
        with self.lock:
            cursor = self.memory['last_processed_turn']
            # Process one exchange at a time, in order. In particular, a forget
            # request must not consume newer statements queued behind it.
            turns = [t for t in self.recent['turns'] if t['id'] > cursor][:1]
            if not turns:
                return None
            prior = [t for t in self.recent['turns']
                     if self.memory['context_after_turn'] < t['id'] < turns[0]['id']][-4:]
            return copy.deepcopy({
                'cursor': cursor, 'today': self.now().date().isoformat(),
                'default_topic_expiry': (self.now().date() + timedelta(days=30)).isoformat(),
                'memory': self._active_memory(), 'earlier_context': prior, 'new_turns': turns,
            })

    @staticmethod
    def _evidence(sources, turns):
        _require(isinstance(sources, list) and 1 <= len(sources) <= 3, 'Each change needs 1–3 new sources.')
        result, seen = [], set()
        for source in sources:
            _require(isinstance(source, dict) and set(source) == {'turn_id', 'quote'}, 'Invalid source fields.')
            ident, quote = source['turn_id'], source['quote']
            _require(type(ident) is int and ident in turns and ident not in seen,
                     'Evidence must cite distinct NEW accepted turns.')
            _require(_text(quote, 300) and quote in turns[ident]['user_speech'],
                     'Evidence quote must be exact user speech, not HAL or API text.')
            seen.add(ident)
            result.append({'turn_id': ident, 'at': turns[ident]['at'], 'quote': quote})
        return result

    def apply(self, batch, changes):
        """Validate the whole patch before committing any entry or queue cursor."""
        with self.lock:
            _require(batch['cursor'] == self.memory['last_processed_turn'], 'Stale memory update.')
            _require(isinstance(changes, dict) and set(changes) == {'forget', 'forget_evidence', 'operations'}
                     and type(changes['forget']) is bool and isinstance(changes['operations'], list)
                     and len(changes['operations']) <= 20, 'Invalid memory changes.')
            turns = {t['id']: t for t in batch['new_turns']}
            forget = changes['forget']
            if forget:
                self._evidence(changes['forget_evidence'], turns)
            else:
                _require(changes['forget_evidence'] == [], 'Unexpected forget evidence.')
            updated, events, touched = copy.deepcopy(self.memory), [], set()
            for op in changes['operations']:
                _require(isinstance(op, dict) and set(op) == {
                    'action', 'section', 'id', 'text', 'basis', 'expires_on', 'reason', 'evidence'},
                    'Invalid operation fields.')
                action, section, ident = op['action'], op['section'], op['id']
                _require(action in ('add', 'update', 'reinforce', 'delete') and
                         section in ('personal', 'topics') and isinstance(ident, str), 'Invalid operation.')
                _require(not forget or action == 'delete', 'Forget batches can only delete entries.')
                _require(_text(op['text'], 800) and _text(op['reason'], 400)
                         and op['basis'] in ('explicit', 'inferred'), 'Invalid operation text or basis.')
                sources = self._evidence(op['evidence'], turns)
                if section == 'personal':
                    _require(op['expires_on'] is None, 'Personal memory cannot expire.')
                else:
                    expiry = _date(op['expires_on'])
                    _require(action == 'delete' or expiry >= self.now().date(), 'Topic expiry is in the past.')
                existing = next((i for i in updated[section] if i['id'] == ident), None)
                if action == 'add':
                    _require(ident == '', 'New memories cannot choose their own id.')
                    _require(not any(i['text'].casefold() == op['text'].casefold()
                                     for i in updated[section]), 'Duplicate memory; update the existing entry.')
                else:
                    _require(existing is not None and ident not in touched, 'Unknown or repeated memory id.')
                    touched.add(ident)
                before = copy.deepcopy(existing)
                if action == 'delete':
                    updated[section].remove(existing)
                    after = None
                else:
                    if action == 'reinforce':
                        _require(op['text'] == existing['text'] and op['basis'] == existing['basis'],
                                 'Reinforcement cannot rewrite a fact.')
                    at = self.now().isoformat()
                    if existing is None:
                        existing = {'id': 'm_' + uuid.uuid4().hex[:12], 'created_at': at,
                                    'evidence_count': 0, 'evidence_dates': [], 'evidence': []}
                        updated[section].append(existing)
                    existing.update(text=op['text'], basis=op['basis'], expires_on=op['expires_on'], updated_at=at)
                    if action == 'update':
                        existing['evidence_count'] = 0
                        existing['evidence_dates'] = []
                    existing['evidence_count'] += len(sources)
                    dates = existing['evidence_dates'] + [s['at'][:10] for s in sources]
                    existing['evidence_dates'] = sorted(set(dates))[-12:]
                    # A correction replaces old supporting quotes as well as text.
                    evidence = [] if action == 'update' else existing['evidence']
                    existing['evidence'] = (evidence + sources)[-3:]
                    after = copy.deepcopy(existing)
                events.append({'action': action, 'section': section, 'before': before,
                               'after': after, 'reason': op['reason'], 'evidence': sources})
            for item in list(updated['topics']):
                if forget or _date(item['expires_on']) < self.now().date():
                    updated['topics'].remove(item)
                    events.append({'action': 'delete', 'section': 'topics', 'before': item, 'after': None,
                                   'reason': 'Forget request cleared topic context.' if forget else 'Topic expired.'})
            updated['last_processed_turn'] = batch['new_turns'][-1]['id']
            if forget:
                updated['context_after_turn'] = updated['last_processed_turn']
            self._validate_memory(updated)
            # Entries and cursor commit in ONE file. A crash before the recent
            # file is trimmed cannot replay reinforcement or recreate deletions.
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
