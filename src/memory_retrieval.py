"""Deterministic, local recall from memory text and LLM-authored search tags.

The caller supplies active memories and accepted recent exchanges. Retrieval
never edits the archive, calls a model, or treats a missing match as evidence
that a memory does not exist. Its character budget is only a token estimate.
"""
import copy
from datetime import datetime, timezone
import json
import re
import unicodedata


SECTIONS = ('personal', 'hal', 'topics')
CONTEXT_TURNS = 3
_COMPACT_FIELDS = ('id', 'text', 'tags', 'basis', 'created_at', 'updated_at', 'expires_on', 'retention')
_STOPWORDS = frozenset('''
    a an and are as at be been being but by can could did do does doing done
    for from had has have having he her hers herself him himself his how i if
    in into is it its itself just me mine more most my myself no nor not of
    off on once only or other our ours ourselves out over own same she should
    so some than that the their theirs them themselves then there these they
    this those through to too under until up us very was we were what when
    where which while who whom why will with would you your yours yourself
    yourselves also any anything each either even ever every everything few
    get gets getting got here much now quite really say says said something
    still such tell telling think thinking thought want wanted well yes yet
    please hal user assistant remember remembered recall know knows knew
    like liked let lets okay ok again discuss discussion talk talking about
'''.split())


def _singular(word):
    """Small plural normalization, deliberately not a general-purpose stemmer."""
    if len(word) > 4 and word.endswith('ies'):
        return word[:-3] + 'y'
    if len(word) > 4 and word.endswith(('sses', 'shes', 'ches', 'xes', 'zes')):
        return word[:-2]
    if len(word) > 3 and word.endswith('s') and not word.endswith(('ss', 'us', 'is')):
        return word[:-1]
    return word


def _terms(text):
    normalized = unicodedata.normalize('NFKD', str(text or '').casefold())
    normalized = ''.join(c for c in normalized if not unicodedata.combining(c))
    normalized = normalized.replace('\u2019', "'").replace('\u2018', "'")
    normalized = re.sub(r"\b([^\W_]+)'s\b", r'\1', normalized)
    words = re.findall(r'[^\W_]+', normalized)
    return {_singular(word) for word in words
            if len(word) > 1 and word not in _STOPWORDS
            and _singular(word) not in _STOPWORDS}


def compact_entry(item):
    """Prompt-facing fields; evidence quotes stay in the local archive."""
    return {field: copy.deepcopy(item[field]) for field in _COMPACT_FIELDS if field in item}


def _browse_sections(query):
    """A few explicit archive/profile questions; never infer intent from a word."""
    phrase = ' '.join(re.findall(r'[^\W_]+', str(query or '').casefold()))
    phrase = re.sub(r'^(?:hey hal |hal |please )', '', phrase)
    phrase = re.sub(r' please$', '', phrase)
    if re.fullmatch(r'what do you (?:remember|know) about me', phrase):
        return ('personal',)
    if re.fullmatch(r'tell me about yourself|what do you (?:remember|know) about yourself|'
                    r'what are your (?:interests|views|opinions|beliefs|preferences|hobbies)', phrase):
        return ('hal',)
    if re.fullmatch(r'what do you remember|what memories do you have', phrase):
        return SECTIONS
    return ()


def _updated_time(item):
    try:
        value = datetime.fromisoformat(item.get('updated_at', ''))
        return value.replace(tzinfo=value.tzinfo or timezone.utc).timestamp()
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _serialized_size(sections):
    return len(json.dumps({section: [compact_entry(item) for item in sections[section]]
                           for section in SECTIONS}, ensure_ascii=False))


def _context(recent_turns):
    weights, details = {}, []
    for age, turn in enumerate(reversed(recent_turns[-CONTEXT_TURNS:])):
        recency = 0.5 ** age
        turn_detail = {'turn_id': turn.get('id'), 'age': age}
        for field, speaker_weight in (('user_speech', 1.0), ('assistant_reply', 0.75)):
            terms = _terms(turn.get(field, ''))
            turn_detail[field + '_terms'] = sorted(terms)
            for term in terms:
                # Repetition should not swamp a new topic; keep the strongest
                # recent occurrence of a word instead of counting every mention.
                weights[term] = max(weights.get(term, 0.0), recency * speaker_weight)
        details.append(turn_detail)
    return weights, details


def retrieve(memory, query, recent_turns, *, token_budget=3000):
    """Return (selected full records, JSON-safe retrieval diagnostics).

    Current-query matches precede context-only matches, using OR rather than
    requiring every query word. Tags carry broader associations supplied by
    the background updater; no hard-coded football/team taxonomy is needed.
    Whole records are packed by relevance into approximately four characters
    per token. Oversized records are logged and skipped, never truncated.
    Explicit general recall/profile questions can fill remaining space from
    the requested sections, newest first, without claiming word matches.
    """
    if type(token_budget) is not int or token_budget < 0:
        raise ValueError('Memory retrieval token_budget must be a nonnegative integer.')
    query_terms = _terms(query)
    browse_sections = _browse_sections(query)
    context_weights, context_details = _context(recent_turns or [])
    context_terms = set(context_weights)
    candidates = []
    available_count = 0
    for section_number, section in enumerate(SECTIONS):
        for item in memory.get(section, []):
            available_count += 1
            tags = item.get('tags', [])
            tag_terms = set().union(*(_terms(tag) for tag in tags)) if tags else set()
            fields = {'text': _terms(item.get('text', '')), 'tags': tag_terms}
            matches = {source: {field: sorted(terms & needles)
                                for field, terms in fields.items()}
                       for source, needles in (('query', query_terms), ('context', context_terms))}
            query_matched = set(matches['query']['text']) | set(matches['query']['tags'])
            context_matched = set(matches['context']['text']) | set(matches['context']['tags'])
            matched = bool(query_matched or context_matched)
            if not matched and section not in browse_sections:
                continue
            query_score = sum(len(matches['query'][field]) * weight
                              for field, weight in (('text', 20), ('tags', 40)))
            context_score = sum(sum(context_weights[term] for term in matches['context'][field]) * weight
                                for field, weight in (('text', 1), ('tags', 2)))
            score = query_score + context_score
            retrieval_reason = 'query' if query_matched else 'context' if context_matched else 'browse'
            entry_characters = len(json.dumps(compact_entry(item), ensure_ascii=False))
            detail = {
                'section': section, 'id': item.get('id'), 'text': item.get('text', ''),
                'tags': copy.deepcopy(tags), 'score': score,
                'query_score': query_score, 'context_score': context_score,
                'reason': retrieval_reason, 'retrieval_reason': retrieval_reason,
                'matched': matches,
                'matched_tags': [tag for tag in tags
                                 if _terms(tag) & (query_terms | context_terms)],
                'estimated_tokens': (entry_characters + 3) // 4,
            }
            rank = (-bool(query_matched), -score, -len(query_matched),
                    -len(context_matched), -_updated_time(item) if not matched else 0.0, section_number,
                    str(item.get('id', '')), item.get('text', ''))
            candidates.append((rank, section, item, detail, entry_characters))

    candidates.sort(key=lambda candidate: candidate[0])
    selected = {section: [] for section in SECTIONS}
    selected_details, omitted = [], []
    max_characters = token_budget * 4
    serialized_characters = _serialized_size(selected)
    for _, section, item, detail, entry_characters in candidates:
        # json.dumps uses a comma and space between adjacent list entries.
        size = serialized_characters + entry_characters + (2 if selected[section] else 0)
        if size > max_characters:
            omitted.append(dict(detail, reason='token_budget',
                                characters_needed=size,
                                remaining_characters=max(0, max_characters - serialized_characters)))
        else:
            selected[section].append(item)
            serialized_characters = size
            selected_details.append(detail)

    browse_only_count = sum(detail['retrieval_reason'] == 'browse' for _, _, _, detail, _ in candidates)
    matched_count = len(candidates) - browse_only_count
    diagnostics = {
        'query': query or '', 'query_terms': sorted(query_terms),
        'browse_sections': list(browse_sections),
        'context_terms': sorted(context_terms), 'context_turns': context_details,
        'token_budget': token_budget, 'character_budget': max_characters,
        'serialized_characters': serialized_characters,
        'estimated_tokens': (serialized_characters + 3) // 4,
        'available_count': available_count, 'matched_count': matched_count,
        'unmatched_count': available_count - matched_count,
        'candidate_count': len(candidates), 'browse_only_count': browse_only_count,
        'selected_count': len(selected_details), 'omitted_count': len(omitted),
        'selected': selected_details, 'omitted': omitted,
    }
    return copy.deepcopy(selected), diagnostics
