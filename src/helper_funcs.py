import re
import spacy
nlp = spacy.load("en_core_web_sm")

# helper function to determine if a query looks like a wikipedia question...
# this is just a fallback if HAL says it doesn't know something
# (ideally, the LLM should decide on its own when to use wikipedia, but it doesn't always)
# if it looks like something wikipedia might know, we'll force a wikipedia look up
def looks_factual(query: str) -> bool:
    FACTUAL_TRIGGERS = [
        r"\bwho\b",
        r"\bwhen\b",
        r"\bwhere\b",
        r"\bwhat\b",
        r"\bwhich\b",
        r"\bhow (old|many|long|far|tall|deep|wide)\b",
        r"\bwhat year\b",
    ]

    query = query.lower()
    return any(re.search(p, query) for p in FACTUAL_TRIGGERS)

# helper function to extract named entities from user input to decide if we should do a wikipedia lookup
def extract_named_entities(user_input: str):
    """
    Check if the input contains relevant named entities and return them.
    Used to decide if we should do a Wikipedia lookup.
    """
    doc = nlp(user_input)
    entities = []
    allowed_types = ["PERSON", "ORG", "WORK_OF_ART", "EVENT", "GPE", "LOC"]
    for ent in doc.ents:
        if allowed_types is None or ent.label_ in allowed_types:
            entities.append(ent.text)
    return entities

# strip user name at the end of (some) sentences on HAL's output
def strip_name_at_sentence_end(text: str, name: str = "Torgo") -> str:
    """
    Remove trailing direct-address (e.g., ", Torgo." or " Torgo.") at the end of sentences
    except when the sentence is a question or matches common/natural stock phrases.
    Preserves original spacing between sentences.
    """
    name_esc = re.escape(name)

    # Include any trailing whitespace in each sentence chunk so we can re-join without losing spaces.
    sentence_re = re.compile(r'[^.!?]+(?:[.!?](?:["\'”’)\]]*)\s*|\Z)', re.DOTALL)

    # Stock phrases that are natural with direct address at the end
    exception_re = re.compile(
        r"(?:"
        r"i'm sorry|"
        r"good (?:morning|afternoon|evening|night)|"
        r"hello|hi|greetings|"
        r"thanks|thank you|"
        r"affirmative|"
        r"understood|acknowledged|very well|certainly|"
        r"you're welcome"
        r")\Z",
        re.IGNORECASE,
    )

    # Trailing “, name” or “ name”, optionally followed by punctuation/quotes and (important) trailing whitespace
    tail_re = re.compile(
        rf'(?:,\s*|\s+){name_esc}\s*'        # comma/space then the name
        rf'(?P<punct>[.!?])?'                # optional sentence-ending punctuation
        rf'(?P<trail>["\'”’)\]]*)'           # optional closing quotes/brackets
        rf'(?P<ws>\s*)$'                     # capture trailing whitespace to preserve it
        , re.IGNORECASE,
    )

    out = []
    for s in sentence_re.findall(text):
        m = tail_re.search(s)
        if not m:
            out.append(s)
            continue

        punct = m.group('punct') or ''

        # Allow any question that ends with the name
        if punct == '?':
            out.append(s)
            continue

        pre = s[:m.start()]
        # Remove the comma/space that belonged to the direct address
        pre_clean = re.sub(r'[,\s]+$', '', pre.strip())

        # Keep known natural phrases like "I'm sorry, Torgo."
        if exception_re.search(pre_clean):
            out.append(s)
            continue

        # Otherwise strip the name; preserve punctuation, quotes, and original trailing whitespace
        new_s = pre_clean + (punct or '') + (m.group('trail') or '') + (m.group('ws') or '')
        out.append(new_s)

    return "".join(out)
