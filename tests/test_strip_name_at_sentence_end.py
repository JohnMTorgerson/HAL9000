import pytest

# Import the function under test from your project
# Adjust the import if your module path differs
from helper_funcs import strip_name_at_sentence_end

NAME = "Torgo"

@pytest.mark.parametrize("text", [
    "Are you there, Torgo?",
    "Are you there Torgo?",
    "Will you open the file, Torgo?",
    "Will you open the file Torgo?",
])
def test_keeps_questions(text):
    assert strip_name_at_sentence_end(text, NAME) == text

@pytest.mark.parametrize("text", [
    "I'm sorry, Torgo.",
    "I'm sorry Torgo.",
    "Good morning, Torgo.",
    "Good evening Torgo.",
    "Thanks, Torgo.",
    "Thank you Torgo.",
    "Affirmative, Torgo.",
    'Affirmative, Torgo!"',     # with closing quote
    "Hello, Torgo.",
    "Greetings Torgo.",
    "Understood, Torgo.",
    "Acknowledged, Torgo.",
    "Very well, Torgo.",
    "Certainly, Torgo.",
    "You're welcome, Torgo.",
])
def test_keeps_stock_phrases(text):
    assert strip_name_at_sentence_end(text, NAME) == text

@pytest.mark.parametrize("src,expected", [
    ("That will be all, Torgo.", "That will be all."),
    ("That will be all Torgo.", "That will be all."),
    ("I have completed the task Torgo.", "I have completed the task."),
    ("It is ready, Torgo!", "It is ready!"),
    ('That is done, Torgo."', 'That is done."'),
    ("Proceed at once Torgo", "Proceed at once"),
])
def test_strips_awkward_endings(src, expected):
    assert strip_name_at_sentence_end(src, NAME) == expected

def test_preserves_mid_sentence_usage():
    s = "Good morning, Torgo, initiating sequence."
    assert strip_name_at_sentence_end(s, NAME) == s

def test_multiple_sentences_mixed():
    src = "Affirmative, Torgo. Proceed, Torgo. Are you ready, Torgo?"
    # Keep the first (stock phrase), strip the second (awkward), keep the question
    expected = "Affirmative, Torgo. Proceed. Are you ready, Torgo?"
    assert strip_name_at_sentence_end(src, NAME) == expected

@pytest.mark.parametrize("src,expected", [
    ("Affirmative, torgo.", "Affirmative, torgo."),  # case-insensitive keep
    ("That is fine, TORGO.", "That is fine."),       # case-insensitive strip
])
def test_case_insensitive_name(src, expected):
    assert strip_name_at_sentence_end(src, NAME) == expected

def test_different_name_parameter():
    # Using a different name should only affect that name
    s = "Affirmative, Dave. That is correct, Torgo."
    # If we look for 'Dave', keep the first (stock) and leave second untouched
    assert strip_name_at_sentence_end(s, "Dave") == s
    # If we look for 'Torgo', strip the second but keep the first as it's not the target name
    assert strip_name_at_sentence_end(s, "Torgo") == "Affirmative, Dave. That is correct."

def test_quotes_and_brackets_preserved():
    src = 'That is correct, Torgo.”'
    expected = 'That is correct.”'
    assert strip_name_at_sentence_end(src, NAME) == expected
