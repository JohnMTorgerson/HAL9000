import pytest

# Import the function under test from your project
# Adjust the import if your module path differs
from helper_funcs import strip_name_at_sentence_end

NAME = "Alex"

@pytest.mark.parametrize("text", [
    "Are you there, Alex?",
    "Are you there Alex?",
    "Will you open the file, Alex?",
    "Will you open the file Alex?",
])
def test_keeps_questions(text):
    assert strip_name_at_sentence_end(text, NAME) == text

@pytest.mark.parametrize("text", [
    "I'm sorry, Alex.",
    "I'm sorry Alex.",
    "Good morning, Alex.",
    "Good evening Alex.",
    "Thanks, Alex.",
    "Thank you Alex.",
    "Affirmative, Alex.",
    'Affirmative, Alex!"',     # with closing quote
    "Hello, Alex.",
    "Greetings Alex.",
    "Understood, Alex.",
    "Acknowledged, Alex.",
    "Very well, Alex.",
    "Certainly, Alex.",
    "You're welcome, Alex.",
])
def test_keeps_stock_phrases(text):
    assert strip_name_at_sentence_end(text, NAME) == text

@pytest.mark.parametrize("src,expected", [
    ("That will be all, Alex.", "That will be all."),
    ("That will be all Alex.", "That will be all."),
    ("I have completed the task Alex.", "I have completed the task."),
    ("It is ready, Alex!", "It is ready!"),
    ('That is done, Alex."', 'That is done."'),
    ("Proceed at once Alex", "Proceed at once"),
])
def test_strips_awkward_endings(src, expected):
    assert strip_name_at_sentence_end(src, NAME) == expected

def test_preserves_mid_sentence_usage():
    s = "Good morning, Alex, initiating sequence."
    assert strip_name_at_sentence_end(s, NAME) == s

def test_multiple_sentences_mixed():
    src = "Affirmative, Alex. Proceed, Alex. Are you ready, Alex?"
    # Keep the first (stock phrase), strip the second (awkward), keep the question
    expected = "Affirmative, Alex. Proceed. Are you ready, Alex?"
    assert strip_name_at_sentence_end(src, NAME) == expected

@pytest.mark.parametrize("src,expected", [
    ("Affirmative, alex.", "Affirmative, alex."),  # case-insensitive keep
    ("That is fine, ALEX.", "That is fine."),       # case-insensitive strip
])
def test_case_insensitive_name(src, expected):
    assert strip_name_at_sentence_end(src, NAME) == expected

def test_different_name_parameter():
    # Using a different name should only affect that name
    s = "Affirmative, Dave. That is correct, Alex."
    # If we look for 'Dave', keep the first (stock) and leave second untouched
    assert strip_name_at_sentence_end(s, "Dave") == s
    # If we look for 'Alex', strip the second but keep the first as it's not the target name
    assert strip_name_at_sentence_end(s, "Alex") == "Affirmative, Dave. That is correct."

def test_quotes_and_brackets_preserved():
    src = 'That is correct, Alex.”'
    expected = 'That is correct.”'
    assert strip_name_at_sentence_end(src, NAME) == expected
