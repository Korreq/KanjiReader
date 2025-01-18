# features/steps/test_saving_history.py
from behave import given, when, then
from kanji_reader.gui import TranslationHistory
import json

@given('I have a history with a record limit of {limit} entries')
def step_impl(context, limit):
    context.history = TranslationHistory(filename="test_translation_history.json", max_entries=int(limit))

@given('the history file already has {count} entries')
def step_impl(context, count):
    with open(context.history.filepath, 'w') as f:
        history = [{'input_text': f'Entry {i}'} for iq in range(int(count))]
        json.dump(history, f)

@when('I save the history')
def step_impl(context):
    context.history.save_translation('input_text', 'input_type', 'translated_text', 'hiragana_text', 'romaji_text')

@then('the oldest entry should be deleted')
def step_impl(context):
    with open(context.history.filepath, 'r') as f:
        history = json.load(f)
        assert len(history) == context.history.max_entries
        assert history[0]['input_text'] != 'Entry 0'

@then('the new entry should be saved')
def step_impl(context):
    with open(context.history.filepath, 'r') as f:
        history = json.load(f)
        assert len(history) == context.history.max_entries
        assert history[-1]['input_text'] == 'input_text'

@then('the history file should have {count} entries')
def step_impl(context, count):
    with open(context.history.filepath, 'r') as f:
        history = json.load(f)
        assert len(history) == int(count)