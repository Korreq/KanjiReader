# use this command to run all behave tests
# behave tests/features
# features/environment.py
import os
from kanji_reader.gui import TranslationHistory

def before_all(context):
    # Create a directory to store the history file
    context.history_dir = 'data/results'
    if not os.path.exists(context.history_dir):
        os.makedirs(context.history_dir)

def after_all(context):
    print("After all")

def before_scenario(context, scenario):
    # Create a TranslationHistory object
    context.history = TranslationHistory(filename="test_translation_history.json", max_entries=10)

def after_scenario(context, scenario):
    print("After scenario")
