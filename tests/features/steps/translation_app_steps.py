from behave import given, when, then
import tkinter as tk
from kanji_reader.gui import TranslationApp  # Import the actual app class


@given('the user has opened the TranslationApp')
def step_impl(context):
    # Create a tkinter root window
    context.root = tk.Tk()
    context.app = TranslationApp(context.root)  # Initialize the app
    context.root.update()  # Update the Tkinter window


@when('the user enters "{input_text}" in the text box and clicks the translate button')
def step_impl(context, input_text):
    # Simulate entering text into the text box
    context.app.text_entry.delete("1.0", tk.END)
    context.app.text_entry.insert("1.0", input_text)

    # Simulate clicking the translate button
    context.app.translate_text()


@then('the translation output should display a translation of "{expected_translation}"')
def step_impl(context, expected_translation):
    # Check if the translation label contains the expected translation text
    result_text = context.app.text_label.cget("text")
    assert expected_translation in result_text, f"Expected translation to include '{expected_translation}', but got '{result_text}'"


@then('the output should include "Hiragana" and "Romaji"')
def step_impl(context):
    # Check if Hiragana and Romaji are included in the output
    result_text = context.app.text_label.cget("text")
    assert "Hiragana" in result_text, f"Expected output to include 'Hiragana', but got '{result_text}'"
    assert "Romaji" in result_text, f"Expected output to include 'Romaji', but got '{result_text}'"
