import json
import os

class TranslationHistory:
    def __init__(self, results_folder="../files/results", filename="translation_history.json", max_entries=1000):
        self.results_folder = os.path.join(os.path.dirname(__file__), results_folder)
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

        self.filepath = os.path.join(self.results_folder, filename)
        self.max_entries = max_entries

    def save_translation(self, input_text, translated_text):
        # If the file exists, load its data; otherwise, start with an empty list
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = []

        # Add the new translation entry to the history
        history.append({"input_text": input_text, "translated_text": translated_text})

        # If the number of entries exceeds the limit, remove the oldest one
        if len(history) > self.max_entries:
            history.pop(0)  # Remove the first (oldest) entry

        # Save the updated history back to the file
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)