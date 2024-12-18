# history.py
import json
import os

class TranslationHistory:
    def __init__(self, results_folder="../files/results"):
        print( os.pardir )
        self.results_folder = os.path.join(os.path.dirname(__file__), results_folder)
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

    def save_translation(self, input_text, translated_text):

        filename = f"{len(os.listdir(self.results_folder)) + 1}.json"
        filepath = os.path.join(self.results_folder, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"input_text": input_text, "translated_text": translated_text}, f, indent=4, ensure_ascii=False)