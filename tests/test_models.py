# Run using this command 
# python -m tests.test_models


import unittest
import csv
from kanji_reader.models import Models


class TestModels(unittest.TestCase):
    def setUp(self):
        self.models = Models()

    def full_stack_run(self, ocr, translator_name, converter, picture):

        text = ocr(picture)
        translated_text = self.models.translate_text(text, translator_name)
        hiragana_text, romaji_text = converter(text)
      
        return text, translated_text, hiragana_text, romaji_text

    def test_translation_from_image(self):
        pictures = {
            "俺が外国でひろってしまってなんだかわからないうちに": "data/images/test/test1.png",
            "おわびに花火大会で色々かってやるぞ！？わたあめとか！たこやきとか！きんぎょすくい！ああ　いいぞ　金魚すくいな！？": "data/images/test/test2.png",
            "学校のパソコンで作ってプリントアウトしてそれにおイモのハンコを押したの": "data/images/test/test3.png",
            "よしじゃあゾウ見ながら弁当食うかー　そこの木影で": "data/images/test/test4.png",
            "そーゆー大そうなのはいなかったとおもうなぁ": "data/images/test/test5.png",
            "あれ？ひまわりって太陽の方むくんじゃなかったっけ？": "data/images/test/test6.png",
        }
        ocrs = {"manga_ocr": self.models.text_from_image_manga_ocr, "got": self.models.text_from_image_got}
        #ocrs = {"manga_ocr": self.models.text_from_image_manga_ocr}
        translators = {"opus_mt": self.models.translate_text, "small100": self.models.translate_text, "mbart": self.models.translate_text}
        #converters = {"elyza": self.models.convert_kanji_to_kana_elyza, "pykakasi": self.models.convert_kanji_to_kana_pykakasi}
        converters = {"pykakasi": self.models.convert_kanji_to_kana_pykakasi}

        # Open the CSV file to save the results
        with open('data/results/test_results.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
        
            # Write the header row
            writer.writerow([
                'Original Text', 'OCR Method', 'Translator', 'Converter', 'OCR Text', 'Translated Text', 
                'Hiragana Text', 'Romaji Text', 'Similarity Score for OCR Text', 
                'Similarity Score for Translated Text', 'Similarity Score for Hiragana Text', 'Similarity Score for Romaji Text'
            ])

            # Iterate through all combinations of Pictures, OCRs, Translators, and Converters
            for text_from_picture, picture in pictures.items():
                for ocr_name, ocr in ocrs.items():    
                    for translator_name, translator in translators.items():
                        for converter_name, converter in converters.items():

                            # Run the full stack
                            text, translated_text, hiragana_text, romaji_text = self.full_stack_run(ocr, translator_name, converter, picture)
                    
                            sentences = [text_from_picture, text, translated_text, hiragana_text, romaji_text]
                            similarity_scores = self.models.compare_with_original(sentences)

                            # Write the results to the CSV file
                            row = [
                                text_from_picture, 
                                ocr_name, 
                                translator_name, 
                                converter_name, 
                                text, 
                                translated_text, 
                                hiragana_text, 
                                romaji_text
                            ]
                    
                            # Add similarity scores to the row
                            for key, score in similarity_scores.items():
                                row.append(f"{score:.4f}")

                            # Write the row to the CSV file
                            writer.writerow(row)


if __name__ == '__main__':
    unittest.main()

    