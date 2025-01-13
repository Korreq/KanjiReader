import unittest
from .models import Models


class TestModels(unittest.TestCase):
    def setUp(self):
        self.models = Models()

    def full_stack_run(self, ocr, translator_name, converter, picture):

        text = ocr(picture)
        translated_text = self.models.translate_text(text, translator_name)
        hiragana_text, romaji_text = converter(text)
      
        return text, translated_text, hiragana_text, romaji_text

    def test_translation_from_image(self):

        picture = "files/images/test.png"
        text_from_picture = "どっちでもいいよそんなの！"

        ocrs = {"manga_ocr": self.models.text_from_image_manga_ocr, "got": self.models.text_from_image_got}
        #ocrs = {"manga_ocr": self.models.text_from_image_manga_ocr}

        translators = {"opus_mt": self.models.translate_text, "small100": self.models.translate_text, "mbart": self.models.translate_text}
        #converters = {"elyza": self.models.convert_kanji_to_kana_elyza, "pykakasi": self.models.convert_kanji_to_kana_pykakasi}
        converters = {"pykakasi": self.models.convert_kanji_to_kana_pykakasi}

        for ocr_name, ocr in ocrs.items():    
            for translator_name, translator in translators.items():
                for converter_name, converter in converters.items():

                    print(f"\nTesting {ocr_name} -> {translator_name} -> {converter_name}\n")

                    text, translated_text, hiragana_text, romaji_text = self.full_stack_run(ocr, translator_name, converter, picture)
                    
                    sentences = [text_from_picture, text, translated_text, hiragana_text, romaji_text]

                    similarity_scores = self.models.compare_with_original(sentences)

                    for sentence in sentences:
                        print(sentence)

                    for key, score in similarity_scores.items():
                        print(f"{key}: {score:.4f}")


if __name__ == '__main__':
    unittest.main()

    