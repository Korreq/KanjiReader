import unittest
from .gui import Models
import torch


class TestModels(unittest.TestCase):
    def setUp(self):
        self.models = Models(test=True)

    #Working test, commented to speed up testing    
    '''
    def test_translate_models(self):
        text = "日本に行くのが楽しみです。"
        sentences = [
            text,
            self.models.translate(text),
            self.models.translate_small100(text),
            self.models.translate_mbart(text)
        ]

        # Compare the translations
        similarity_scores = self.models.compare_with_original(sentences)

        # Output the similarity scores between each pair of translations
        for pair, score in similarity_scores.items():
            print(f"{pair}: {score:.4f}")
    '''



    #Trash code, left for future reference
    '''
    def test_convert_kana(self):
        test_sentences = [
            ("漢字のテスト", ("かんじのてすと", "kanji no tesuto")) # Example Kanji -> Kana pair
        ]
        
        for kanji_text, (expected_hiragana, expected_romanji) in test_sentences:
            hiragana, romanji = self.converter.convert_kana(kanji_text)
            self.assertEqual(hiragana, expected_hiragana)
            self.assertEqual(romanji, expected_romanji)    

        similarity_scores = self.converter.compare_with_original(test_sentences)
        
        for key, score in similarity_scores.items():
            self.assertGreater(score, 0.9, f"Expected high similarity for {key}. Got similarity: {score}")

    '''
    '''
    def test_convert(self):

        text = "日本に行くのが楽しみです。"

        output = self.models.convert_kanji_to_kana_pykakasi(text)
        output2 = self.models.convert_kanji_to_kana_elyza(text)
        print(text)
        print(output)
        print(output2)
    
    '''
    '''
    def test_text_from_image(self):

        picture = "/home/kouht/Documents/Github/KanjiReader/files/images/test.png"
        text_mangaocr = self.models.text_from_image(picture)
        text_got = self.models.text_from_image_got(picture)
        print(text_got)
        print(text_mangaocr)
    '''

    def full_stack_run(self, ocr, translator, converter, picture):

        text = ocr(picture)
        print( "after ocr" )
        translated_text = translator(text)
        print( "after translation" )
        hiragana_text, romaji_text = converter(text)
        print( "after conversion" )

        return text, translated_text, hiragana_text, romaji_text

    def test_translation_from_image(self):

        picture = "files/images/test.png"
        text_from_picture = "どっちでもいいよそんなの！"

        ocrs = {"manga_ocr": self.models.text_from_image, "got": self.models.text_from_image_got}
        translators = {"opus_mt": self.models.translate, "small100": self.models.translate_small100, "mbart": self.models.translate_mbart}
        converters = {"elyza": self.models.convert_kanji_to_kana_elyza, "pykakasi": self.models.convert_kanji_to_kana_pykakasi}
        #converters = {"pykakasi": self.models.convert_kanji_to_kana_pykakasi}

        for ocr_name, ocr in ocrs.items():    
            for translator_name, translator in translators.items():
                for converter_name, converter in converters.items():

                    print(f"\nTesting {ocr_name} -> {translator_name} -> {converter_name}\n")

                    text, translated_text, hiragana_text, romaji_text = self.full_stack_run(ocr, translator, converter, picture)
                    
                    sentences = [text_from_picture, text, translated_text, hiragana_text, romaji_text]

                    similarity_scores = self.models.compare_with_original(sentences)


                    for sentence in sentences:
                        print(sentence)

                    for key, score in similarity_scores.items():
                        print(f"{key}: {score:.4f}")
                        #self.assertGreater(score, 0.9, f"Expected high similarity for {key}. Got similarity: {score}")
 


if __name__ == '__main__':
    unittest.main()

    