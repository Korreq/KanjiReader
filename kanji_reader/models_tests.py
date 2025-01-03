import unittest
from .gui import Models
import torch


class TestModels(unittest.TestCase):
    def setUp(self):
        self.models = Models()

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
   
    def test_convert(self):

        text = "日本に行くのが楽しみです。"

        output = self.models.convert_kanji_to_kana_pykakasi(text)
        print(text)
        print(output)
    
    '''
    def test_text_from_image(self):
        picture = "/home/yoga/Documents/Github/KanjiReader/files/images/test2.png"
        #text_mangaocr = self.models.text_from_image(picture)
        #text_blip = self.models.text_from_image_blip(picture)
        text_got = self.models.text_from_image_got(picture)
        print(text_got)
    '''

if __name__ == '__main__':
    unittest.main()

    