from .kana_conversion_models import KanaConversionModels
from .ocr_models import OCRModels
from .semantic_comparison_models import SemanticComparisonModels
from .translation_models import TranslationModels

class Models:
    """Handles all model-related tasks, including translation, OCR, Kanji to Kana conversion, and semantic comparison."""

    def __init__(self):
        self.kana_conversion_models = KanaConversionModels()
        self.ocr_models = OCRModels()
        self.semantic_comparison_models = SemanticComparisonModels()
        self.translation_models = TranslationModels()

    def translate_text(self, text, model_name):
        return self.translation_models.translate_text(text, model_name)

    def text_from_image_manga_ocr(self, image_path):
        return self.ocr_models.text_from_image_manga_ocr(image_path)

    def text_from_image_got( self, image_path):
        return self.ocr_models.text_from_image_got(image_path)

    def convert_kanji_to_kana_elyza(self, text):
        return self.kana_conversion_models.convert_kanji_to_kana_elyza(text)

    def convert_kanji_to_kana_pykakasi( self, text):
        return self.kana_conversion_models.convert_kanji_to_kana_pykakasi(text)

    def compare_semantics(self, sentences):
        return self.semantic_comparison_models.compare_semantics(sentences)

    def compare_with_original(self, sentences):
        return self.semantic_comparison_models.compare_with_original(sentences)