from transformers import AutoTokenizer, MarianMTModel
import PIL.Image
from manga_ocr import MangaOcr
import pykakasi

class Models:
    """Class for handling all the models and their respective tasks."""

    def __init__(self):
        self._translator = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
        self._tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
        self._kana_converter = pykakasi.kakasi()
        self._ocr = MangaOcr()

    def translate(self, text: str) -> str:
        """Translate a given text from Japanese to English."""
        batch = self._tokenizer([text], return_tensors="pt")
        generated_ids = self._translator.generate(**batch)
        return self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def text_from_image(self, picture: str) -> str:
        """Extract text from an image."""
        return self._ocr(PIL.Image.open(picture).convert("RGB"))

    def convert_kana(self, text: str) -> dict:
        """Convert a given text from kanji to kana."""
        return self._kana_converter.convert(text)


        