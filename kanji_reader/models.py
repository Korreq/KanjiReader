from transformers import (
    MarianMTModel, AutoTokenizer, AutoModelForSeq2SeqLM,
    VisionEncoderDecoderModel, AutoFeatureExtractor,
    BlipForConditionalGeneration, BlipProcessor,
    M2M100ForConditionalGeneration, M2M100Tokenizer, T5Tokenizer, T5ForConditionalGeneration
)
import PIL.Image
import cv2
import pytesseract
import torch
import os


class Models:
    """Handles all model-related tasks, including translation, OCR, and Kanji to Kana conversion."""

    def __init__(self):
        # Initialize translation models
        self._translator = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
        self._tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ja-en")

        # Initialize OCR models
        self._ocr_model = VisionEncoderDecoderModel.from_pretrained("kha-white/manga-ocr-base")
        self._ocr_feature_extractor = AutoFeatureExtractor.from_pretrained("kha-white/manga-ocr-base")
        self._blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        self._blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")

        # Tesseract for generic OCR
        self._got_model = pytesseract

    # ================================
    # Translation Methods
    # ================================

    def translate(self, text: str) -> str:
        """Translate Japanese to English using MarianMT."""
        return self._translate_model(text, "Helsinki-NLP/opus-mt-ja-en")

    def translate_nllb(self, text: str) -> str:
        """Translate using the NLLB model."""
        return self._translate_model(text, "facebook/nllb-200-distilled-600M", m2m_model=True)

    def translate_mbart(self, text: str) -> str:
        """Translate using the mBART model."""
        return self._translate_model(text, "facebook/mbart-large-50-many-to-many-mmt", m2m_model=True)

    def _translate_model(self, text: str, model_name: str, m2m_model=False) -> str:
        """Helper method for translation with different models."""
        if m2m_model:
            model = M2M100ForConditionalGeneration.from_pretrained(model_name)
            tokenizer = M2M100Tokenizer.from_pretrained(model_name)
            inputs = tokenizer([text], return_tensors="pt", src_lang="ja")
            outputs = model.generate(inputs["input_ids"], forced_bos_token_id=tokenizer.lang_code_to_id["en"])
        else:
            model = MarianMTModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            batch = tokenizer([text], return_tensors="pt")
            generated_ids = model.generate(**batch)
            outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return outputs[0]

    # ================================
    # OCR Methods
    # ================================

    def text_from_image(self, picture: str) -> str:
        """Extract text from an image using Manga OCR."""
        image = PIL.Image.open(picture)
        inputs = self._ocr_feature_extractor(images=image, return_tensors="pt")
        outputs = self._ocr_model.generate(**inputs)
        tokenizer = AutoTokenizer.from_pretrained("kha-white/manga-ocr-base")
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def text_from_image_blip(self, picture: str) -> str:
        """Extract text using the BLIP model."""
        return self._extract_text_from_image_blip(picture)

    def text_from_image_got(self, picture: str) -> str:
        """Extract text using Tesseract OCR."""
        return self._extract_text_from_image_got(picture)

    def _extract_text_from_image_blip(self, picture: str) -> str:
        """Helper method for extracting text with the BLIP model."""
        image = PIL.Image.open(picture)
        inputs = self._blip_processor(images=image, return_tensors="pt")
        outputs = self._blip_model.generate(**inputs)
        caption = self._blip_processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return caption

    def _extract_text_from_image_got(self, picture: str) -> str:
        """Helper method for extracting text using Tesseract OCR."""
        image = cv2.imread(picture)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        temp_file = "temp.png"
        cv2.imwrite(temp_file, thresh)
        text = pytesseract.image_to_string(PIL.Image.open(temp_file))
        os.remove(temp_file)
        return text

    # ================================
    # Kanji to Kana Conversion Methods
    # ================================

    def convert_kana(self, text: str):
        """Convert Kanji to Hiragana and Romanji."""
        return self._convert_kana_with_model(text, "Miwa-Keita/zenz-v2-gguf")

    def convert_kana_byt5(self, text: str):
        """Convert Kanji to Hiragana and Romanji using ByT5."""
        return self._convert_kana_with_model_byt5(text)

    def convert_kana_gemma(self, text: str):
        """Convert Kanji to Hiragana and Romanji using Gemma."""
        return self._convert_kana_with_model_gemma(text)

    def _convert_kana_with_model(self, text: str, model_name: str):
        """Helper method for Kanji to Kana conversion."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        hiragana_text = tokenizer.convert_ids_to_tokens(outputs[0].argmax(-1))[0]
        romanji_text = hiragana_text.translate(str.maketrans('', '', ''))  # Remove non-roman characters
        return hiragana_text, romanji_text

    def _convert_kana_with_model_byt5(self, text: str):
        """Helper method for Kanji to Kana conversion using ByT5."""
        return self._convert_kana_with_t5("google/byt5-small", text)

    def _convert_kana_with_model_gemma(self, text: str):
        """Helper method for Kanji to Kana conversion using Gemma."""
        return self._convert_kana_with_t5("google/gemma-2-2b-jpn-it", text)

    def _convert_kana_with_t5(self, model_name: str, text: str):
        """General method for Kanji to Kana conversion using T5-based models."""
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)

        hiragana_text = self._generate_kana_output(tokenizer, model, "kanji_to_hiragana", text)
        romanji_text = self._generate_kana_output(tokenizer, model, "kanji_to_romanji", text)

        return hiragana_text, romanji_text

    def _generate_kana_output(self, tokenizer, model, task: str, text: str):
        """Generate Kana (Hiragana or Romanji) using the provided model."""
        input_ids = tokenizer.encode(f"{task}: {text}", return_tensors="pt")
        outputs = model.generate(input_ids)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
