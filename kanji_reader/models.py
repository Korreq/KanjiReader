from transformers import (
    MarianMTModel, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline,
    VisionEncoderDecoderModel, AutoFeatureExtractor,
    BlipForConditionalGeneration, BlipProcessor, AutoModel,
    M2M100ForConditionalGeneration, M2M100Tokenizer, T5Tokenizer, T5ForConditionalGeneration,
    MBartForConditionalGeneration, MBart50TokenizerFast
)

from sklearn.metrics.pairwise import cosine_similarity
import PIL.Image
import cv2
import pytesseract
import torch
import os

'''

    paraphrase-multilingual-MiniLM-L12-v2 use for models semantic comparision

'''



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
        """Translate Japanese to English using Helsinki-NLP/opus-mt-ja-en."""
        model_name = "Helsinki-NLP/opus-mt-ja-en"
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        batch = tokenizer([text], return_tensors="pt")
        generated_ids = model.generate(**batch)
        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return output

    def translate_small100(self, text: str) -> str:
        """Translate Japanese to English using alirezamsh/small100."""
        model_name = "alirezamsh/small100"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, tgt_lang="en")

        encode = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(**encode)
        output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return output

    def translate_mbart(self, text: str) -> str:
        """Translate Japanese to English using facebook/mbart-large-50-many-to-many-mmt."""
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

        tokenizer.src_lang = "ja_XX" 
        tokenizer.tgt_lang = "en_XX" 

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=400)
        # Generate translation
        with torch.no_grad():
            translated_tokens = model.generate(**inputs)
        # Decode the translated tokens to text
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text

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

    # ================================
    # Semantic Comparision
    # ================================

    def compare_semantics(self, sentences):
        model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)

        sentence_embeddings = self.mean_pooling( model_output, encoded_input['attention_mask'] )

        sentence_embeddings = sentence_embeddings / sentence_embeddings.norm(p=2, dim=1, keepdim=True)

        return sentence_embeddings

    def mean_pooling(self, model_output, attention_mask):

        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def compare_embeddings(self, embedding1, embedding2):
        # Reshape embeddings to 2D (1 sample, N dimensions) to work with cosine_similarity
        embedding1 = embedding1.unsqueeze(0) if embedding1.dim() == 1 else embedding1
        embedding2 = embedding2.unsqueeze(0) if embedding2.dim() == 1 else embedding2

        # Compute cosine similarity between two sentence embeddings
        cos_sim = cosine_similarity(embedding1.cpu().numpy(), embedding2.cpu().numpy())
        return cos_sim[0][0]
    
    def compare_with_original(self, sentences):
        # Get the sentence embeddings for all translations
        embeddings = self.compare_semantics(sentences)

        # Compare all pairs of translations (cosine similarity between each pair)
        similarity_scores = {}
        for i, translated_sentence in enumerate(sentences[1:]):
            similarity_score = self.compare_embeddings(embeddings[0], embeddings[i + 1])  # Compare original (index 0) with each translation
            similarity_scores[f"Original vs Translation {i+1}"] = similarity_score

        return similarity_scores