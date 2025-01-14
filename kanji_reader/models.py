from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, 
    M2M100ForConditionalGeneration, M2M100Tokenizer,
    MBartForConditionalGeneration, MBart50TokenizerFast,
    VisionEncoderDecoderModel, ViTImageProcessor,
    AutoModel, AutoModelForCausalLM,  
)
from sklearn.metrics.pairwise import cosine_similarity
import PIL.Image
import torch
import pykakasi


class Models:
    """Handles all model-related tasks, including translation, OCR, Kanji to Kana conversion, and semantic comparison."""
    
    def __init__(self):
        # Initialize all models and tokenizers to None
        self.models = {
            'translator': {
                'opus_mt': (None, None),
                'small100': (None, None),
                'mbart': (None, None),
            },
            'ocr': {
                'manga_ocr': (None, None, None),  # Store image processor, model, tokenizer for Manga OCR
                'got': (None, None),
            },
            'kana': {
                'elyza': (None, None),
                'pykakasi': None,
            },
            'similarity': {
                'minilm': (None, None),
            }
        }

    def _load_model(self, category, model_name, model_class, tokenizer_class, *args, **kwargs):
        """Helper function to load models lazily."""
        if self.models[category][model_name][0] is None:
            model = model_class.from_pretrained(*args, **kwargs)
            tokenizer = tokenizer_class.from_pretrained(*args, **kwargs)
            self.models[category][model_name] = (model, tokenizer)
        return self.models[category][model_name]

    # ================================
    # Translation Models (Lazy loading)
    # ================================
    
    def translator_opus_mt(self):
        return self._load_model(
            'translator', 'opus_mt', AutoModelForSeq2SeqLM, AutoTokenizer, "Helsinki-NLP/opus-mt-ja-en"
        )

    def translator_small100(self):
        """Load the small100 translation model."""
        model = self._load_model(
            'translator', 'small100', M2M100ForConditionalGeneration, M2M100Tokenizer, "alirezamsh/small100"
        )
        tokenizer = model[1]
        # Set target language for the tokenizer here
        tokenizer.tgt_lang = "en"  # Set the target language here
        return model[0], tokenizer

    def translator_mbart(self):
        """Load the MBart translation model."""
        model = self._load_model(
            'translator', 'mbart', MBartForConditionalGeneration, MBart50TokenizerFast, "facebook/mbart-large-50-many-to-many-mmt"
        )
        tokenizer = model[1]
        # Set source and target language for the tokenizer
        tokenizer.src_lang = "ja_XX"  # Set the source language to Japanese
        tokenizer.tgt_lang = "en_XX"  # Set the target language to English
        return model[0], tokenizer

    # ================================
    # OCR Models (Lazy loading)
    # ================================

    def ocr_manga_ocr(self):
        if self.models['ocr']['manga_ocr'][0] is None:  # Check if model is not loaded
            # Load both the model and image processor for Manga OCR
            self.models['ocr']['manga_ocr'] = (
                ViTImageProcessor.from_pretrained("kha-white/manga-ocr-base"),
                VisionEncoderDecoderModel.from_pretrained("kha-white/manga-ocr-base"),
                AutoTokenizer.from_pretrained("kha-white/manga-ocr-base")
            )
        return self.models['ocr']['manga_ocr']

    def ocr_got(self):
        model, tokenizer = self._load_model(
            'ocr', 'got', AutoModel, AutoTokenizer, "srimanth-d/GOT_CPU", trust_remote_code=True
        )
        
        model.config.use_safetensor = True
        model.config.pad_token_id = tokenizer.eos_token_id
        return model, tokenizer

    # ================================
    # Kanji to Kana Conversion Models (Lazy loading)
    # ================================

    def kana_elyza(self):
        return self._load_model(
            'kana', 'elyza', AutoModelForCausalLM, AutoTokenizer, "elyza/Llama-3-ELYZA-JP-8B", torch_dtype="auto", device_map="auto"
        )

    def kana_pykakasi(self):
        if self.models['kana']['pykakasi'] is None:
            self.models['kana']['pykakasi'] = pykakasi.kakasi()
        return self.models['kana']['pykakasi']

    # ================================
    # Semantic Comparison Models (Lazy loading)
    # ================================

    def similarity_minilm(self):
        return self._load_model(
            'similarity', 'minilm', AutoModel, AutoTokenizer, "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    # ================================
    # Translation Methods
    # ================================

    def translate_text(self, text: str, model: str) -> str:
        """Translates Japanese to English using the specified model."""
        model, tokenizer = getattr(self, f'translator_{model}')()
        batch = tokenizer([text], return_tensors="pt")
        generated_ids = model.generate(**batch)
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # ================================
    # OCR Methods
    # ================================

    def text_from_image_manga_ocr(self, picture: str) -> str:
        """Extract text from an image using Manga OCR."""
        image = PIL.Image.open(picture)
        image_processor, model, tokenizer = self.ocr_manga_ocr()
        inputs = image_processor(images=image, return_tensors="pt").pixel_values
        outputs = model.generate(inputs)
        return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(" ","")

    def text_from_image_got(self, picture: str) -> str:
        """Extract text from an image using srimanth-d/GOT_CPU."""    
        model, tokenizer = self.ocr_got()

        inputs = tokenizer(picture, return_tensors="pt", padding=True, truncation=True)
        inputs["attention_mask"] = (inputs["input_ids"] != model.config.pad_token_id).long()

        return model.chat(tokenizer, picture, ocr_type='ocr')

    # ================================
    # Kanji to Kana Conversion Methods
    # ================================

    def convert_kanji_to_kana_elyza(self, text: str):
        """Convert Kanji text to Hiragana and Romaji using Elyza's model."""
        model, tokenizer = self.kana_elyza()
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": "Output the Kanji text in Hiragana, followed by Latin."}, 
             {"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        token_ids = tokenizer.encode(prompt, return_tensors="pt")
        output_ids = model.generate(token_ids)
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True).split("\n", 1)
        return output[0].strip(), output[-1].strip()

    def convert_kanji_to_kana_pykakasi(self, text: str):
        """Convert Kanji text to Hiragana and Romaji using pykakasi."""
        result = self.kana_pykakasi().convert(text)
        hiragana = " ".join(item['hira'] for item in result)
        romaji = " ".join(item['hepburn'] for item in result)
        return hiragana.strip(), romaji.strip()

    # ================================
    # Semantic Comparison Methods
    # ================================

    def compare_semantics(self, sentences):
        """Compute sentence embeddings using a semantic similarity model."""
        model, tokenizer = self.similarity_minilm()
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        return self.mean_pooling(model_output, encoded_input['attention_mask'])

    def mean_pooling(self, model_output, attention_mask):
        """Mean pool the model output to get the sentence embeddings."""
        token_embeddings = model_output[0]  # Token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def compare_embeddings(self, embedding1, embedding2):
        """Compute cosine similarity between two sentence embeddings."""
        cos_sim = cosine_similarity(embedding1.unsqueeze(0).cpu().numpy(), embedding2.unsqueeze(0).cpu().numpy())
        return cos_sim[0][0]

    def compare_with_original(self, sentences):
        """Compare the embeddings of all translations with the original text."""
        embeddings = self.compare_semantics(sentences)
        similarity_scores = {
            f"Original compared to sentence {i+1}": self.compare_embeddings(embeddings[0], emb)
            for i, emb in enumerate(embeddings[1:])
        }
        return similarity_scores
