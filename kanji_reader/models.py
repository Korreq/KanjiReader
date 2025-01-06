from transformers import (
    MarianMTModel, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline,
    VisionEncoderDecoderModel, AutoFeatureExtractor, AutoModelForCausalLM,
    BlipForConditionalGeneration, BlipProcessor, AutoModel,
    M2M100ForConditionalGeneration, M2M100Tokenizer, T5Tokenizer, T5ForConditionalGeneration,
    MBartForConditionalGeneration, MBart50TokenizerFast, AutoModelForTokenClassification, ViTImageProcessor
)

from sklearn.metrics.pairwise import cosine_similarity
import PIL.Image
import cv2
import pytesseract
import torch
import os
import pykakasi

class Models:
    """Handles all model-related tasks, including translation, OCR, and Kanji to Kana conversion."""

    def __init__(self, test=False):
       
        # Initialize translation model for main app
        self._translator_model_opus_mt = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
        self._translator_tokenizer_opus_mt = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ja-en")

        # Initialize OCR model for main app
        self._ocr_model_manga_ocr = VisionEncoderDecoderModel.from_pretrained("kha-white/manga-ocr-base")
        self._ocr_image_processor_manga_ocr = ViTImageProcessor.from_pretrained("kha-white/manga-ocr-base")
        self._ocr_tokenizer_manga_ocr = AutoTokenizer.from_pretrained("kha-white/manga-ocr-base")

        if test:
            # Initialize translation models

            ''' Left it here, if we wanted to use another model in main app
            # Helsinki-NLP/opus-mt-ja-en
            self._translator_model_opus_mt = AutoModel.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
            self._translator_tokenizer_opus_mt = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
            #'''

            # alirezamsh/small100
            self._translator_small100 = AutoModelForSeq2SeqLM.from_pretrained("alirezamsh/small100")
            self._tokenizer_small100 = AutoTokenizer.from_pretrained("alirezamsh/small100", tgt_lang="en")

            # facebook/mbart-large-50-many-to-many-mmt
            self._translator_mbart = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
            self._tokenizer_mbart = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")


            # Initialize OCR models

            ''' Left it here, if we wanted to use another model in main app
            # kha-white/manga-ocr-base
            self._ocr_model_manga_ocr = VisionEncoderDecoderModel.from_pretrained("kha-white/manga-ocr-base")
            self._ocr_image_processor_manga_ocr = AutoFeatureExtractor.from_pretrained("kha-white/manga-ocr-base")
            self._ocr_tokenizer_manga_ocr = AutoTokenizer.from_pretrained("kha-white/manga-ocr-base")
            #'''

            # srimanth-d/GOT_CPU
            self._ocr_tokenizer_got = AutoTokenizer.from_pretrained("srimanth-d/GOT_CPU", trust_remote_code=True)
            self._ocr_model_got = AutoModel.from_pretrained("srimanth-d/GOT_CPU", trust_remote_code=True, use_safetensors=True, pad_token_id=self._ocr_tokenizer_got.eos_token_id)

            # Initialize Kanji to Kana conversion model

            # elyza/Llama-3-ELYZA-JP-8B
            self._converter_model_elyza = AutoModelForCausalLM.from_pretrained("elyza/Llama-3-ELYZA-JP-8B")
            self._converter_tokenizer_elyza = AutoTokenizer.from_pretrained("elyza/Llama-3-ELYZA-JP-8B")

            # pykakasi
            self._kakasi = pykakasi.kakasi()


            # Initialize semantic similarity model

            # sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
            self._similarity_model_minilm = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            self._similarity_tokenizer_minilm = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # ================================
    # Translation Methods
    # ================================

    def translate(self, text: str) -> str:
        """Translate Japanese to English using Helsinki-NLP/opus-mt-ja-en."""
        model = self._translator_model_opus_mt
        tokenizer = self._translator_tokenizer_opus_mt

        batch = tokenizer([text], return_tensors="pt")
        generated_ids = model.generate(**batch)
        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return output

    def translate_small100(self, text: str) -> str:
        """Translate Japanese to English using alirezamsh/small100."""
        model = self._translator_small100
        tokenizer = self._tokenizer_small100
        
        encode = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(**encode)
        output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return output

    def translate_mbart(self, text: str) -> str:
        """Translate Japanese to English using facebook/mbart-large-50-many-to-many-mmt."""
        model = self._translator_mbart
        tokenizer = self._tokenizer_mbart

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

        inputs = self._ocr_image_processor_manga_ocr(images=image, return_tensors="pt").pixel_values
        outputs = self._ocr_model_manga_ocr.generate(inputs)
        tokenizer = self._ocr_tokenizer_manga_ocr

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    #Need to have older version of python to run it and it's very bad
    def text_from_image_got(self, picture: str) -> str:
            
        tokenizer = self._ocr_tokenizer_got
        model = self._ocr_model_got
        model = model.eval()

        output = model.chat(tokenizer, picture, ocr_type='ocr')

        return output
    
    # ================================
    # Kanji to Kana Conversion Methods
    # ================================

    def convert_kanji_to_kana_elyza(self, text: str):
        """Process Kanji text and return Hiragana and Romaji using elyza/Llama-3-ELYZA-JP-8B model."""

        System_prompt = (
            "Output the Kanji text in Hiragana, followed by Latin, with no extra text."
        )

        tokenizer = self._converter_tokenizer_elyza
        model = self._converter_model_elyza
        model.eval()
        tokenizer.pad_token = tokenizer.eos_token

        messages = [
            {"role": "system", "content": System_prompt},
            {"role": "user", "content": text},
        ]

        prompt = tokenizer.apply_chat_template( messages, tokenize=False, add_generation_prompt=True )
        token_ids = tokenizer.encode( prompt, add_special_tokens=False, return_tensors="pt" )

        attention_mask = torch.ones(token_ids.shape, device=token_ids.device)

        with torch.no_grad():
            output_ids = model.generate(
                token_ids.to(model.device),
                attention_mask=attention_mask,
                max_new_tokens=512,
                do_sample=False,
                top_k=50,
                top_p=None,
                temperature=None,
            )

        output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):], skip_special_tokens=True)
        
        output = output.split("\n", 1)       
        hiragana = output[0].strip() 
        romaji = output[-1].strip() 

        return hiragana, romaji

    def convert_kanji_to_kana_pykakasi(self, text: str):

        kks = self._kakasi
        result = kks.convert(text)

        hiragana = romaji = ""

        for item in result:
            hiragana += item['hira'] + " "
            romaji += item['hepburn'] + " "
            
        return hiragana, romaji

    #Requires authentication
    def convert_kana_gemma(self, text: str):
        """Convert Kanji to Hiragana and Romanji using Gemma."""
        pipe = pipeline(
            "text-generation",
            model="google/gemma-2-2b-jpn-it",
            model_kwargs={"torch_dtype": torch.float16},
            device=0
        )

        input_text_prompt = f"Convert following text from Kanji to Hiragana:\n\n{text}"
        messages = [
            {"role": "user", "content": input_text_prompt},
        ]

        outputs = pipe(messages, return_full_text=False, max_new_tokens=1024)
        hiragana_response = outputs[0]["generated_text"].strip()
        
        input_text_prompt = f"Convert following text from Kanji to Romanji:\n\n{text}"
        messages = [
            {"role": "user", "content": input_text_prompt},
        ]
        
        outputs = pipe(messages, return_full_text=False, max_new_tokens=1024)
        romaji_response = outputs[0]["generated_text"].strip()

        return hiragan_response, romaji_response

    # ================================
    # Semantic Comparision
    # ================================

    def compare_semantics(self, sentences):

        tokenizer = self._similarity_tokenizer_minilm
        model = self._similarity_model_minilm

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
            similarity_scores[f"Original compared to sentence {i+1}"] = similarity_score

        return similarity_scores