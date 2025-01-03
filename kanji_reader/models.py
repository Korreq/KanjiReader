from transformers import (
    MarianMTModel, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline,
    VisionEncoderDecoderModel, AutoFeatureExtractor, AutoModelForCausalLM,
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
import pykakasi

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

    #Need to have older version of python to run it
    def text_from_image_got(self, picture: str) -> str:
      
        tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
        model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
        model = model.eval()

        output = model.chat(tokenizer, picture, ocr_type='ocr')

        return output

    '''
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
    '''
    # ================================
    # Kanji to Kana Conversion Methods
    # ================================

    def convert_kanji_to_kana_elyza(self, text: str):
        """
        Process Kanji text and return Hiragana and Romanji using elyza/Llama-3-ELYZA-JP-8B model.
        """

        System_prompt = (
            "Convert the given Kanji text to Hiragana and Romanji and return only the result"
            "First output should be the Hiragana version of the text."
            "Second output should be Romanji version of the text. No other text should be included."
        )

        model_name = "elyza/Llama-3-ELYZA-JP-8B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
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
                max_new_tokens=1024,
                do_sample=False,
                top_k=50,
                top_p=None,
                temperature=None,
            )

        output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):], skip_special_tokens=True) + "}"

        

        print(output)

        return output

    def convert_kanji_to_kana_pykakasi(self, text: str):

        kks = pykakasi.kakasi()
        result = kks.convert(text)

        hiragana = romanji = ""

        for item in result:
            hiragana += item['hira'] + " "
            romanji += item['hepburn'] + " "
            
        return hiragana, romanji

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
        romanji_response = outputs[0]["generated_text"].strip()

        return hiragan_response, romanji_response

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