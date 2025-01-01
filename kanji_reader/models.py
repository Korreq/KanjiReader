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
import json

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

    '''
        searching for new models becuse these are bad


        elyza/Llama-3-ELYZA-JP-8B
        llm-jp/llm-jp-3-13b

        
        google/byt5-large

    
    '''

    def convert_kanji_to_kana_elyza(self, text: str):
        """
        Process Kanji text and return Hiragana and Romanji in JSON format using elyza/Llama-3-ELYZA-JP-8B model.
        """

        System_prompt = (
             "Convert the given Kanji text to Hiragana and Romanji and return only the result in JSON format. "
            "The JSON should have two keys: 'hiragana' for the Hiragana transcription and "
            "'romanji' for the Romanized version of the text. No other text should be included."
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

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

        try:
            result = json.loads(output)
        except json.JSONDecodeError:
            result = {
                "hiragana": "Error: Invalid output format",
                "romanji": "Error: Invalid output format"
            }

        print(output)

        return json.dumps(result)

    def convert_kanji_to_kana_jp(self, text: str):

        model_name = "llm-jp/llm-jp-3-13b"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()

        prompt = f"Convert the following Kanji text to Hiragana and Romanji:\n\n{text}"

        
        tokenized_input = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

        attention_mask = torch.ones(tokenized_input.shape, device=tokenized_input.device)

        with torch.no_grad():
            output_id = model.generate(
                tokenized_input,
                attention_mask=attention_mask,
                max_new_tokens=512,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
                repetition_penalty=1.05,

            )[0]

        output = tokenizer.decode(output_id)

        return output

    def convert_kana(self, text: str):
        """Convert Kanji to Hiragana and Romanji."""
        model_name = "Miwa-Keita/zenz-v2-gguf"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs)
        hiragana_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        outputs = model.generate(**inputs)
        romanji_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return hiragana_text, romanji_text

    def convert_kana_byt5(self, text: str):
        """Convert Kanji to Hiragana and Romanji using ByT5."""

        model = T5ForConditionalGeneration.from_pretrained('google/byt5-large')

        input_ids = torch.tensor([list("Life is like a box of chocolates.".encode("utf-8"))]) + 3  # add 3 for special tokens
        labels = torch.tensor([list("La vie est comme une boîte de chocolat.".encode("utf-8"))]) + 3  # add 3 for special tokens

        loss = model(input_ids, labels=labels).loss # forward pass


        model_name = "google/byt5-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)

        prompt = f"Convert following Kanji text to Hiragana and Romanji:\n\n{text}"

        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, padding=True)  

        outputs = model.generate(
            **inputs,
            max_length=1024, num_beams=5, early_stopping=True
        )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return result

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