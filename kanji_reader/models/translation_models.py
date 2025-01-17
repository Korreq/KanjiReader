from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, M2M100ForConditionalGeneration, M2M100Tokenizer, MBartForConditionalGeneration, MBart50TokenizerFast


class TranslationModels:
    def __init__(self):
        # Initialize translation models and tokenizers
        self.models = {
            'opus_mt': (None, None),
            'small100': (None, None),
            'mbart': (None, None),
        }

    def translator_opus_mt(self):
        # Load OPUS-MT translation model
        if self.models['opus_mt'][0] is None:
            model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
            tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
            self.models['opus_mt'] = (model, tokenizer)
        return self.models['opus_mt']

    def translator_small100(self):
        # Load small100 translation model
        if self.models['small100'][0] is None:
            model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100")
            tokenizer = M2M100Tokenizer.from_pretrained("alirezamsh/small100")
            tokenizer.tgt_lang = "en"
            self.models['small100'] = (model, tokenizer)
        return self.models['small100']

    def translator_mbart(self):
        # Load MBart translation model
        if self.models['mbart'][0] is None:
            model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
            tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
            tokenizer.src_lang = "ja_XX"
            tokenizer.tgt_lang = "en_XX"
            self.models['mbart'] = (model, tokenizer)
        return self.models['mbart']

    def translate_text(self, text, model_name):
        # Translate Japanese to English using the specified model
        model, tokenizer = getattr(self, f'translator_{model_name}')()
        batch = tokenizer([text], return_tensors="pt")
        generated_ids = model.generate(**batch)
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]