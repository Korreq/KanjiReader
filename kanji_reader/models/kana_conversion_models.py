from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pykakasi


class KanaConversionModels:
    def __init__(self):
        # Initialize Kana conversion models and tokenizers
        self.models = {
            'elyza': (None, None),
            'pykakasi': None,
        }

    def kana_elyza(self):
        # Load Elyza's Kana conversion model
        if self.models['elyza'][0] is None:
            model = AutoModelForCausalLM.from_pretrained("elyza/Llama-3-ELYZA-JP-8B", torch_dtype="auto", device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained("elyza/Llama-3-ELYZA-JP-8B")
            
            model.eval()
            tokenizer.pad_token = tokenizer.eos_token

            self.models['elyza'] = (model, tokenizer)
        return self.models['elyza']

    def kana_pykakasi(self):
        # Load pykakasi Kana conversion model
        if self.models['pykakasi'] is None:
            self.models['pykakasi'] = pykakasi.kakasi()
        return self.models['pykakasi']

    def convert_kanji_to_kana_elyza(self, text: str):
        """Convert Kanji text to Hiragana and Romaji using Elyza's model."""
        model, tokenizer = self.kana_elyza()
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": "Output the Kanji text in Hiragana, followed by Latin, with no extra text."}, 
             {"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )

        token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

        attention_mask = torch.ones(token_ids.shape, device=token_ids.device)

        with torch.no_grad():
            output_ids = model.generate(
                token_ids.to(model.device),
                attention_mask=attention_mask,
                max_new_tokens=2048,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=50
            )

        output = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().split("\n")

        return output[-2].strip(), output[-1].strip()

    def convert_kanji_to_kana_pykakasi(self, text: str):
        """Convert Kanji text to Hiragana and Romaji using pykakasi."""
        result = self.kana_pykakasi().convert(text)
        hiragana = " ".join(item['hira'] for item in result)
        romaji = " ".join(item['hepburn'] for item in result)
        return hiragana.strip(), romaji.strip()