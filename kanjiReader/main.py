from transformers import AutoTokenizer, MarianMTModel

def main():

    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ja-en")

    text = "今日はいい天気ですね"

    batch = tokenizer( [text], return_tensors="pt" )

    generated_ids = model.generate( **batch )

    print( tokenizer.batch_decode( generated_ids, skip_special_tokens=True )[0] )

if __name__ == "__main__":

    main()