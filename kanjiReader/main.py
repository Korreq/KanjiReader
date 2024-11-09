#pip install transformers torch sentencepiece sacremoses
from transformers import AutoTokenizer, MarianMTModel
#pip install pillow
from PIL import Image
#pip install manga_ocr
from manga_ocr import MangaOcr


def translate( text ):

    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ja-en")

    batch = tokenizer( [text], return_tensors="pt" )

    generated_ids = model.generate( **batch )

    return tokenizer.batch_decode( generated_ids, skip_special_tokens=True )[0]


def textFromImage( img ):

    mocr = MangaOcr()

    return mocr( Image.open( img ).convert("RGB") )


def main():

    #text = "今日はいい天気ですね"

    image = r"files/images/test.png"

    textImg = textFromImage( image )

    print( textImg )

    print( translate( textImg ) )


if __name__ == "__main__":

    main()