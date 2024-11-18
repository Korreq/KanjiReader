from transformers import AutoTokenizer, MarianMTModel
import PIL.Image
from manga_ocr import MangaOcr
import pykakasi

class Models:

    def translate( self, text ):

        model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ja-en")

        batch = tokenizer( [text], return_tensors="pt" )

        generated_ids = model.generate( **batch )

        return tokenizer.batch_decode( generated_ids, skip_special_tokens=True )[0]


    def textFromImage( self, picture ):

        ocr = MangaOcr()

        return ocr( PIL.Image.open( picture ).convert("RGB") )

    #Add formating to output 
    def convertKana( self, text ):

        kanaConverter = pykakasi.kakasi() 

        return kanaConverter.convert( text )


        