#pip install transformers torch sentencepiece sacremoses
from transformers import AutoTokenizer, MarianMTModel
#pip install pillow
from PIL import Image
#pip install manga_ocr
from manga_ocr import MangaOcr

from tkinter import *

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

    app = Tk(baseName='Kanji Reader')

    menuApp = Menu(app)

    app.config(menu=menuApp)

    inputMenu = Menu(menuApp)
    menuApp.add_cascade(label='Input', menu = inputMenu)

    inputMenu.add_command(label='Write')
    inputMenu.add_command(label='Read from image')
    inputMenu.add_separator()
    inputMenu.add_command(label='Exit', command=app.quit)


    Label(app, text='Text to translate').grid(row=0)
    entry = Entry( app )
    entry.grid(row=1)
    button = Button( app, text='Translate', width=25, command=app.destroy )
    button.grid(row=2)

    mainloop()
    #text = "今日はいい天気ですね"
'''
    image = r"files/images/test.png"

    textImg = textFromImage( image )

    print( textImg )

    print( translate( textImg ) )
'''

if __name__ == "__main__":

    main()