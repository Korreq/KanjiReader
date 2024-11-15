#pip install transformers torch sentencepiece sacremoses #( cpu only configuration )
from transformers import AutoTokenizer, MarianMTModel
#pip install pillow
from PIL import Image
#pip install manga_ocr
from manga_ocr import MangaOcr
#pip install pykakasi
import pykakasi
#if not installed check how to install python-tk on your system
from tkinter import *


#Might change for more accurate translation model
def translate( text ):

    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ja-en")

    batch = tokenizer( [text], return_tensors="pt" )

    generated_ids = model.generate( **batch )

    return tokenizer.batch_decode( generated_ids, skip_special_tokens=True )[0]


def textFromImage( img ):

    mocr = MangaOcr()

    return mocr( Image.open( img ).convert("RGB") )


def convertKana( text ):

    kks = pykakasi.kakasi() 

    return kks.convert( text )


def main():

    kks = pykakasi.kakasi()

    app = Tk()

    app.title("Kanji Reader")

    '''
    #Gui Stuff
    menuApp = Menu(app)

    app.config(menu=menuApp)

    inputMenu = Menu(menuApp)
    menuApp.add_cascade(label='Input', menu = inputMenu)

    inputMenu.add_command(label='Write')
    inputMenu.add_command(label='Read from image')
    inputMenu.add_separator()
    inputMenu.add_command(label='Exit', command=app.quit)

    frame = Frame(app, height=100, width=200)

    frame.grid(row=1, column=1)

    Label(frame, text='Text to translate').grid(row=0)
    entry = Entry( frame )
    entry.grid(row=1)
    button = Button( frame, text='Translate', width=25, command=frame.destroy )
    button.grid(row=2)

    mainloop()
'''

    #OCR + Translation + Conversion stuff

    text = "今日はいい天気ですね"
    '''
    image = r"files/images/test.png"

    textImg = textFromImage( image )

    print( textImg )

    print( translate( textImg ) )
'''

    result = convertKana( text )

    for item in result:

        print("{}: kana '{}', hiragana '{}', romaji: '{}'".format(item['orig'], item['kana'], item['hira'], item['hepburn']))



if __name__ == "__main__":

    main()