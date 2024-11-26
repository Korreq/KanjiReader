'''
installation commands, make sure to run them on your virtual enviroment.:
    pip install transformers torch sentencepiece sacremoses #( cpu only configuration )
    pip install pillow
    pip install manga_ocr
    pip install pykakasi
'''

from tkinter import *
#if not installed check how to install python-tk on your system
from models import Models


def main():

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
    
    imageInput = r"files/images/test.png"

    models = Models()

    textImg = models.text_from_image( imageInput )

    print( textImg )

    print( text )

    print( models.translate( text ) )

    result = models.convert_kana( text )

    for item in result:

        print( item )

        #print("{}: kana '{}', hiragana '{}', romaji: '{}'".format(item['orig'], item['kana'], item['hira'], item['hepburn']))


if __name__ == "__main__":

    main()