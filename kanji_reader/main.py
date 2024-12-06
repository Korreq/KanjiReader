'''
If you are missing some modules, check setup.py file
'''

import pyautogui
import tkinter as tk
from kanji_reader.gui import TranslationApp

def main():
    root = tk.Tk()
    app = TranslationApp(root)
    root.mainloop()

    #image = pyautogui.screenshot("files/images/screen.png")

if __name__ == "__main__":
    main()





