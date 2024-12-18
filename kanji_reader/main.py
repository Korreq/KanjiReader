'''
If you are missing some modules, check setup.py file

If your terminal don't see kanji_reader package, run main by this command: 
python -m kanji_reader.main
'''



import tkinter as tk
from .gui import TranslationApp

def main():
    root = tk.Tk()
    app = TranslationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()





