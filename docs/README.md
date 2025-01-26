# KanjiReader
Japanese into English translation app that also converts kanji to hiragana

How to set up:

 Ensure you're in a Python virtual environment before running these commands
 
If trying to install torch on windows check currently suported python version from their site:
https://pytorch.org/get-started/locally/
Currently supported python version: 3.12 //Checked 12.12.2024
    
Steps to create virtual enviroment with older python version:

  1.Download desired python version from offical site:
    https://www.python.org/downloads/windows/
    
  2.Make sure to install virtualenv on your GLOBAL python enviroment:
    pip install virtualenv
    
  3.Create virtual enviroment with desired python version path                    
    python -m virtualenv -p C:\Users\...\AppData\Local\Programs\Python\Python312\python.exe venv
        
  To install the necessary packages, run this command in your terminal: 
    pip install -e kanji_reader
    
  If there are some errors during package install, you need to install all specified packages in setup() manualy
        pip install transformers torch sentencepiece sacremoses pillow manga-ocr pykakasi pyautogui tk 

  To remove all not esentiall installed packages, run uninstall_all.py
    
  Then, you can reinstall additional packages for this project by running the install command again

To run test use this command:
python -m tests.test_models
