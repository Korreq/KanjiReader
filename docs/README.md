# KanjiReader
Japanese into English translation app that also converts kanji to hiragana

How to set up:

 Ensure you're in a Python virtual environment before running these commands
 
If trying to install torch on windows check currently suported python version from their site: <br>
https://pytorch.org/get-started/locally/ <br>
Currently supported python version: 3.12 //Checked 12.12.2024
    
Steps to create virtual enviroment with older python version: <br>
  1.Download desired python version from offical site: <br>
    https://www.python.org/downloads/windows/ <br>
  2.Make sure to install virtualenv on your GLOBAL python enviroment: <br>
    pip install virtualenv <br>
  3.Create virtual enviroment with desired python version path <br>
  python -m virtualenv -p C:\Users\...\AppData\Local\Programs\Python\Python312\python.exe venv <br>

  To install the necessary packages, run this command in your terminal: <br>
    pip install -e kanji_reader
    
  If there are some errors during package install, you need to install all specified packages in setup() manualy <br>
        pip install transformers torch sentencepiece sacremoses pillow manga-ocr pykakasi pyautogui tk 

  To remove all not esentiall installed packages, run uninstall_all.py
    
  Then, you can reinstall additional packages for this project by running the install command again

To run test use this command:
python -m tests.test_models

Evaluation Metrics: <br>
![image](https://github.com/user-attachments/assets/859eca10-448f-40c5-a112-85b978f1b44a) <br>
Metrics indicate, that the the most precise models combination is: <br>
manga_ocr + mbart + pykakasi with total avarage score of 0.652125 <br>

This total avarage comes from 4 avarages for each models' tasks. For this combination
it looked like this: <br>
Avarage OCR, Avarage Translation, Avarage Hiragana, Avarage Romaji <br>
0.99154, 0.74952, 0.6059, 0.26154

Clearly it shows the problem with pykakasi conversion. The returned converted sentences are mostly correct
but pykakasi tends to convert っ to つ , which hurts its simalirity even greater for romaji sentences.

