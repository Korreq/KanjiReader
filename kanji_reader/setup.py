'''
    # Ensure you're in a Python virtual environment before running these commands 
     
    # To install the necessary packages, run this command in your terminal: 
    pip install -e kanji_reader
    
    # To remove all not esentiall installed packages, run uninstall_all.py
    
    # Then, you can reinstall additional packages for this project by running the install command again
'''
from setuptools import setup, find_packages

setup(
    name="kanji_reader",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "transformers",  # NLP model
        "torch",  # Deep learning framework
        "sentencepiece",  # Tokenization library
        "sacremoses",  # Preprocessing utilities
        "pillow",  # Image processing library
        "manga-ocr",  # OCR for manga
        "pykakasi",  # Kanji to kana conversion
        "pyautogui",  # GUI automation
        "tk"  # GUI toolkit
    ],
)

