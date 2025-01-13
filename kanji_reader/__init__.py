"""
This package contains all the necessary modules for the translation application.
- `models` module: Defines the Models class for handling OCR and translation.
- `gui` module: Defines the GUI for the translation application.
"""

from .models_tests import TestModels
from .models import Models
from .gui import TranslationApp
from .history import TranslationHistory
