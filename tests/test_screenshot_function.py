import unittest
from unittest.mock import MagicMock, patch
import tkinter as tk
from PIL import Image, ImageTk
from kanji_reader import TranslationApp  # Import your TranslationApp class


class TestTranslationApp(unittest.TestCase):

    def setUp(self):
        # Create a mock tkinter root window for testing
        self.root = tk.Tk()
        self.app = TranslationApp(self.root)

    def test_translate_text(self):
        # Test that the translate function correctly updates the label with translation information
        input_text = "こんにちは"
        label_mock = MagicMock()

        # Mock models' translate method
        self.app.models.translate_text = MagicMock(return_value="Hello")
        self.app.models.convert_kanji_to_kana_pykakasi = MagicMock(return_value=("こんにちは", "kon'nichiwa"))

        self.app.translate(label_mock, input_text, "text")

        # Assert that the translation method was called with the correct text
        self.app.models.translate_text.assert_called_with(input_text, 'opus_mt')

        # Assert that the label was updated with the correct translation, hiragana, and romaji
        label_mock.config.assert_called_with(
            text="Text: こんにちは \n Translation: Hello \n Hiragana: こんにちは \n Romaji: kon'nichiwa"
        )

    @patch('tkinter.filedialog.askopenfilename')
    @patch('kanji_reader.Models.text_from_image_manga_ocr')
    def test_upload_image(self, mock_ocr, mock_file_dialog):
        # Simulate file dialog and OCR
        mock_file_dialog.return_value = "test_image.png"
        mock_ocr.return_value = "Kanji detected text"

        # Mock the translate method to just print the output for now
        self.app.translate = MagicMock()

        # Run the upload image functionality
        self.app.upload_image()

        # Assert that the file dialog was opened and the OCR function was called
        mock_file_dialog.assert_called_once()
        mock_ocr.assert_called_once_with("test_image.png")

        # Check that the translate function was called with the correct input
        self.app.translate.assert_called_with(self.app.text_photo_label, "Kanji detected text", "image")


    def test_crop_screenshot_window(self):
        # Test that crop window handles mouse clicks and drags
        image_path = "data/images/temp/screenshot.png"
        self.app.image = Image.new('RGB', (100, 100))  # Mock image for cropping
        self.app.image_tk = ImageTk.PhotoImage(self.app.image)

        # Create a crop window and simulate mouse events
        crop_window = self.app.create_crop_window()
        self.app.on_mouse_click(MagicMock(x=10, y=10))
        self.app.on_mouse_drag(MagicMock(x=50, y=50))

        # Check that the rectangle was updated
        self.assertEqual(self.app.rect, [10, 10, 50, 50])


if __name__ == '__main__':
    unittest.main()
