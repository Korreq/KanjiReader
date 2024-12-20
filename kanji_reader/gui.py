import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from kanji_reader.models import Models


class TranslationApp:
    def __init__(self, root):
        self.root = root
        self.models = Models()
        self.setup_gui()

    def setup_gui(self):
        self.root.title("Translation App")

        tab_control = ttk.Notebook(self.root)
        self.tab1 = ttk.Frame(tab_control)
        self.tab2 = ttk.Frame(tab_control)

        tab_control.add(self.tab1, text='Text Translation')
        tab_control.add(self.tab2, text='Image Translation')
        tab_control.pack(expand=1, fill='both')

        self.setup_text_tab()
        self.setup_image_tab()

    def setup_text_tab(self):
        self.text_entry = tk.Text(self.tab1, height=10, width=50)
        self.text_entry.pack(pady=10)
        translate_button = tk.Button(self.tab1, text="Translate", command=self.translate_text)
        translate_button.pack(pady=5)
        self.translation_label = tk.Label(self.tab1, text="")
        self.translation_label.pack(pady=10)

    def setup_image_tab(self):
        upload_button = tk.Button(self.tab2, text="Upload Image", command=self.upload_image)
        upload_button.pack(pady=20)
        self.translation_photo_label = tk.Label(self.tab2, text="")
        self.translation_photo_label.pack(pady=10)

    def translate_text(self):
        input_text = self.text_entry.get("1.0", tk.END).strip()
        translated_text = f"Translation: {self.models.translate(input_text)}"
        self.translation_label.config(text=translated_text)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            input_text = self.models.text_from_image(file_path)
            translated_text = f"Translation: {self.models.translate(input_text)}"
            self.translation_photo_label.config(text=translated_text)
