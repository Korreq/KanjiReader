import time
import tkinter as tk
from codeop import compile_command
from tkinter import ttk, filedialog
from tkinter.ttk import Label

from PIL import Image, ImageTk, ImageDraw
import pyautogui

from ..models import Models
from .history import TranslationHistory


class TranslationApp:
    def __init__(self, root):
        self.root = root
        self.models = Models()
        self.history_Handler = TranslationHistory()

        # Initialize variables for crop functionality
        self.history = self.history_Handler.get_translation()
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.image_tk = None
        self.image = None
        self.image_label = None

        # Setup GUI components
        self.setup_gui()

    def setup_gui(self):
        self.root.title("Kanji Reader")

        tab_control = ttk.Notebook(self.root)

        self.tab1 = ttk.Frame(tab_control)
        self.tab2 = ttk.Frame(tab_control)
        self.tab3 = ttk.Frame(tab_control)
        self.tab4 = ttk.Frame(tab_control)

        tab_control.add(self.tab1, text="Text Translation")
        tab_control.add(self.tab2, text="Image Translation")
        tab_control.add(self.tab3, text="Screenshot Translation")
        tab_control.add(self.tab4, text="History")
        tab_control.pack(expand=1, fill="both")

        self.setup_text_tab()
        self.setup_image_tab()
        self.setup_screenshot_tab()
        self.setup_history_tab()

        tab_control.bind("<<NotebookTabChanged>>", self.on_tab_changed)

    def on_tab_changed(self, event):
        selected_tab = event.widget.select()
        if selected_tab == str(self.tab4):  # History Tab selected
            self.refresh_history_list()

    def setup_text_tab(self):
        self.text_entry = tk.Text(self.tab1, height=10, width=50)
        self.text_entry.pack(pady=10)

        translate_button = tk.Button(self.tab1, text="Translate", command=self.translate_text)
        translate_button.pack(pady=5)

        self.text_label = tk.Label(self.tab1, text="")
        self.text_label.pack(pady=10)

    def setup_image_tab(self):
        upload_button = tk.Button(self.tab2, text="Upload Image", command=self.upload_image)
        upload_button.pack(pady=20)

        self.text_photo_label = tk.Label(self.tab2, text="")
        self.text_photo_label.pack(pady=10)

    def setup_screenshot_tab(self):
        take_screenshot_button = tk.Button(self.tab3, text="Take Screenshot", command=self.take_screenshot)
        take_screenshot_button.pack(pady=20)

        self.text_screenshot_label = tk.Label(self.tab3, text="")
        self.text_screenshot_label.pack(pady=10)

    def setup_history_tab(self):
        # History Name Label
        history_name_label = tk.Label(self.tab4, text="History", anchor="center")
        history_name_label.pack(pady=10)

        # Create a frame to hold the listbox and history label in the same row
        history_frame = tk.Frame(self.tab4)
        history_frame.pack(pady=10, fill=tk.BOTH, expand=1)

        # Create the scrollable listbox
        scrollbar = tk.Scrollbar(history_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        self.history_list = tk.Listbox(history_frame, height=5, yscrollcommand=scrollbar.set)
        self.history = self.history_Handler.get_translation()
        for translation in self.history:
            self.history_list.insert("end", translation["input_text"])
        self.history_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=self.history_list.yview)

        # Create the history label on the right side of the listbox
        self.history_label = tk.Label(history_frame,
                                      text="Click on an item in the list to show more information about translation")
        self.history_label.pack(side=tk.RIGHT, padx=10, fill=tk.BOTH, expand=True)

        # Bind listbox item selection to an event handler
        self.history_list.bind("<<ListboxSelect>>", lambda event: self.on_item_selected())

        # Initial population of the history list
        self.refresh_history_list()


    def refresh_history_list(self):
        # Clear current history listbox content
        self.history_list.delete(0, tk.END)

        # Get updated history and populate listbox
        self.history = self.history_Handler.get_translation()
        for translation in self.history:
            self.history_list.insert("end", translation["input_text"])


    def on_item_selected(self):
        # Get selected index from the listbox
        selected_index = self.history_list.curselection()

        if selected_index:
            # Get the translation at the selected index
            index = selected_index[0]
            selected_translation = self.history[index]

            # Update the label with more information about the selected translation
            self.history_label.config(text=f"Text: {selected_translation["input_text"]}\n"
                                      f"Input Type: {selected_translation["input_type"]}\n"
                                      f"Translation: {selected_translation["translated_text"]}\n"
                                      f"Hiragana: {selected_translation["hiragana_text"]}\n"
                                      f"Romaji: {selected_translation["romaji_text"]}\n")

    def translate(self, label:Label, input_text, input_type:str):
        translation = self.models.translate_text(input_text, 'opus_mt')
        hiragana, romaji = self.models.convert_kanji_to_kana_pykakasi(input_text)

        label.config(text=f"""Text: {input_text} \n Translation: {translation} \n Hiragana: {hiragana} \n Romaji: {romaji}""")

        # Save translation to history
        self.history_Handler.save_translation(input_text, input_type, translation, hiragana, romaji)

    def translate_text(self):
        input_text = self.text_entry.get("1.0", tk.END).strip()
        self.translate(self.text_label,input_text, "text")

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            input_text = self.models.text_from_image_manga_ocr(file_path)
            self.translate(self.text_photo_label, input_text, "image")


    def take_screenshot(self):
        self.root.iconify()
        time.sleep(1)  # Allow the app to minimize before taking the screenshot
        screenshot = pyautogui.screenshot()
        self.root.deiconify()

        screenshot.save("data/images/temp/screenshot.png")
        cropped_image = self.crop_screenshot_window("data/images/temp/screenshot.png")

        if cropped_image:
            cropped_image.save("data/images/temp/crop.png")

        
        input_text = self.models.text_from_image_manga_ocr("data/images/temp/crop.png")
        self.translate(self.text_screenshot_label, input_text, "screenshot")

    def crop_screenshot_window(self, image_path: str):
        self.image = Image.open(image_path)
        self.image_tk = ImageTk.PhotoImage(self.image)

        crop_window = self.create_crop_window()
        crop_window.grab_set()  # Disable interaction with the main window
        self.root.wait_window(crop_window)  # Wait until the user closes the crop window

        if self.start_x is not None and self.start_y is not None and self.rect:
            crop_box = (self.start_x, self.start_y, self.rect[2], self.rect[3])
            return self.image.crop(crop_box)

        return None

    def create_crop_window(self):
        crop_window = tk.Toplevel(self.root)
        crop_window.title("Select Crop Area")
        crop_window.geometry(f"{self.image_tk.width()}x{self.image_tk.height()}")
        crop_window.transient(self.root)

        self.image_label = tk.Label(crop_window, image=self.image_tk)
        self.image_label.pack()

        crop_window.bind("<Button-1>", self.on_mouse_click)
        crop_window.bind("<B1-Motion>", self.on_mouse_drag)

        return crop_window

    def on_mouse_click(self, event):
        if self.start_x is None and self.start_y is None:
            self.start_x, self.start_y = event.x, event.y
            self.rect = [self.start_x, self.start_y, self.start_x + 1, self.start_y + 1]
        else:
            #self.rect[2], self.rect[3] = event.x, event.y
            if event.x < self.start_x and event.y < self.start_y:
                self.start_x, self.start_y = event.x, event.y
            else:
                if event.x <= self.start_x:
                    self.start_x = event.x
                elif event.x <= self.start_x + ((self.rect[2] - self.start_x) // 2):
                    self.start_x = event.x
                else:
                    self.rect[2] = event.x
                if event.y <= self.start_y:
                    self.start_y = event.y
                elif event.y <= self.start_y + ((self.rect[3] - self.start_y) // 2):
                    self.start_y = event.y
                else:
                    self.rect[3] = event.y


        self.draw_rectangle()  # Redraw the rectangle

    def on_mouse_drag(self, event):
        if self.start_x is not None and self.start_y is not None:
            self.rect[2], self.rect[3] = event.x, event.y
            self.draw_rectangle()

    def draw_rectangle(self):
        image_copy = self.image.copy()
        draw = ImageDraw.Draw(image_copy)
        draw.rectangle([self.start_x, self.start_y, self.rect[2], self.rect[3]], outline="red", width=2)

        self.image_tk = ImageTk.PhotoImage(image_copy)
        self.image_label.config(image=self.image_tk)


