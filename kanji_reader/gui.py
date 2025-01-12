import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import pyautogui

from .models import Models
from .history import TranslationHistory


class TranslationApp:
    def __init__(self, root):
        self.root = root
        self.models = Models()
        self.history = TranslationHistory()

        # Initialize variables for crop functionality
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

        tab_control.add(self.tab1, text="Text Translation")
        tab_control.add(self.tab2, text="Image Translation")
        tab_control.add(self.tab3, text="Screenshot Translation")
        tab_control.pack(expand=1, fill="both")

        self.setup_text_tab()
        self.setup_image_tab()
        self.setup_screenshot_tab()

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

    def setup_screenshot_tab(self):
        take_screenshot_button = tk.Button(self.tab3, text="Take Screenshot", command=self.take_screenshot)
        take_screenshot_button.pack(pady=20)

        self.translation_screenshot_label = tk.Label(self.tab3, text="")
        self.translation_screenshot_label.pack(pady=10)

    def translate_text(self):
        input_text = self.text_entry.get("1.0", tk.END).strip()
        translated_text = f"Translation: {self.models.translate(input_text)}"
        self.translation_label.config(text=translated_text)

        self.history.save_translation(input_text, translated_text)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            input_text = self.models.text_from_image(file_path)
            translated_text = f"Translation: {self.models.translate(input_text)}"
            self.translation_photo_label.config(text=translated_text)

            # Save translation to history
            self.history.save_translation(input_text, translated_text)

    def take_screenshot(self):
        self.root.iconify()
        time.sleep(1)  # Allow the app to minimize before taking the screenshot
        screenshot = pyautogui.screenshot()
        self.root.deiconify()

        screenshot.save("files/images/screenshot.png")
        cropped_image = self.crop_screenshot_window("files/images/screenshot.png")

        if cropped_image:
            cropped_image.save("files/images/screenshot2.png")

        input_text = self.models.text_from_image("files/images/screenshot.png")
        translated_text = f"Translation: {self.models.translate(input_text)}"
        self.translation_screenshot_label.config(text=translated_text)

        # Save translation to history
        self.history.save_translation(input_text, translated_text)

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
                if event.x < self.start_x:
                    self.start_x, self.rect[3] = event.x, event.y
                elif event.y < self.start_y:
                    self.start_y, self.rect[2] = event.y, event.x
                else:
                    self.rect[2], self.rect[3] = event.x, event.y  # Set the bottom-right corner

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


