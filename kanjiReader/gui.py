import tkinter as tk
from tkinter import ttk, filedialog, messagebox

def translate_text():
    input_text = text_entry.get("1.0", tk.END).strip()
    # Tu wstaw model tłumaczenia gnojku
    translated_text = f"Tłumaczenie: {input_text}"
    translation_label.config(text=translated_text)

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Tu wstaw logikę wysyłania zdjęcia do back-endu
        translated_text = "UwU"
        #messagebox.showinfo("Sukces", f"Zdjęcie zostało wysłane: {file_path}")
        translation_photo_label.config(text=translated_text)


#Tworzenie głównego okna
root = tk.Tk()
root.title("Aplikacja GUI")

#Obsługa zakładek
tab_control = ttk.Notebook(root)
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab_control.add(tab1, text='Tłumaczenie')
tab_control.add(tab2, text='Zdjęcie')
tab_control.pack(expand=1, fill='both')

#Zakładka 1
text_entry = tk.Text(tab1, height=10, width=50)
text_entry.pack(pady=10)
translate_button = tk.Button(tab1, text="Przetłumacz", command=translate_text)
translate_button.pack(pady=5)
translation_label = tk.Label(tab1, text="")
translation_label.pack(pady=10)

#Zakładka 2
upload_button = tk.Button(tab2, text="Wstaw zdjęcie", command=upload_image)
upload_button.pack(pady=20)
translation_photo_label = tk.Label(tab2, text="")
translation_photo_label.pack(pady=10)

#Uruchomienie aplikacji
root.mainloop()