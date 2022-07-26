import tkinter as tk
from tkinter import filedialog

def ask_directory():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askdirectory()
    return file_path

def ask_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path
