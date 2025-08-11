import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Select folder containing the images ===
folder_path = filedialog.askdirectory(title="Select Folder Containing PNG Files")
afm_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.png')]
print(f"Selected folder: {folder_path}")
print(f"AFM files: {afm_files}")

# === Functions ===

def enhance_contrast(image):
    lower_percentile = 0.05
    upper_percentile = 99.95
    img_array = image
    threshold = np.percentile(img_array, lower_percentile)
    max_val = np.percentile(img_array, upper_percentile)
    clipped = np.clip(img_array, threshold, max_val)
    scaled = (clipped - threshold) / (max_val - threshold) * 255
    return scaled.astype(np.uint8)

def update_counter_label(manual=False):
    if manual:
        counter_label.config(text=f"Manual Categorisation. Click any undetected spots.")

def on_canvas_click(event):
    global spot_data
    x, y = event.x, event.y
    popup = tk.Toplevel(root)
    popup.title("Categorise Spot")
    popup.geometry("300x150")

    label = tk.Label(popup, text=f"Categorise the spot at ({x}, {y}):", font=("Arial", 12))
    label.pack(pady=10)

    def categorise_new_spot(category):
        global spot_data
        spot_data = pd.concat([spot_data, pd.DataFrame({
            'x': [x],
            'y': [y],
            'size': [None],
            'category': [category]
        })], ignore_index=True)
        spot_data.to_csv(csv_file_path, index=False)
        popup.destroy()

    ttk.Button(popup, text="Large", command=lambda: categorise_new_spot("large")).pack(side=tk.LEFT, padx=20, pady=20)
    ttk.Button(popup, text="Small", command=lambda: categorise_new_spot("small")).pack(side=tk.RIGHT, padx=20, pady=20)

def highlight_all_spots():
    canvas.delete("highlight")
    for _, row in spot_data.iterrows():
        x, y = row['x'], row['y']
        canvas.create_oval(x-8, y-8, x+8, y+8, outline="blue", width=2, tags="highlight")
    canvas.update()

def finish_manual_categorisation():
    print("Manual categorisation finished.")
    root.destroy()

def start_manual_categorisation():
    update_counter_label(manual=True)
    highlight_all_spots()
    canvas.bind("<Button-1>", on_canvas_click)
    ttk.Button(button_frame, text="Finish", command=finish_manual_categorisation).pack(side=tk.LEFT, padx=5, pady=5)

# === Main loop over images ===
for idx, afm_file in enumerate(afm_files):
    image = cv2.imread(afm_file, cv2.IMREAD_GRAYSCALE)
    if image.min() >= 0.0 and image.max() <= 1.0:
        image = (image * 65535).astype(np.uint16)

    image1 = enhance_contrast(image)

    csv_file_path = afm_file.replace('.png', '_categorised.csv')

    # Load existing CSV
    if os.path.exists(csv_file_path):
        spot_data = pd.read_csv(csv_file_path)
    else:
        spot_data = pd.DataFrame(columns=['x', 'y', 'size', 'category'])
        print(f"No existing CSV found for {afm_file}, starting with empty DataFrame.")

    root = tk.Tk()
    root.title(f"Manual Spot Addition - Image {idx + 1} of {len(afm_files)}")

    canvas = tk.Canvas(root, width=image.shape[1], height=image.shape[0], bg="white")
    canvas.pack()

    image1_pil = Image.fromarray((plt.cm.afmhot(image1)[:, :, :3] * 255).astype(np.uint8))
    image1_tk = ImageTk.PhotoImage(image1_pil)
    canvas.create_image(0, 0, anchor=tk.NW, image=image1_tk)

    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X)

    counter_label = tk.Label(root, text="", font=("Arial", 12))
    counter_label.pack()

    start_manual_categorisation()
    root.mainloop()
