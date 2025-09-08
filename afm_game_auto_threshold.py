import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import cv2
import pandas as pd
from skimage import filters, measure
from scipy import ndimage as ndi
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np


# Select folder containing the images
# Prompt user to select a folder
folder_path = filedialog.askdirectory(title="Select Folder Containing PNG Files")

# Make a list of full filepaths for all PNG files in the selected folder
afm_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.png')]

print(f"Selected folder: {folder_path}")
print(f"AFM files: {afm_files}")



from PIL import Image
import numpy as np

def enhance_contrast(image):
    """
    Enhance contrast of an 8-bit grayscale image by:
    - Setting the brightest pixel(s) to white (255)
    - Setting the darkest 10% of pixels to black (0)

    Args:
        image : input image file
    """

    lower_percentile=0.5
    upper_percentile=99.5

    img_array = image

    # Get percentiles
    threshold = np.percentile(img_array, lower_percentile)
    max_val = np.percentile(img_array, upper_percentile)

    print(f"Lower threshold: {threshold}")
    print(f"Upper threshold: {max_val}")

    # Clip values outside the range
    clipped = np.clip(img_array, threshold, max_val)

    # Scale to 0-255
    scaled = (clipped - threshold) / (max_val - threshold) * 255

    # Convert to uint8
    output_image = scaled.astype(np.uint8)

    return output_image



# Function to update the counter label
def update_counter_label(manual=False):
    if manual:
        counter_label.config(text=f"Manual Categorisation. Click any undetected spots.")
    else:
        counter_label.config(text=f"Spot {current_spot_index + 1} of {total_spots}")

# Function to handle user clicks for undetected spots
def on_canvas_click(event):
    global spot_data, undo_stack
    x, y = event.x, event.y
    print(f"User clicked at ({x}, {y}).")

    # Create a pop-up window for categorisation
    popup = tk.Toplevel(root)
    popup.title("Categorise Spot")
    popup.geometry("300x150")

    label = tk.Label(popup, text=f"Categorise the spot at ({x}, {y}):", font=("Arial", 12))
    label.pack(pady=10)

    def categorise_new_spot(category):
        global spot_data, undo_stack
        print(f"Spot at ({x}, {y}) labeled as '{category}'.")
        # Add the new spot to the DataFrame
        spot_data = pd.concat([spot_data, pd.DataFrame({
            'x': [x],
            'y': [y],
            'size': [None],  # Size is unknown for manually added spots
            'category': [category]
        })], ignore_index=True)
        # Push the categorised spot onto the undo stack
        undo_stack.append(({'x': x, 'y': y}, category))
        # Save to CSV file
        spot_data.to_csv(csv_file_path, index=False)
        popup.destroy()

    # Buttons for categorisation
    large_button = ttk.Button(popup, text="Large", command=lambda: categorise_new_spot("large"))
    large_button.pack(side=tk.LEFT, padx=20, pady=20)

    small_button = ttk.Button(popup, text="Small", command=lambda: categorise_new_spot("small"))
    small_button.pack(side=tk.RIGHT, padx=20, pady=20)

# Function to highlight all spots
def highlight_all_spots():
    canvas.delete("highlight")
    for spot in spots_image1:
        x, y = spot.centroid[1], spot.centroid[0]
        canvas.create_oval(x-8, y-8, x+8, y+8, outline="blue", width=2, tags="highlight")
    canvas.update()

# Function to finish the manual categorisation phase
def finish_manual_categorisation():
    print("Manual categorisation finished.")
    root.destroy()

# After all spots are categorised
def start_manual_categorisation():
    print("All spots have been categorised. Highlighting all spots.")
    update_counter_label(manual=True)
    highlight_all_spots()
    canvas.bind("<Button-1>", on_canvas_click)

    # Add a Finish button
    finish_button = ttk.Button(button_frame, text="Finish", command=finish_manual_categorisation)
    finish_button.pack(side=tk.LEFT, padx=5, pady=5)

# Modify the categorise_spot function to call start_manual_categorisation
def categorise_spot(category, spot, canvas, root):
    global spot_data, undo_stack, current_spot_index, first_spot, spots_iter
    if category == "skip":
        print('Skipping categorisation function for undo')
    elif category == "not_a_spot":
        print(f"Spot at ({spot.centroid[1]:.1f}, {spot.centroid[0]:.1f}) labeled as 'Not a Spot'.")
        undo_stack.append((spot, category))
    elif category == "combination":
        print(f"Spot at ({spot.centroid[1]:.1f}, {spot.centroid[0]:.1f}) labeled as 'Combination of Multiple Spots'.")
        spot_data = pd.concat([spot_data, pd.DataFrame({
            'x': [spot.centroid[1]],
            'y': [spot.centroid[0]],
            'size': [spot.area],
            'category': [category]
        })], ignore_index=True)
        undo_stack.append((spot, category))
    else:
        spot_data = pd.concat([spot_data, pd.DataFrame({
            'x': [spot.centroid[1]],
            'y': [spot.centroid[0]],
            'size': [spot.area],
            'category': [category]
        })], ignore_index=True)
        undo_stack.append((spot, category))
        print(f"Spot at ({spot.centroid[1]:.1f}, {spot.centroid[0]:.1f}) labeled as '{category}'.")

    spot_data.to_csv(csv_file_path, index=False)

    try:
        next_spot = next(spots_iter)
        first_spot = next_spot
        current_spot_index += 1
        update_counter_label()
        highlight_spot(next_spot, canvas)
    except StopIteration:
        start_manual_categorisation()

# Function to undo the last categorisation
def undo_last_action(canvas):
    global spot_data, undo_stack, current_spot_index, spots_iter
    if undo_stack:
        last_spot, last_category = undo_stack.pop()
        # Remove the last categorised spot from the DataFrame
        spot_data = spot_data[~((spot_data['x'] == last_spot.centroid[1]) & 
                                (spot_data['y'] == last_spot.centroid[0]) & 
                                (spot_data['category'] == last_category))]
        print(f"Undo: Removed spot at ({last_spot.centroid[1]:.1f}, {last_spot.centroid[0]:.1f}) labeled as '{last_category}'.")
        # Highlight the undone spot again
        highlight_spot(last_spot, canvas)
        # Re-add the undone spot to the iterator
        new_list = [last_spot] + list(spots_iter)
        print("New List: " + str(new_list))
        spots_iter = iter(new_list)
        current_spot_index -= 1
        update_counter_label()
        categorise_spot("skip", last_spot, canvas, root)
    else:
        print("Nothing to undo.")        


# Function to highlight a spot on the canvas
def highlight_spot(spot, canvas):
    canvas.delete("highlight")
    x, y = spot.centroid[1], spot.centroid[0]
    canvas.create_oval(x-8, y-8, x+8, y+8, outline="blue", width=2, tags="highlight")
    canvas.update()

def find_best_offset_first_derivative_peak(offsets, num_spots_list, peak_prominence=1.0, drop_threshold=20):
    """
    Finds the best offset based on the first local maximum in the smoothed derivative.
    - peak_prominence: how prominent a derivative peak must be to count
    - drop_threshold: fraction of the peak height where we consider it to have 'dropped back down'
    Ignores offsets where num_spots is NaN or zero.
    """
    offsets = np.array(offsets)
    num_spots = np.array(num_spots_list, dtype=float)

    # Mask invalid
    valid_mask = (~np.isnan(num_spots)) & (num_spots > 0)
    if not np.any(valid_mask):
        return None #, None, None

    offsets_valid = offsets[valid_mask]
    num_spots_valid = num_spots[valid_mask]

    # Compute and smooth derivative
    deriv = np.diff(num_spots_valid) / np.diff(offsets_valid)
    deriv_smooth = np.convolve(deriv, np.ones(5)/5, mode='same')

    # Find peaks in the derivative
    peaks, _ = find_peaks(deriv_smooth, prominence=peak_prominence)
    if len(peaks) == 0:
        # return offsets_valid, num_spots_valid, ("no_peak", offsets_valid[np.argmax(num_spots_valid)], deriv_smooth)
        return offsets_valid[np.argmax(num_spots_valid)]

    first_peak_idx = peaks[0]
    peak_height = deriv_smooth[first_peak_idx]

    print('First peak height:', peak_height)

    # Find the right edge where derivative drops below threshold fraction of peak height
    right_edge_idx = first_peak_idx
    print('Looking for next point where derivative < ', -drop_threshold + peak_height)
    for j in range(first_peak_idx + 1, len(deriv_smooth)):
        
        if deriv_smooth[j] < -drop_threshold + peak_height:
            right_edge_idx = j
            print('Found the point at: ', deriv_smooth[j])
            break

    best_offset = offsets_valid[right_edge_idx]
    # return offsets_valid, num_spots_valid, ("derivative_peak", best_offset, deriv_smooth)    
    return best_offset



for idx, afm_file in enumerate(afm_files):

    # Define image
    # image = io.imread(afm_file, as_gray=True)
    image = cv2.imread(afm_file, cv2.IMREAD_GRAYSCALE)

    print(f"Image min: {np.min(image)}, max: {np.max(image)}")

    # Apply local thresholding to detect spots
    # Fit a plane to the image
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    A = np.c_[x.ravel(), y.ravel(), np.ones_like(x.ravel())]
    coeff, _, _, _ = np.linalg.lstsq(A, image.ravel(), rcond=None)
    plane = (coeff[0] * x + coeff[1] * y + coeff[2]).reshape(image.shape)

    # Set any pixel with a value higher than the plane to the level of the plane
    filtered_image = np.minimum(image, plane)

    # Find the best offset for local thresholding
    offsets = list(range(1, 21)) * 5
    block_size = 19


    num_spots_list = []

    for offset_nm in offsets:
        local_thresh = filters.threshold_local(filtered_image, block_size=block_size, offset=offset_nm)
        dark_spots = filtered_image < local_thresh
        labeled_spots, _ = ndi.label(dark_spots)
        spot_sizes = np.bincount(labeled_spots.flat)[1:]
        num_spots = np.sum((spot_sizes >= 6) & (spot_sizes <= 199))
        if num_spots > 150:
            num_spots_list.append(np.nan)  # invalid
        else:
            num_spots_list.append(num_spots if num_spots > 0 else np.nan)    

    best_offset = find_best_offset_first_derivative_peak(
        offsets,
        num_spots_list
    )

    print(f"Best offset determined: {best_offset}")

    # Apply the thresholding
    threshold_image = filters.threshold_local(filtered_image, block_size=19, offset=best_offset)
    dark_spots_image = filtered_image < threshold_image

    # Label the dark spots
    labeled_spots, num_spots = ndi.label(dark_spots_image)

    # Filter to spots between 6 and 199 pixels
    spots = measure.regionprops(labeled_spots, intensity_image=image)
    spots_image1 = [spot for spot in spots if 6 <= spot.area <= 199]

    # Create an enhanced contrast image for display
    # image1 = enhance_contrast(image)
    image1 = image.copy()

    # Initialise a DataFrame to store categorised spots
    spot_data = pd.DataFrame(columns=['x', 'y', 'size', 'category'])

    # CSV file path to save categorised spots
    csv_file_path = afm_file.replace('.png', '_categorised.csv')

    # Stack to keep track of categorised spots for undo functionality
    undo_stack = []

    # Counter to track the current spot index
    current_spot_index = 0
    total_spots = len(spots_image1)

    # Create the main application window
    root = tk.Tk()
    root.title(f"Spot Categorisation - Image {idx + 1} of {len(afm_files)}")

    # Create a canvas to display the image
    canvas = tk.Canvas(root, width=image.shape[1], height=image.shape[0], bg="white")
    canvas.pack()

    # Convert the image to a format suitable for tkinter
    # image1_pil = Image.fromarray(image)
    image1_pil = Image.fromarray((plt.cm.afmhot(image1)[:, :, :3] * 255).astype(np.uint8))

    image1_tk = ImageTk.PhotoImage(image1_pil)
    canvas.create_image(0, 0, anchor=tk.NW, image=image1_tk)

    # Create a frame for buttons
    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X)

    # Create an iterator for the spots
    spots_iter = iter([spot for spot in spots_image1 if 5 <= spot.area <= 100])

    # Highlight the first spot
    try:
        global first_spot
        first_spot = next(spots_iter)
        highlight_spot(first_spot, canvas)
    except StopIteration:
        print("No spots to categorise.")
        # root.destroy()

    # Create a counter label
    counter_label = tk.Label(root, text=f"Spot 1 of {total_spots}", font=("Arial", 12))
    counter_label.pack()

    # Create buttons for categorisation
    categories = [('Large', 'large'),
                ('Small', 'small'),
                ('Not a Spot', 'not_a_spot'),
                ('Combination', 'combination')]
    for text, category in categories:
        button = ttk.Button(button_frame, text=text, command=lambda c=category: categorise_spot(c, first_spot, canvas, root))
        button.pack(side=tk.LEFT, padx=5, pady=5)
        button.pack(side=tk.LEFT, padx=5, pady=5)

    # Add an Undo button
    undo_button = ttk.Button(button_frame, text="Undo", command=lambda: undo_last_action(canvas))
    undo_button.pack(side=tk.LEFT, padx=5, pady=5)

    root.mainloop()
