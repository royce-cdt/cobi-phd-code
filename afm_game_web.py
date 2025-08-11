from flask import Flask, render_template, request, jsonify
import os
import cv2
import pandas as pd
from skimage import filters, measure
from scipy import ndimage as ndi
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from matplotlib import cm

app = Flask(__name__)

# Folder containing the images
folder_path = 'afm_data/'
afm_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.png')]

# Global variables to store state
current_image_index = 0
current_spot_index = {}
spots_data = {}
undo_stack = []

def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Fit a plane to the image
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    A = np.c_[x.ravel(), y.ravel(), np.ones_like(x.ravel())]
    coeff, _, _, _ = np.linalg.lstsq(A, image.ravel(), rcond=None)
    plane = (coeff[0] * x + coeff[1] * y + coeff[2]).reshape(image.shape)

    # Set any pixel with a value higher than the plane to the level of the plane
    filtered_image = np.minimum(image, plane)

    # Apply the thresholding
    threshold_image = filters.threshold_local(filtered_image, block_size=19, offset=12)
    dark_spots_image = filtered_image < threshold_image

    # Label the dark spots
    labeled_spots, num_spots = ndi.label(dark_spots_image)

    # Filter to spots between 5 and 100 pixels
    spots = measure.regionprops(labeled_spots, intensity_image=image)
    spots_filtered = [spot for spot in spots if 5 <= spot.area <= 100]

    return image, spots_filtered

@app.route('/')
def index():
    global current_image_index, afm_files, current_spot_index
    if current_image_index >= len(afm_files):
        return "No more images to process."

    image_path = afm_files[current_image_index]
    image, spots = process_image(image_path)

    # Apply colormap
    colormap = cm.afmhot
    colored_image = colormap(image / 255.0)
    colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)

    # Save spots for current image
    spots_data[current_image_index] = spots

    # Get which spot to highlight
    spot_idx = current_spot_index.get(current_image_index, 0)
    if spot_idx < len(spots):
        spot = spots[spot_idx]
        cy, cx = map(int, spot.centroid)
        radius = int(np.sqrt(spot.area / np.pi)) + 5
        cv2.circle(colored_image, (cx, cy), radius, (0, 0, 255), 2)

    # Convert image to base64
    pil_image = Image.fromarray(colored_image)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return render_template('index.html', image_base64=image_base64, spots=len(spots), spot_index=spot_idx)



@app.route('/categorise', methods=['POST'])
def categorise():
    global current_image_index, spots_data, undo_stack, current_spot_index

    data = request.json
    try:
        spot_index = int(data.get('spot_index', -1))
    except (ValueError, TypeError):
        spot_index = -1  # Handle if spot_index is not a number

    category = data.get('category')

    if current_image_index in spots_data:
        spots = spots_data[current_image_index]
        if 0 <= spot_index < len(spots):
            spot = spots[spot_index]
            undo_stack.append((spot_index, category))
            print(f"Spot {spot_index} categorised as {category}.")
            current_spot_index[current_image_index] = spot_index + 1
            return jsonify({'success': True})

    return jsonify({'success': False})



@app.route('/undo', methods=['POST'])
def undo():
    global undo_stack
    if undo_stack:
        last_action = undo_stack.pop()
        print(f"Undo last action: {last_action}")
        return jsonify({'success': True})
    return jsonify({'success': False})

@app.route('/next_image', methods=['POST'])
def next_image():
    global current_image_index
    current_image_index += 1
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)
