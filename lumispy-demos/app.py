from flask import Flask
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pickle
from skimage import measure, filters, morphology, segmentation
from scipy import ndimage as ndi

# Initialize the Flask server
server = Flask(__name__)

# Initialize the Dash app
app = dash.Dash(__name__, server=server, url_base_pathname='/')

# Getting back the objects:
with open(r'C:\Users\cobia\OneDrive - University of Cambridge\CL\COBI20250707\HYP-LONGEND00\obj.pkl', 'rb') as f:
    MI_sub, m_centre2_data, m_intensity_data = pickle.load(f)

# Create the initial figure
fig = px.imshow(MI_sub, color_continuous_scale='viridis')
fig.update_layout(
    title=dict(text="CL Intensity Map (TDs Contoured)", x=0.5, xanchor='center'),
    clickmode='event+select', coloraxis_colorbar=dict(title='CL Intensity (a.u.)'), plot_bgcolor='white')

# Step 1: Detect dark spots using a local threshold
threshold = filters.threshold_local(MI_sub, block_size=21, offset=5)
dark_spots = MI_sub < threshold

# Step 2: Clean the binary mask (optional but recommended)
dark_spots = morphology.remove_small_objects(dark_spots, min_size=4)  # filter noise
dark_spots = morphology.binary_closing(dark_spots, morphology.disk(2))  # close small holes



# # Step 3: Compute distance transform (inverted because we want distance inside blobs)
# distance = ndi.distance_transform_edt(dark_spots)

# # Step 4: Find local maxima of distance map as seeds
# local_max = morphology.local_maxima(distance)
# markers, _ = ndi.label(local_max)

# # Step 5: Apply watershed on the inverted distance map
# labeled_spots = segmentation.watershed(-distance, markers, mask=dark_spots)

# Label the dark spots (if not using watershed)
labeled_spots, num_spots = ndi.label(dark_spots)

# Find the coordinates of the spots
spots = measure.regionprops(labeled_spots)

print(f"Initial spot count: {len(spots)}")

## -------- TESTING SPLITTING HIGHLY ECCENTRIC SPOTS --------
# from skimage.draw import ellipse
# from skimage.util import img_as_ubyte

# # Initialize new label map
# new_label_map = np.zeros_like(labeled_spots, dtype=np.int32)
# label_counter = 1  # Start labeling from 1

# for region in spots:
#     if region.eccentricity <= 0.5:
#         # Accept as-is
#         new_label_map[labeled_spots == region.label] = label_counter
#         label_counter += 1
#     else:
#         # Extract the sub-region around the eccentric blob
#         minr, minc, maxr, maxc = region.bbox
#         region_mask = labeled_spots[minr:maxr, minc:maxc] == region.label

#         # Compute distance transform inside this blob
#         distance_local = ndi.distance_transform_edt(region_mask)

#         # Get local maxima as markers
#         local_max = morphology.local_maxima(distance_local)
#         markers, _ = ndi.label(local_max)

#         # Apply local watershed
#         split_labels = segmentation.watershed(-distance_local, markers, mask=region_mask)

#         # Insert split labels back into the global label map
#         for split_val in np.unique(split_labels):
#             if split_val == 0:
#                 continue  # background
#             mask = split_labels == split_val
#             new_label_map[minr:maxr, minc:maxc][mask] = label_counter
#             label_counter += 1

# # Optional: measure new spots
# spots = measure.regionprops(new_label_map)
# print(f"Final spot count: {len(spots)}")

## --------- END OF TESTING SPLITTING HIGHLY ECCENTRIC SPOTS --------

## --------- TESTING SPLITTING HIGHLY ECCENTRIC SPOTS ITERATIVELY --------

# from skimage import morphology, segmentation, measure
# from scipy import ndimage as ndi
# import numpy as np

# # Settings
# eccentricity_threshold = 0.5
# max_iterations = 20
# min_seed_distance = 1  # Minimum distance between seed points in pixels

# # Start with empty new label map
# new_label_map = np.zeros_like(labeled_spots, dtype=np.int32)
# label_counter = 1

# for region in spots:
#     mask_full = labeled_spots == region.label
#     continue
#     if region.eccentricity <= eccentricity_threshold:
#         new_label_map[mask_full] = label_counter
#         label_counter += 1
#         continue

#     # Begin recursive splitting
#     split_mask = mask_full.copy()
#     for iteration in range(max_iterations):
#         # Bounding box
#         minr, minc, maxr, maxc = region.bbox
#         region_mask = split_mask[minr:maxr, minc:maxc]

#         # Distance transform
#         distance = ndi.distance_transform_edt(region_mask)

#         # Find local maxima in distance map
#         # Reduce the minimum distance between peaks at each iteration
#         min_distance = max(1, min_seed_distance - iteration)
#         local_max = morphology.local_maxima(distance)
#         markers, _ = ndi.label(local_max)

#         # Apply watershed
#         labels_local = segmentation.watershed(-distance, markers, mask=region_mask)

#         # Analyze subregions
#         subregions = measure.regionprops(labels_local)

#         # Check if all subregions meet the eccentricity condition
#         all_good = all(r.eccentricity <= eccentricity_threshold for r in subregions)

#         if all_good or iteration == max_iterations - 1:
#             # Transfer labeled subregions to global map
#             for r in subregions:
#                 if r.area < 5:
#                     continue  # Skip tiny fragments
#                 sub_mask = labels_local == r.label
#                 new_label_map[minr:maxr, minc:maxc][sub_mask] = label_counter
#                 label_counter += 1
#             break
#         else:
#             # Recurse into next iteration with the current split mask
#             split_mask[minr:maxr, minc:maxc] = labels_local > 0

# print(f"Final circular spot count: {label_counter - 1}")
# # print(f"Maximum final eccentricity: {max(r.eccentricity for r in measure.regionprops(new_label_map))}")

# ## --------- END OF TESTING SPLITTING HIGHLY ECCENTRIC SPOTS ITERATIVELY --------




# Add contours for each spot
for spot in spots:
    contour = measure.find_contours(labeled_spots == spot.label, 0.5)
    for n, contour in enumerate(contour):
        fig.add_trace(go.Scatter(x=contour[:, 1], y=contour[:, 0], mode='lines', line=dict(color='red', width=1)))

# Add a trace for highlighting the selected spot
highlight_trace = go.Scatter(x=[], y=[], mode='lines', line=dict(color='yellow', width=2), name='Highlight')
fig.add_trace(highlight_trace)

# Hide the legend
fig.update_layout(showlegend=False)

# Update the hover template to show x, y position, and intensity value
fig.update_traces(hovertemplate='x: %{x:.0f}<br>y: %{y:.0f}<extra></extra>')

# Define the layout of the app
app.layout = html.Div([
    html.H1("CL Intensity Map Analysis", style={'font-family': 'Open Sans, verdana, arial, sans-serif'}),
    html.P("Data was collected from silane-treated GaN sample at room temperature.", style={'font-family': 'Open Sans, verdana, arial, sans-serif'}),
    html.P("Dislocations can be seen as dark spots (circled in red)", style={'font-family': 'Open Sans, verdana, arial, sans-serif'}),
    html.P("Click on a spot in the CL Intensity Map to see detailed information and zoomed-in views.", style={'font-family': 'Open Sans, verdana, arial, sans-serif'}),
    html.Div([
        dcc.Graph(id='main-plot', figure=fig),
        dcc.Graph(id='arrow-plot')
    ], style={'display': 'flex'}),
    html.Div([
        dcc.Graph(id='zoomed-plot'),
        dcc.Graph(id='zoomed-plot-centre2')
    ], style={'display': 'flex'})
], style={'font-family': 'Open Sans, verdana, arial, sans-serif'})

# Calculate the length of the scale bar in pixels
scale_bar_length_um = 1  # 1 µm
scale_bar_length_px = scale_bar_length_um * MI_sub.shape[1] / 4.

# Add scale bar to the main figure
fig.add_shape(
    type="line",
    x0=10, y0=MI_sub.shape[0] - 10,
    x1=10 + scale_bar_length_px, y1=MI_sub.shape[0] - 10,
    line=dict(color="white", width=3)
)
fig.add_annotation(
    x=10 + scale_bar_length_px / 2, y=MI_sub.shape[0] - 20,
    text="1 µm",
    showarrow=False,
    font=dict(color="white")
)

# Define a function to add scale bar to zoomed-in figures
def add_scale_bar(fig, zoomed_area_shape):
    fig.add_shape(
        type="line",
        x0=1, y0=zoomed_area_shape[0] - 1,
        x1=1 + (scale_bar_length_px / 10), y1=zoomed_area_shape[0] - 1,
        line=dict(color="white", width=3)
    )
    fig.add_annotation(
        x=1 + (scale_bar_length_px / 10) / 2, y=zoomed_area_shape[0] - 2,
        text="100 nm",
        showarrow=False,
        font=dict(color="white")
    )
    return fig

# Update the callback to add scale bars to zoomed-in plots
@app.callback(
    [Output('zoomed-plot', 'figure'),
     Output('zoomed-plot-centre2', 'figure'),
     Output('arrow-plot', 'figure'),
     Output('main-plot', 'figure')],
    [Input('main-plot', 'clickData')]
)
def update_zoomed_plots(clickData):
    if clickData is None:
        # Initialize empty figures with colorbars
        empty_fig_MI_sub = px.imshow(np.zeros((1, 1)), color_continuous_scale='viridis')
        empty_fig_MI_sub.update_layout(
            title=dict(text="Zoomed-in Area of CL Intensity", x=0.5, xanchor='center'),
            coloraxis_colorbar=dict(title='CL Intensity (arb. units)'))
        
        empty_fig_centre2 = px.imshow(np.zeros((1, 1)), color_continuous_scale='RdBu')
        empty_fig_centre2.update_layout(
            title=dict(text="Zoomed-in Area of GaN NBE CL Peak Shift", x=0.5, xanchor='center'),
            coloraxis_colorbar=dict(title='GaN NBE CL Peak Shift (eV)'))
        
        empty_arrow_fig = go.Figure()
        empty_arrow_fig.update_layout(
            title=dict(text="Direction of blue/red-shift\ngradient around spot", x=0.5, xanchor='center'),
            width=fig.layout.height, height=fig.layout.height)
        return empty_fig_MI_sub, empty_fig_centre2, empty_arrow_fig, fig  # Return initialized figures if no click data

    # Get the coordinates of the clicked point
    x = int(clickData['points'][0]['x']) + 1
    y = int(clickData['points'][0]['y']) + 1

    # Get the label of the spot for that point
    label = labeled_spots[y, x]

    if label > 0:
        spot = spots[label - 1]
        
        # Get the centroid of the spot
        cy, cx = spot.centroid
        cy, cx = int(cy), int(cx)

        # Define the zoomed-in area
        zoom_px = 10
        zoomed_area_MI_sub = MI_sub[max(cy-zoom_px, 0):cy+zoom_px, max(cx-zoom_px, 0):cx+zoom_px]
        zoomed_area_centre2 = m_centre2_data[max(cy-zoom_px, 0):cy+zoom_px, max(cx-zoom_px, 0):cx+zoom_px]

        # Fit a plane to zoomed_area_centre2
        X, Y = np.meshgrid(np.arange(zoomed_area_centre2.shape[1]), np.arange(zoomed_area_centre2.shape[0]))
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = zoomed_area_centre2.flatten()
        A = np.c_[X_flat, Y_flat, np.ones(X_flat.shape)]
        C, _, _, _ = np.linalg.lstsq(A, Z_flat, rcond=None)
        plane = C[0] * X + C[1] * Y + C[2]
        
        # Calculate the gradient of the plane
        grad_y, grad_x = np.gradient(plane)
        avg_grad_y = np.mean(grad_y)
        avg_grad_x = np.mean(grad_x)

        avg_grad_x_norm = avg_grad_x * 0.8 / np.sqrt(avg_grad_x**2 + avg_grad_y**2)
        avg_grad_y_norm = -avg_grad_y * 0.8 / np.sqrt(avg_grad_x**2 + avg_grad_y**2)

        # Create the arrow plot
        arrow_fig = go.Figure()

        # Add the arrow with a triangular point
        # arrow_fig.add_trace(go.Scatter(
        #     x=[0, avg_grad_x_norm],
        #     y=[0, avg_grad_y_norm],
        #     mode='lines+markers',
        #     line=dict(color='black', width=2),
        #     marker=dict(size=10, symbol= "arrow-bar-up")
        # ))

        # arrow_fig.update_layout(
        #     shapes=[
        #         dict(
        #             type="line",
        #             x0=0, y0=0,
        #             x1=avg_grad_x_norm, y1=avg_grad_y_norm,
        #             line=dict(color="black", width=2)
        #         ),
        #         dict(
        #             type="path",
        #             path="M {} {} L {} {} L {} {} Z".format(
        #                 avg_grad_x_norm - 0.1, avg_grad_y_norm - 0.05,
        #                 avg_grad_x_norm, avg_grad_y_norm,
        #                 avg_grad_x_norm - 0.1, avg_grad_y_norm + 0.05
        #             ),
        #             fillcolor="black",
        #             line=dict(color="black")
        #         )
        #     ]
        # )

        
        # Define the arrow start and end points
        x_start, y_start = 0, 0
        x_end, y_end = avg_grad_x_norm, avg_grad_y_norm

        # Calculate the angle of the arrow
        theta = np.arctan2(y_end - y_start, x_end - x_start)

        # Define the base points of the triangle relative to the tip (before rotation)
        arrow_size = 0.1  # Adjust for the size of the arrowhead
        base_left = (-arrow_size, -arrow_size / 2)
        base_right = (-arrow_size, arrow_size / 2)
        tip = (0, 0)

        # Function to rotate a point
        def rotate_point(x, y, angle):
            x_rot = x * np.cos(angle) - y * np.sin(angle)
            y_rot = x * np.sin(angle) + y * np.cos(angle)
            return x_rot, y_rot

        # Rotate the base points
        base_left_rot = rotate_point(*base_left, theta)
        base_right_rot = rotate_point(*base_right, theta)

        # Translate the points to the tip of the arrow
        base_left_rot = (base_left_rot[0] + x_end, base_left_rot[1] + y_end)
        base_right_rot = (base_right_rot[0] + x_end, base_right_rot[1] + y_end)
        tip = (x_end, y_end)

        # Create the path for the arrowhead
        path = f"M {base_left_rot[0]} {base_left_rot[1]} L {tip[0]} {tip[1]} L {base_right_rot[0]} {base_right_rot[1]} Z"

        # Add the arrowhead shape to the plot
        arrow_fig.update_layout(
            shapes=[
                dict(
                    type="line",
                    x0=x_start, y0=y_start,
                    x1=x_end, y1=y_end,
                    line=dict(color="black", width=2)
                ),
                dict(
                    type="path",
                    path=path,
                    fillcolor="black",
                    line=dict(color="black")
                )
            ]
        )



        # Add the text
        arrow_fig.add_annotation(
            x=0.5, y=0.5,
            text=f"Area of the spot: {spot.area} px<sup>2</sup>",
            showarrow=False,
            font=dict(size=12),
            xref="paper", yref="paper",
            xanchor="center", yanchor="middle",
            bgcolor="white", opacity=0.5
        )

        # Update the layout
        arrow_fig.update_layout(
            xaxis=dict(range=[-1, 1], showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[-1, 1], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
            title=dict(text=f"Direction of blue/red-shift\ngradient around spot at ({cy}, {cx})", x=0.5, xanchor='center'),
            width=fig.layout.height, height=fig.layout.height
        )

        # Highlight the selected spot
        contour = measure.find_contours(labeled_spots == label, 0.5)
        highlight_x = []
        highlight_y = []
        for n, contour in enumerate(contour):
            highlight_x.extend(contour[:, 1])
            highlight_y.extend(contour[:, 0])
        fig.data[-1].x = highlight_x
        fig.data[-1].y = highlight_y

    else:
        # Keep the existing zoomed figures and remove highlight
        fig.data[-1].x = []
        fig.data[-1].y = []
        return dash.no_update, dash.no_update, dash.no_update, fig

    # Create the zoomed-in plots
    zoomed_fig_MI_sub = px.imshow(zoomed_area_MI_sub, color_continuous_scale='viridis')
    zoomed_fig_MI_sub.update_layout(
        title=dict(text="Zoomed-in Area of CL Intensity", x=0.5, xanchor='center'),
        coloraxis_colorbar=dict(title='CL Intensity (arb. units)'))
    zoomed_fig_MI_sub = add_scale_bar(zoomed_fig_MI_sub, zoomed_area_MI_sub.shape)

    zoomed_fig_centre2 = px.imshow(zoomed_area_centre2, color_continuous_scale='RdBu')
    zoomed_fig_centre2.update_layout(
        title=dict(text="Zoomed-in Area of GaN CL Peak Shift", x=0.5, xanchor='center'),
        coloraxis_colorbar=dict(title='GaN CL Peak Shift (eV)'))
    zoomed_fig_centre2 = add_scale_bar(zoomed_fig_centre2, zoomed_area_centre2.shape)

    return zoomed_fig_MI_sub, zoomed_fig_centre2, arrow_fig, fig

# Run the app
if __name__ == '__main__':
    server.run(debug=True)
