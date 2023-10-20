import cv2, os
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

def add_axis_to_video(frame, length=100):
    """Add a normalized coordinate axis to a frame."""
    h, w, _ = frame.shape
    
    # Origin at upper-left corner
    origin = (50, 50)
    
    # Define x and y axis endpoints based on length
    x_axis_end = (origin[0] + length, origin[1])
    y_axis_end = (origin[0], origin[1] + length)  # Notice the change here
    
    # Draw axis lines
    cv2.line(frame, origin, x_axis_end, (0, 255, 0), 2)
    cv2.line(frame, origin, y_axis_end, (0, 0, 255), 2)
    
    # Axis labels
    cv2.putText(frame, "x", x_axis_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "y", y_axis_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    
def interpolate_nans(data):
    for row in data:
        nans = np.isnan(row)
        non_nans = ~nans

        # If the entire row is NaNs, handle it (e.g., set to zero or some default value)
        if np.all(nans):
            row[:] = 0  # or any other default value or method of handling
        else:
            # Interpolate NaNs as you do now
            row[nans] = np.interp(nans.nonzero()[0], non_nans.nonzero()[0], row[non_nans])

    return data


def plot_landmark_data(data, landmark_index, data_label, interpolated_data=None):
    """
    Plot landmark data for a given landmark index.

    Parameters:
    - data: The main data array (feature_coordinates in your code).
    - landmark_index: Index of the landmark to plot.
    - data_label: Title/Label for the main data.
    - interpolated_data: If provided, this data is plotted alongside the main data.
    """

    start_idx = 3 * landmark_index  # x-coordinate index
    landmark_coords = data[start_idx:start_idx + 3, :]  # x, y, z rows of the landmark

    # Create the plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot the x, y, z coordinates for the landmark
    for idx, coord in enumerate(['x', 'y', 'z']):
        # Thick line for the main data
        axes[idx].plot(landmark_coords[idx], label=data_label, color='blue', linestyle='-', linewidth=2)
        
        # If interpolated data is provided, plot it as well
        if interpolated_data is not None:
            interpolated_coords = interpolated_data[start_idx:start_idx + 3, :]
            # Thin open circle for interpolated data
            axes[idx].plot(interpolated_coords[idx], label='Interpolated Data', color='red', linestyle='', marker='o', markerfacecolor='none', markersize=6)
        
        axes[idx].legend()
        axes[idx].set_ylabel(f"{coord} Value")
        axes[idx].grid(True)

    axes[2].set_xlabel("Frame Index")
    fig.suptitle(f"Data for Landmark {landmark_index}")

    plt.tight_layout()
    plt.show()

def get_video_frame_rate(video_path):
    """Retrieve the frame rate of the video using ffprobe."""
    cmd_output = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]).decode('utf-8').strip()
    num, denom = map(int, cmd_output.split('/'))
    return num / denom

def find_video_for_landmark(landmark_filename, root_dir="vids"):
    # Extract just the filename without path
    filename_only = os.path.basename(landmark_filename)
    
    if "_raw_worldlandmarks" in filename_only:
        base_name = filename_only.replace("_raw_worldlandmarks", "")
    elif "_raw_landmarks" in filename_only:
        base_name = filename_only.replace("_raw_landmarks", "")
    else:
        base_name = filename_only  # fallback to original name if neither suffix is found

    base_name = os.path.splitext(base_name)[0]  # stripping off the file extension

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".mp4") and file.startswith(base_name):
                return os.path.join(subdir, file)
    return None

def resize_frame(frame, width):
    aspect_ratio = frame.shape[1] / frame.shape[0]
    new_width = width
    new_height = int(new_width / aspect_ratio)
    return cv2.resize(frame, (new_width, new_height))


def select_raw_landmark_files():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select the raw landmarks file", 
                                           filetypes=(("Numpy files", "*.npy"), ("All files", "*.*")))
    root.destroy()  # Destroy the root window
    if not file_path:
        raise ValueError("No file was selected.")
    return [file_path]

def select_descriptors_to_visualize(descriptors):
    root = tk.Tk()
    root.title("Select Descriptors to Visualize")

    selected_descriptors = []

    desc_var_values = [tk.IntVar() for _ in descriptors]

    def on_select():
        selected_descriptors.extend([descriptors[i] for i in range(len(descriptors)) if desc_var_values[i].get() == 1])
        root.quit()

    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=1)

    # Add a canvas in that frame
    canvas = tk.Canvas(frame)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    # Add a scroll bar to the canvas
    scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Attach the canvas to the scrollbar
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    # Create a frame inside the canvas
    second_frame = tk.Frame(canvas)

    # Add that new frame to a window in the canvas
    canvas.create_window((0, 0), window=second_frame, anchor="nw")

    for i, desc in enumerate(descriptors):
        ttk.Checkbutton(second_frame, text=desc, variable=desc_var_values[i]).pack(anchor=tk.W, padx=10, pady=5)

    ttk.Button(second_frame, text="Select", command=on_select).pack(pady=20)

    root.mainloop()
    root.destroy()
    return selected_descriptors

# Angle calculation function
def angle_between_three_points(A, B, C):
    BA = A - B
    BC = C - B
    dot_product = np.dot(BA, BC)
    magnitude_BA = np.linalg.norm(BA)
    magnitude_BC = np.linalg.norm(BC)
    if magnitude_BA * magnitude_BC == 0:  # to handle division by zero
        return 0
    cos_theta = dot_product / (magnitude_BA * magnitude_BC)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(theta)

def calculate_angles(loaded_landmark_data, landmark_names, angle_definitions):
    angles_matrix = []
    for landmarks in angle_definitions.values():
        coords = [loaded_landmark_data[:, 4 * landmark_names.index(lm): 4 * landmark_names.index(lm) + 3] for lm in landmarks]
        processed_coords = [interpolate_nans(c.copy()) for c in coords]
        angles = [angle_between_three_points(*[pc[:, i] for pc in processed_coords]) for i in range(loaded_landmark_data.shape[0])]
        angles_matrix.append(angles)
    return np.array(angles_matrix)

