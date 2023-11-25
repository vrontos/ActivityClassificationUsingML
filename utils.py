import cv2, os
import numpy as np
import subprocess
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import ttk
from tkinter import filedialog
from collections import deque

DESIRED_LANDMARKS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# 0 - nose
# 1 - left eye (inner)
# 2 - left eye
# 3 - left eye (outer)
# 4 - right eye (inner)
# 5 - right eye
# 6 - right eye (outer)
# 7 - left ear
# 8 - right ear
# 9 - mouth (left)
# 10 - mouth (right)
# 11 - left shoulder
# 12 - right shoulder
# 13 - left elbow
# 14 - right elbow
# 15 - left wrist
# 16 - right wrist
# 17 - left pinky
# 18 - right pinky
# 19 - left index
# 20 - right index
# 21 - left thumb
# 22 - right thumb
# 23 - left hip
# 24 - right hip
# 25 - left knee
# 26 - right knee
# 27 - left ankle
# 28 - right ankle
# 29 - left heel
# 30 - right heel
# 31 - left foot index
# 32 - right foot index

LANDMARK_NAMES = [
    'nose', 'left eye (inner)', 'left eye', 'left eye (outer)', 'right eye (inner)', 'right eye',
    'right eye (outer)', 'left ear', 'right ear', 'mouth (left)', 'mouth (right)', 'left shoulder',
    'right shoulder', 'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left pinky',
    'right pinky', 'left index', 'right index', 'left thumb', 'right thumb', 'left hip', 'right hip',
    'left knee', 'right knee', 'left ankle', 'right ankle', 'left heel', 'right heel', 'left foot index',
    'right foot index'
]

DESIRED_LANDMARK_NAMES = [LANDMARK_NAMES[i] for i in DESIRED_LANDMARKS]

ANGLE_DEFINITIONS = {
    'angle_right_elbow': ['right shoulder', 'right elbow', 'right wrist'],
    'angle_right_shoulder': ['right elbow', 'right shoulder', 'right hip'],
    'angle_right_hip': ['right shoulder', 'right hip', 'right knee'],
    'angle_right_knee': ['right hip', 'right knee', 'right ankle'],
    'angle_left_elbow': ['left shoulder', 'left elbow', 'left wrist'],
    'angle_left_shoulder': ['left elbow', 'left shoulder', 'left hip'],
    'angle_left_hip': ['left shoulder', 'left hip', 'left knee'],
    'angle_left_knee': ['left hip', 'left knee', 'left ankle']
}

WINDOW_SIZE = 60  # Define the appropriate window size for video frames

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

def get_video_frame_rate(video_path):
    # TODO: The fps in Wyze v3 changes to daytime and nighttime, so consider this when training and testing the model
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

def select_raw_landmark_files(initial_dir=None):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    if initial_dir:
        initial_dir_path = os.path.join(os.getcwd(), initial_dir)
    else:
        initial_dir_path = os.getcwd()
        
    file_path = filedialog.askopenfilename(title="Select the raw landmarks file", 
                                           filetypes=(("Numpy files", "*.npy"), ("All files", "*.*")),
                                           initialdir=initial_dir_path)
    
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

def check_feature_health(final_feature_matrix, num_landmarks):
    # TODO: correct the visibility-angles bug and generally make it more beautiful.
    print("Shape of final feature matrix:", final_feature_matrix.shape)
    
    if np.isnan(final_feature_matrix).any():
        print("Warning: NaN values detected in the feature matrix!")
    else:
        print("No NaN values in the feature matrix.")
    
    print("Min and Max values in the feature matrix:")
    print(np.min(final_feature_matrix), np.max(final_feature_matrix))
    
    if np.isinf(final_feature_matrix).any():
        print("Warning: Infinite values detected in the feature matrix!")
    else:
        print("No Infinite values in the feature matrix.")
    
    min_visibility = np.min(final_feature_matrix[-num_landmarks:])
    max_visibility = np.max(final_feature_matrix[-num_landmarks:])
    print(f"Visibility value range: {min_visibility} to {max_visibility}")
    
def check_for_rows_or_cols_of_nans(data):
    # Convert to NumPy array if not already
    data_np = np.array(data)

    # Check for rows full of NaNs
    nan_rows = np.where(np.all(np.isnan(data_np), axis=1))[0]
    if nan_rows.size > 0:
        print(f"Warning: Rows {nan_rows} are full of NaNs.")

    # Check for columns full of NaNs
    nan_cols = np.where(np.all(np.isnan(data_np), axis=0))[0]
    if nan_cols.size > 0:
        print(f"Warning: Columns {nan_cols} are full of NaNs.")

def plot_feature_importances(clf, descriptors):
    # Get feature importances
    importances = clf.feature_importances_

    # Assuming 'descriptors' is a list of feature names
    statistical_features = ['mean', 'std', 'min', 'max']

    # Calculate importance for each statistical feature
    feature_importances = {}
    num_original_features = len(descriptors)
    for i, stat_feature in enumerate(statistical_features):
        start_idx = i * num_original_features
        end_idx = start_idx + num_original_features
        feature_importance = np.sum(importances[start_idx:end_idx])
        feature_importances[stat_feature] = feature_importance

    # Normalize importances so they sum to 1
    total_importance = sum(feature_importances.values())
    normalized_importances = {feature: importance / total_importance for feature, importance in feature_importances.items()}

    # Print and plot statistical feature importances
    print("Statistical Feature Importances:", normalized_importances)
    plt.figure()
    plt.bar(normalized_importances.keys(), normalized_importances.values())
    plt.ylabel('Importance')
    plt.title('Statistical Feature Importances')
    plt.show()

    # Feature Grouping and Importance Calculation
    feature_groups = {
        'position': [],
        'velocity': [],
        'distances': [],
        'visibility': [],
        'angles': [],
    }

    # Assign each descriptor to its feature group
    for descriptor in descriptors:
        for group in feature_groups:
            if group in descriptor:
                feature_groups[group].append(descriptor)

    # Calculate importance for each feature group
    group_importances = {}
    for group, features in feature_groups.items():
        indices = [descriptors.index(feature) for feature in features]
        group_importance = np.sum([importances[index] for index in indices])
        group_importances[group] = group_importance

    # Normalize importances for feature groups
    total_importance = sum(group_importances.values())
    normalized_group_importances = {group: importance / total_importance for group, importance in group_importances.items()}

    # Print and plot feature group importances
    print("Feature Group Importances:", normalized_group_importances)
    plt.figure()
    plt.bar(normalized_group_importances.keys(), normalized_group_importances.values())
    plt.ylabel('Importance')
    plt.title('Feature Group Importances')
    plt.show()

def smooth_data(data, alpha, history_length):
    num_features, num_frames = data.shape
    history = [deque(maxlen=history_length) for _ in range(num_features)]
    smoothed_data = np.zeros_like(data)
    
    for frame_idx in range(num_frames):
        for feature_idx in range(num_features):
            if not np.isnan(data[feature_idx, frame_idx]):
                history[feature_idx].append(data[feature_idx, frame_idx])
                avg = np.mean(history[feature_idx])
                smoothed_data[feature_idx, frame_idx] = alpha * data[feature_idx, frame_idx] + (1 - alpha) * avg
            elif history[feature_idx]:
                avg = np.mean(history[feature_idx])
                smoothed_data[feature_idx, frame_idx] = (1 - alpha) * avg
            else:
                smoothed_data[feature_idx, frame_idx] = np.nan
                
    return smoothed_data