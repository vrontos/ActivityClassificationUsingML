import numpy as np
import matplotlib.pyplot as plt
from calculate_feature_matrix import calculate_feature_matrix
from utils import select_raw_landmark_files
import mplcursors
from config import DESIRED_LANDMARK_NAMES


def plot_selected_feature(feature_matrix, num_landmarks, descriptors, ax, selected_feature, cursor=None):
    ax.clear()
    if cursor:  # If there's an existing cursor, remove it
        cursor.remove()

    labels = []
    if selected_feature == "Positions":
        data = feature_matrix[:num_landmarks*3]
        title = 'Landmark Positions Over Time'
        labels = [f"{name} Position {d}" for name in DESIRED_LANDMARK_NAMES for d in ['x', 'y', 'z']]
    elif selected_feature == "Velocities":
        data = feature_matrix[num_landmarks*3:num_landmarks*6]
        title = 'Landmark Velocities Over Time'
        labels = [f"{name} Velocity {d}" for name in DESIRED_LANDMARK_NAMES for d in ['x', 'y', 'z']]
    elif selected_feature == "Pairwise Distances":
        data = feature_matrix[num_landmarks*6:num_landmarks*6 + num_landmarks**2]
        title = 'Pairwise Distances Over Time'
        # The pairwise distances label generation is a bit more complex since it involves combinations of landmarks
        labels = [f"Distance {name1} to {name2}" for index1, name1 in enumerate(DESIRED_LANDMARK_NAMES) for index2, name2 in enumerate(DESIRED_LANDMARK_NAMES) if index2 >= index1]
    elif selected_feature == "Visibilities":
        data = feature_matrix[-num_landmarks:]
        title = 'Visibility for Landmarks Over Time'
        labels = [f"{name} Visibility" for name in DESIRED_LANDMARK_NAMES]

    lines = ax.plot(data.T)
    ax.set_title(title)
    ax.set_xlabel('Frame Number')
    plt.tight_layout()

    def hover_handler(sel):
        line = sel.artist
        index = lines.index(line)
        sel.annotation.set_text(labels[index])

    cursor = mplcursors.cursor(hover=True)
    cursor.connect("add", hover_handler)

    plt.draw()
    return cursor  # Return the new cursor



def interactive_plot(feature_matrix, num_landmarks, descriptors):
    fig, ax = plt.subplots(figsize=(8, 8))
    feature_options = ["Positions", "Velocities", "Pairwise Distances", "Visibilities"]
    current_index = 0
    cursor = None  # Initially no cursor

    cursor = plot_selected_feature(feature_matrix, num_landmarks, descriptors, ax, feature_options[current_index], cursor)
    
    def onclick(event):
        nonlocal current_index, cursor
        if event.inaxes is None: return
        current_index = (current_index + 1) % 4
        cursor = plot_selected_feature(feature_matrix, num_landmarks, descriptors, ax, feature_options[current_index], cursor)
        
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()


file_path = select_raw_landmark_files()
loaded_landmark_data = np.load(file_path[0])
feature_matrix, num_landmarks, descriptors = calculate_feature_matrix(loaded_landmark_data)
interactive_plot(feature_matrix, num_landmarks, descriptors)
