import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import select_raw_landmark_files

# 1. Select the raw landmarks file
file_path = select_raw_landmark_files()

# 2. Load the selected file
loaded_landmark_data = np.load(file_path[0])

def get_point(frame_num, landmark_index):
    """Get the 3D coordinates for the given landmark at the specified frame."""
    return loaded_landmark_data[frame_num, landmark_index*4:landmark_index*4+3]

def get_visibility(frame_num, landmark_index):
    """Get the visibility for the given landmark at the specified frame."""
    return loaded_landmark_data[frame_num, landmark_index*4+3]

def update_lines(num, lines):
    landmarks = [get_point(num, i) for i in range(13)]
    visibilities = [get_visibility(num, i) for i in range(13)]

    connections = [
        (0, 1),  # Nose to Left Shoulder
        (0, 2),  # Nose to Right Shoulder
        (1, 3),  # Left Shoulder to Left Elbow
        (2, 4),  # Right Shoulder to Right Elbow
        (3, 5),  # Left Elbow to Left Wrist
        (4, 6),  # Right Elbow to Right Wrist
        (1, 7),  # Left Shoulder to Left Hip
        (2, 8),  # Right Shoulder to Right Hip
        (7, 9),  # Left Hip to Left Knee
        (8, 10), # Right Hip to Right Knee
        (9, 11), # Left Knee to Left Ankle
        (10, 12) # Right Knee to Right Ankle
    ]

    for i, (start, end) in enumerate(connections):
        lines[i].set_data([landmarks[start][0], landmarks[end][0]], 
                          [landmarks[start][1], landmarks[end][1]])
        lines[i].set_3d_properties([landmarks[start][2], landmarks[end][2]])

        # Adjust line's opacity based on visibility
        if visibilities[start] == 0 or visibilities[end] == 0:
            lines[i].set_alpha(0.3)  # Reduced opacity if either landmark is not visible
        else:
            lines[i].set_alpha(1.0)  # Full opacity if both landmarks are visible

    return lines

# Attaching 3D axis to the figure
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# Create lines initially without data
lines = [ax.plot([], [], [], color='blue')[0] for _ in range(12)]

# Setting the axes properties
ax.set(xlim3d=(-1, 1), xlabel='X')
ax.set(ylim3d=(-1, 1), ylabel='Y')
ax.set(zlim3d=(-1, 1), zlabel='Z')

# Number of frames the animation should run for
num_frames = min(500, len(loaded_landmark_data))  # Using the first 150 frames or all available frames, whichever is smaller

# Creating the Animation object
ani = animation.FuncAnimation(
    fig, update_lines, num_frames, fargs=(lines,), interval=100)

# Before showing the plot, set the view angle
ax.view_init(elev=-90, azim=-90)
plt.show()
