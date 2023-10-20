import numpy as np
import matplotlib.pyplot as plt
from utils import select_raw_landmark_files, interpolate_nans

#TODO: load them from config
DESIRED_LANDMARKS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
LANDMARK_NAMES = [
    'nose', 'left eye (inner)', 'left eye', 'left eye (outer)', 'right eye (inner)', 'right eye',
    'right eye (outer)', 'left ear', 'right ear', 'mouth (left)', 'mouth (right)', 'left shoulder',
    'right shoulder', 'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left pinky',
    'right pinky', 'left index', 'right index', 'left thumb', 'right thumb', 'left hip', 'right hip',
    'left knee', 'right knee', 'left ankle', 'right ankle', 'left heel', 'right heel', 'left foot index',
    'right foot index'
]
DESIRED_LANDMARK_NAMES = [LANDMARK_NAMES[i] for i in DESIRED_LANDMARKS]

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

def process_landmark_coordinates(raw_coords):
    return interpolate_nans(raw_coords)

file_path = select_raw_landmark_files()
loaded_landmark_data = np.load(file_path[0])

angle_definitions = {
    'angle_right_elbow': ['right shoulder', 'right elbow', 'right wrist'],
    'angle_right_shoulder': ['right elbow', 'right shoulder', 'right hip'],
    'angle_right_hip': ['right shoulder', 'right hip', 'right knee'],
    'angle_right_knee': ['right hip', 'right knee', 'right ankle'],
    'angle_left_elbow': ['left shoulder', 'left elbow', 'left wrist'],
    'angle_left_shoulder': ['left elbow', 'left shoulder', 'left hip'],
    'angle_left_hip': ['left shoulder', 'left hip', 'left knee'],
    'angle_left_knee': ['left hip', 'left knee', 'left ankle']
}

plt.figure(figsize=(10, 8))  # adjusting figure size

for idx, (angle_name, landmarks) in enumerate(angle_definitions.items()):
    coords = [loaded_landmark_data[:, 4 * DESIRED_LANDMARK_NAMES.index(lm): 4 * DESIRED_LANDMARK_NAMES.index(lm) + 3] for lm in landmarks]
    processed_coords = [process_landmark_coordinates(c) for c in coords]
    angles = [angle_between_three_points(*[pc[i] for pc in processed_coords]) for i in range(loaded_landmark_data.shape[0])]
    
    plt.subplot(len(angle_definitions), 1, idx + 1)
    plt.plot(angles, label=angle_name)
    plt.legend(loc='upper right')
    plt.xlabel("Frame Number")
    plt.ylabel("Angle (°)")

plt.tight_layout()
plt.show()
