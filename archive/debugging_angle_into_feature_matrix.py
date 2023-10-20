import numpy as np
import matplotlib.pyplot as plt
from utils import select_raw_landmark_files, interpolate_nans, angle_between_three_points
from config import DESIRED_LANDMARK_NAMES, ANGLE_DEFINITIONS


def calculate_feature_matrix(loaded_landmark_data):
    # Extract x, y, z coordinates and transpose
    feature_coordinates = np.array(loaded_landmark_data)[:, ::4]
    feature_coordinates = np.hstack((feature_coordinates, loaded_landmark_data[:, 1::4]))
    feature_coordinates = np.hstack((feature_coordinates, loaded_landmark_data[:, 2::4]))
    feature_coordinates = feature_coordinates.T

    # Interpolate NaNs for coordinates
    feature_coordinates_interpolated = interpolate_nans(feature_coordinates.copy())

    # Calculate angles
    angle_features = []
    for idx, (angle_name, landmarks) in enumerate(ANGLE_DEFINITIONS.items()):
        coords = [loaded_landmark_data[:, 4 * DESIRED_LANDMARK_NAMES.index(lm): 4 * DESIRED_LANDMARK_NAMES.index(lm) + 3] for lm in landmarks]
        processed_coords = [interpolate_nans(c) for c in coords]
        angles = [angle_between_three_points(*[pc[i] for pc in processed_coords]) for i in range(loaded_landmark_data.shape[0])]
        angle_features.append(angles)
    angle_features = np.array(angle_features)

    # Stack all features
    final_feature_matrix = np.vstack((
        feature_coordinates_interpolated,
        angle_features
    ))
    
    num_landmarks = feature_coordinates_interpolated.shape[0] // 3

    # Generate row descriptions
    descriptors = []
    
    # Positional data
    for i in range(num_landmarks):
        descriptors.extend([f"position x landmark {i}", 
                            f"position y landmark {i}", 
                            f"position z landmark {i}"])
    
    # Angle data
    for angle_name in ANGLE_DEFINITIONS.keys():
        descriptors.append(angle_name)
        
    return final_feature_matrix, num_landmarks, descriptors

def plot_features(final_feature_matrix, num_landmarks, descriptors):
    num_angles = len(ANGLE_DEFINITIONS)
    
    # Plot the interpolated coordinates
    plt.figure(figsize=(12, 8))
    for i in range(num_landmarks):
        for j in range(3):  # x, y, z coordinates
            plt.subplot(num_landmarks, 3, i*3 + j + 1)
            plt.plot(final_feature_matrix[i*3 + j])
            plt.title(descriptors[i*3 + j])
    plt.tight_layout()
    plt.show()
    
    # Plot the angles
    plt.figure(figsize=(10, 8))
    for idx, angle_name in enumerate(ANGLE_DEFINITIONS.keys()):
        plt.subplot(num_angles, 1, idx + 1)
        plt.plot(final_feature_matrix[num_landmarks*3 + idx])
        plt.title(descriptors[num_landmarks*3 + idx])
    plt.tight_layout()
    plt.show()

file_path = select_raw_landmark_files()
loaded_landmark_data = np.load(file_path[0])
final_feature_matrix, num_landmarks, descriptors = calculate_feature_matrix(loaded_landmark_data)
plot_features(final_feature_matrix, num_landmarks, descriptors)