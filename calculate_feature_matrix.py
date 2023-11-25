import numpy as np
import matplotlib.pyplot as plt
from utils import interpolate_nans, angle_between_three_points, check_feature_health, DESIRED_LANDMARK_NAMES, ANGLE_DEFINITIONS, smooth_data
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def calculate_feature_matrix(loaded_landmark_data, norm_std_flag=None):
    # Extract x, y, z coordinates and transpose
    feature_coordinates = np.array(loaded_landmark_data)[:, ::4]
    feature_coordinates = np.hstack((feature_coordinates, loaded_landmark_data[:, 1::4]))
    feature_coordinates = np.hstack((feature_coordinates, loaded_landmark_data[:, 2::4]))
    feature_coordinates = feature_coordinates.T

    # Interpolate NaNs for coordinates
    feature_coordinates_interpolated = interpolate_nans(feature_coordinates.copy())
    
    # Smooth landmark coordinates: alpha controls the smoothness, and history_length defines the window size.
    feature_coordinates_smoothed = smooth_data(feature_coordinates_interpolated, alpha=0.5, history_length=5) # alpha=1 -> no filtering, alpha=0.5 -> very light smooth, alpha=0.01 -> heavy smooth
    
    # Plot all rows of interpolated data
    for row in feature_coordinates_interpolated:
        plt.plot(row, color='blue', alpha=0.2, label='Interpolated')
    # Plot all rows of smoothed data
    for row in feature_coordinates_smoothed:
        plt.plot(row, color='orange', alpha=0.2, label='Smoothed')
    plt.title('Interpolated vs Smoothed Data (Joint Coordinates)')
    plt.xlabel('Time or Frame Index')
    plt.ylabel('Value')
    plt.show()

    feature_coordinates_interpolated = feature_coordinates_smoothed;

    # Calculate velocities
    feature_velocities = np.hstack((np.zeros((feature_coordinates_interpolated.shape[0], 1)), np.diff(feature_coordinates_interpolated, axis=1)))

    num_landmarks = feature_coordinates_interpolated.shape[0] // 3
    num_frames = feature_coordinates_interpolated.shape[1]
    reshaped_data = feature_coordinates_interpolated.reshape(num_landmarks, 3, num_frames)

    # Calculate pairwise distances
    feature_joint_distances = np.zeros((num_landmarks, num_landmarks, num_frames))
    for i in range(num_landmarks):
        for j in range(num_landmarks):
            if i != j:
                squared_diffs = (reshaped_data[i] - reshaped_data[j])**2
                distances = np.sqrt(np.sum(squared_diffs, axis=0))
                feature_joint_distances[i, j] = distances

    feature_joint_distances = feature_joint_distances.reshape(-1, feature_joint_distances.shape[2])

    # Extract visibility data
    feature_visibilities = loaded_landmark_data[:, 3::4].T
    feature_visibilities_interpolated = interpolate_nans(feature_visibilities.copy())
    
    # Calculate angles
    feature_angles = []
    for idx, (angle_name, landmarks) in enumerate(ANGLE_DEFINITIONS.items()):
        coords = [loaded_landmark_data[:, 4 * DESIRED_LANDMARK_NAMES.index(lm): 4 * DESIRED_LANDMARK_NAMES.index(lm) + 3] for lm in landmarks]
        processed_coords = [interpolate_nans(c) for c in coords]
        angles = [angle_between_three_points(*[pc[i] for pc in processed_coords]) for i in range(loaded_landmark_data.shape[0])]
        feature_angles.append(angles)
    feature_angles = np.array(feature_angles)

    # Stack all features
    final_feature_matrix = np.vstack((
        feature_coordinates_interpolated,
        feature_velocities,
        feature_joint_distances,
        feature_visibilities_interpolated,
        feature_angles
    ))
    
    # Check briefly the "health" of the features (e.g. if NaNs, Inf are found)
    check_feature_health(final_feature_matrix, num_landmarks)
    
    # Normalize or standardize if needed
    if norm_std_flag == "norm":
        final_feature_matrix = MinMaxScaler().fit_transform(final_feature_matrix.T).T
    elif norm_std_flag == "std":
        final_feature_matrix = StandardScaler().fit_transform(final_feature_matrix.T).T
    
    
    # Generate row descriptions
    descriptors = []
    
    # Positional data
    for i in range(num_landmarks):
        descriptors.extend([f"position x landmark {i}", 
                            f"position y landmark {i}", 
                            f"position z landmark {i}"])
    
    # Velocity data
    for i in range(num_landmarks):
        descriptors.extend([f"velocity x landmark {i}", 
                            f"velocity y landmark {i}", 
                            f"velocity z landmark {i}"])
    
    # Pairwise distances between landmarks
    for i in range(num_landmarks):
        for j in range(num_landmarks):
            if i != j:
                descriptors.append(f"distance landmark {i} to landmark {j}")
    
    # Visibility data
    for i in range(num_landmarks):
        descriptors.append(f"visibility landmark {i}")
        
    # Angle data
    for angle_name in ANGLE_DEFINITIONS.keys():
        descriptors.append(angle_name)

    return final_feature_matrix, num_landmarks, descriptors
