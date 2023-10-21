import numpy as np
import matplotlib.pyplot as plt
from calculate_feature_matrix import calculate_feature_matrix
from utils import select_raw_landmark_files

def plot_all_features(feature_matrix, num_landmarks, descriptors):
    # Setting up the figure and subplots
    fig, axs = plt.subplots(4, 1, figsize=(8, 8))
    
    # Plotting positions
    position_data = feature_matrix[:num_landmarks*3]
    axs[0].plot(position_data.T)
    axs[0].set_title('Landmark Positions Over Time')
    axs[0].set_xlabel('Frame Number')
    axs[0].set_ylabel('Position')
    
    # Plotting velocities
    velocity_data = feature_matrix[num_landmarks*3:num_landmarks*6]
    axs[1].plot(velocity_data.T)
    axs[1].set_title('Landmark Velocities Over Time')
    axs[1].set_xlabel('Frame Number')
    axs[1].set_ylabel('Velocity')
    
    # Plotting pairwise distances
    pairwise_data = feature_matrix[num_landmarks*6:num_landmarks*6 + num_landmarks**2]
    axs[2].plot(pairwise_data.T)
    axs[2].set_title('Pairwise Distances Over Time')
    axs[2].set_xlabel('Frame Number')
    axs[2].set_ylabel('Distance')
    
    # Plotting visibilities
    visibility_data = feature_matrix[-num_landmarks:]
    axs[3].plot(visibility_data.T)
    axs[3].set_title('Visibility for Landmarks Over Time')
    axs[3].set_xlabel('Frame Number')
    axs[3].set_ylabel('Visibility')

    # Adjusting the layout and displaying the plots
    plt.tight_layout()
    plt.show()


file_path = select_raw_landmark_files()
loaded_landmark_data = np.load(file_path[0])
feature_matrix, num_landmarks, descriptors = calculate_feature_matrix(loaded_landmark_data)

plot_all_features(feature_matrix, num_landmarks, descriptors)

