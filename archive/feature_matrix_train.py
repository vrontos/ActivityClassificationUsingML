import numpy as np
import matplotlib.pyplot as plt
from calculate_feature_matrix import calculate_feature_matrix
from utils import select_raw_landmark_files

file_path = select_raw_landmark_files()
loaded_landmark_data = np.load(file_path[0])

final_feature_matrix, num_landmarks, descriptors = calculate_feature_matrix(loaded_landmark_data)

# Example: Plotting the X coordinate of the first landmark over time
plt.figure()  # Create a new figure
plt.plot(final_feature_matrix[0])
plt.title('X coordinate of 1st Landmark over Time')
plt.xlabel('Frame')
plt.ylabel('X Coordinate')
plt.show()

plt.figure()  # Create a new figure
plt.hist(final_feature_matrix.ravel(), bins=50)
plt.title('Feature Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

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

# Save the final feature matrix for future use
np.save('final_feature_matrix.npy', final_feature_matrix)
