import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from utils import get_video_frame_rate, select_raw_landmark_files, WINDOW_SIZE
from sklearn.metrics import accuracy_score
from calculate_feature_matrix import calculate_feature_matrix

# Load the trained model
clf = load("trained_random_forest_3d_model.joblib")

# Select the raw landmarks file
file_path = select_raw_landmark_files("raw_worldlandmarks/test")
loaded_landmark_data = np.load(file_path[0])

# Calculate the final feature matrix
final_feature_matrix, num_landmarks, _ = calculate_feature_matrix(loaded_landmark_data,None)

# Constructing labels CSV and video path based on the selected landmarks file
base_name = os.path.basename(file_path[0]).replace("_raw_worldlandmarks.npy", "")
labels_csv_name = f"labels_{base_name}.csv"
video_name = f"{base_name}.mp4"

# Assuming the script is executed from the ActivityClassificationUsingML directory
labels_csv_path = os.path.join("vids", "test", labels_csv_name)
video_path = os.path.join("vids", "test", video_name)

# Check if the labels CSV file and video file exist
if not os.path.exists(labels_csv_path):
    raise FileNotFoundError(f"Labels CSV file not found: {labels_csv_path}")
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")

# Load the correct labels
labels_df = pd.read_csv(labels_csv_path)

# Get video frame rate for test video
frame_rate = get_video_frame_rate(video_path)

# Convert the time-based labels to frame-based
labels_for_frames = [
    labels_df[(labels_df['start_time'] <= idx / frame_rate) & 
              (labels_df['end_time'] > idx / frame_rate)]['label'].values[0]
    for idx in range(final_feature_matrix.shape[1] - WINDOW_SIZE + 1)
]

# Extracting statistical features for test data
X_test = []
frames_test = final_feature_matrix.shape[1]
for i in range(0, frames_test - WINDOW_SIZE + 1):
    window_data_test = final_feature_matrix[:, i:i+WINDOW_SIZE].T
    mean_vals_test = np.mean(window_data_test, axis=0)
    std_vals_test = np.std(window_data_test, axis=0)
    min_vals_test = np.min(window_data_test, axis=0)
    max_vals_test = np.max(window_data_test, axis=0)
    #epsilon = 1e-8
    #skewness_vals_test = np.divide((window_data_test - mean_vals_test) ** 3, (std_vals_test + epsilon) ** 3).mean(axis=0)
    # TODO: kurtosis
    # TODO: fft
    features_test = np.hstack([mean_vals_test, std_vals_test, min_vals_test, max_vals_test])
    #features_test = np.hstack([mean_vals_test, std_vals_test, min_vals_test, max_vals_test, skewness_vals_test])

    X_test.append(features_test)
X_test = np.array(X_test)

# Predict using your trained model
y_pred_test = clf.predict(X_test)

# Plot the true and predicted labels for the test video
plt.figure(figsize=(10, 5))
plt.plot(labels_for_frames, label='True Labels', c='blue')
plt.plot(y_pred_test, label='Predicted Labels', c='red', linestyle='dashed')
plt.title('True vs Predicted Labels for the Test Video')
plt.xlabel('Frame Index')
plt.ylabel('Label')
plt.legend()
plt.tight_layout()
plt.show()

# Calculate and print accuracy
accuracy = accuracy_score(labels_for_frames, y_pred_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
