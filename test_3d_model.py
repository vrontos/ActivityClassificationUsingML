import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from utils import get_video_frame_rate
from sklearn.metrics import accuracy_score


# Load the trained model
clf = load("trained_random_forest_3d_model.joblib")

# Load the processed feature matrix for the testing video
final_feature_matrix_test = np.load('final_feature_matrix.npy')

# Constants
window_size = 100

# Extracting statistical features for test data
X_test = []
frames_test = final_feature_matrix_test.shape[1]
for i in range(0, frames_test - window_size + 1):
    window_data_test = final_feature_matrix_test[:, i:i+window_size].T
    mean_vals_test = np.mean(window_data_test, axis=0)
    std_vals_test = np.std(window_data_test, axis=0)
    min_vals_test = np.min(window_data_test, axis=0)
    max_vals_test = np.max(window_data_test, axis=0)
    features_test = np.hstack([mean_vals_test, std_vals_test, min_vals_test, max_vals_test])
    X_test.append(features_test)

X_test = np.array(X_test)

# Predict using your trained model
y_pred_test = clf.predict(X_test)

# Load the correct labels
labels_csv_path_test = os.path.join(os.getcwd(), "vids", "test", "labels_test_vid2.csv")
labels_df_test = pd.read_csv(labels_csv_path_test)

# Get video frame rate for test video
combined_video_path_test = os.path.join(os.getcwd(), "vids", "test", "test_vid2.mp4")
frame_rate_test = get_video_frame_rate(combined_video_path_test)

# Convert the time-based labels to frame-based
labels_for_frames_test = [
    labels_df_test[(labels_df_test['start_time'] <= idx / frame_rate_test) & 
                   (labels_df_test['end_time'] > idx / frame_rate_test)]['label'].values[0]
    for idx in range(frames_test - window_size + 1)
]

# Plot the true and predicted labels for the test video
plt.figure(figsize=(10, 5))
plt.plot(labels_for_frames_test, label='True Labels', c='blue')
plt.plot(y_pred_test, label='Predicted Labels', c='red', linestyle='dashed')
plt.title('True vs Predicted Labels for the Test Video')
plt.xlabel('Frame Index')
plt.ylabel('Label')
plt.legend()
plt.tight_layout()
plt.show()

# Calculate and print accuracy
accuracy = accuracy_score(labels_for_frames_test, y_pred_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
