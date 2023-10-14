import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from utils import get_video_frame_rate
import matplotlib.pyplot as plt
from joblib import dump

# Load feature matrix
final_feature_matrix = np.load('final_feature_matrix.npy')

# Define paths
combined_video_path = os.path.join(os.getcwd(), "vids", "train", "train_all_vids.mp4")
labels_csv_path = os.path.join(os.getcwd(), "vids", "train", "labels.csv")

# Load labels from CSV
labels_df = pd.read_csv(labels_csv_path)

# Get video frame rate
frame_rate = get_video_frame_rate(combined_video_path)

# Associate each frame with its corresponding label
labels_for_frames = [
    labels_df[(labels_df['start_time'] <= idx / frame_rate) & 
              (labels_df['end_time'] > idx / frame_rate)]['label'].values[0]
    for idx in range(final_feature_matrix.shape[1])
]

# Constants
window_size = 100  # Define the appropriate window size for video frames

# Extracting statistical features
X, y = [], []
frames = final_feature_matrix.shape[1]
for i in range(0, frames - window_size + 1):
    window_data = final_feature_matrix[:, i:i+window_size].T  # Extracting the window of frames
    mean_vals = np.mean(window_data, axis=0)
    std_vals = np.std(window_data, axis=0)
    min_vals = np.min(window_data, axis=0)
    max_vals = np.max(window_data, axis=0)
    features = np.hstack([mean_vals, std_vals, min_vals, max_vals])
    X.append(features)
    y.append(labels_for_frames[i + window_size - 1])

X = np.array(X)
y = np.array(y)

# Model training
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, np.arange(len(X)), test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Visualizing the actual vs predicted classes
frame_indices = np.arange(window_size-1, len(labels_for_frames))
test_frame_indices = frame_indices[test_indices]

# Create a figure and axis
plt.figure(figsize=(15, 6))
plt.scatter(test_frame_indices, y_test, c='blue', marker='o', s=3, label='Actual')
plt.scatter(test_frame_indices, y_pred, c='red', marker='x', s=3, label='Predicted')

# Highlight differences/errors between actual and predicted
errors = y_test != y_pred
plt.scatter(test_frame_indices[errors], y_test[errors], c='yellow', marker='s', s=30, label='Errors', alpha=0.5)

plt.xlabel('Frame')
plt.ylabel('Class')
plt.title('Actual vs Predicted Classes')
plt.legend()
plt.grid(True)
plt.show()

# Save the trained model
model_path = "trained_random_forest_3d_model.joblib"
dump(clf, model_path)
