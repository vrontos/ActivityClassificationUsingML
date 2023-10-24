import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
from calculate_feature_matrix import calculate_feature_matrix
from utils import select_raw_landmark_files, get_video_frame_rate
import matplotlib.pyplot as plt
from utils import WINDOW_SIZE

# Select the raw landmarks file
file_path = select_raw_landmark_files("raw_worldlandmarks")[0]

# Load landmarks data
loaded_landmark_data = np.load(file_path)

# Calculate the final feature matrix
final_feature_matrix, num_landmarks, descriptors = calculate_feature_matrix(loaded_landmark_data)

# Construct paths based on the selected landmarks file
base_dir = os.getcwd()
landmarks_rel_path = os.path.relpath(file_path, os.path.join(base_dir, "raw_worldlandmarks"))
video_rel_path = landmarks_rel_path.replace("_raw_worldlandmarks.npy", ".mp4")
labels_rel_path = os.path.join(os.path.dirname(video_rel_path), "labels.csv")

video_path = os.path.join(base_dir, "vids", video_rel_path)
labels_csv_path = os.path.join(base_dir, "vids", labels_rel_path)

# Check if the video file and labels file exist
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")
if not os.path.exists(labels_csv_path):
    raise FileNotFoundError(f"Labels CSV file not found: {labels_csv_path}")

# Load labels from CSV
labels_df = pd.read_csv(labels_csv_path)

# Get video frame rate
frame_rate = get_video_frame_rate(video_path)

# Associate each frame with its corresponding label
labels_for_frames = [
    labels_df[(labels_df['start_time'] <= idx / frame_rate) & 
              (labels_df['end_time'] > idx / frame_rate)]['label'].values[0]
    for idx in range(final_feature_matrix.shape[1])
]

# Extracting statistical features
X, y = [], []
frames = final_feature_matrix.shape[1]
for i in range(0, frames - WINDOW_SIZE + 1):
    window_data = final_feature_matrix[:, i:i+WINDOW_SIZE].T  # Extracting the window of frames
    mean_vals = np.mean(window_data, axis=0)
    std_vals = np.std(window_data, axis=0)
    min_vals = np.min(window_data, axis=0)
    max_vals = np.max(window_data, axis=0)
    #epsilon = 1e-8
    #skewness_vals = np.divide((window_data - mean_vals) ** 3, (std_vals + epsilon) ** 3).mean(axis=0)
    
    features = np.hstack([mean_vals, std_vals, min_vals, max_vals])
    #features = np.hstack([mean_vals, std_vals, min_vals, max_vals, skewness_vals])

    X.append(features)
    y.append(labels_for_frames[i + WINDOW_SIZE - 1])

X = np.array(X)
y = np.array(y)

# Model training
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, np.arange(len(X)), test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Visualizing the actual vs predicted classes
frame_indices = np.arange(WINDOW_SIZE-1, len(labels_for_frames))
test_frame_indices = frame_indices[test_indices]

# Highlight differences/errors between actual and predicted
errors = y_test != y_pred

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
model_path = "trained_random_forest_3d_model.joblib"
dump(clf, model_path)

# # Get feature importances
importances = clf.feature_importances_

# Assuming 'descriptors' is a list of feature names
feature_groups = {
    'position': [],
    'velocity': [],
    'distances': [],
    'visibility': [],
    'angles': [],
}

# Assign each descriptor to its feature group
for descriptor in descriptors:
    for group in feature_groups:
        if group in descriptor:
            feature_groups[group].append(descriptor)

# Calculate importance for each feature group
group_importances = {}
for group, features in feature_groups.items():
    indices = [descriptors.index(feature) for feature in features]
    group_importance = np.sum([importances[index] for index in indices])
    group_importances[group] = group_importance

# Normalize importances so they sum to 1
total_importance = sum(group_importances.values())
normalized_importances = {group: importance / total_importance for group, importance in group_importances.items()}

print(normalized_importances)

plt.bar(normalized_importances.keys(), normalized_importances.values())
plt.ylabel('Importance')
plt.title('Feature Group Importances')
plt.show()