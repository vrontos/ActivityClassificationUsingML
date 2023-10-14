import os, subprocess, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from utils import get_video_frame_rate
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

# Split data for training and testing
X = final_feature_matrix.T
y = labels_for_frames
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
model_path = "trained_random_forest_2d_model.joblib"
dump(clf, model_path)