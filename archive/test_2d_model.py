import numpy as np
import matplotlib.pyplot as plt
from joblib import load

# Load the trained model
clf = load("trained_random_forest_2d_model.joblib")

# Load the processed feature matrix for the testing video
final_feature_matrix_test = np.load('final_feature_matrix.npy')

# Predict labels for the test video
y_pred_test = clf.predict(final_feature_matrix_test.T)

# Plot the predicted labels for the test video
plt.figure(figsize=(10, 5))
plt.plot(y_pred_test)
plt.title('Predicted Labels for the Test Video')
plt.xlabel('Frame Index')
plt.ylabel('Predicted Label')
plt.tight_layout()
plt.show()
