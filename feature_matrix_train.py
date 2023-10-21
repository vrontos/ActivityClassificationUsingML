import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import ttk
from calculate_feature_matrix import calculate_feature_matrix
from utils import plot_landmark_data

# Create a GUI window to select raw_landmarks files
root = tk.Tk()
root.title("Select Raw Landmarks")

# List all available raw_landmarks files
raw_landmark_dir = os.path.join(os.getcwd(), "raw_landmarks")
raw_landmark_files = [f for f in os.listdir(raw_landmark_dir) 
                      if f.endswith("_raw_landmarks.npy") 
                      and "train" in f 
                      and "_all" in f]  # Condition added here

var_values = [tk.IntVar() for _ in raw_landmark_files]
selected_files = []

def on_select():
    selected_files.extend([raw_landmark_files[i] for i in range(len(raw_landmark_files)) if var_values[i].get() == 1])
    root.quit()

# Create checkboxes for each raw_landmarks file
for i, file in enumerate(raw_landmark_files):
    ttk.Checkbutton(root, text=file, variable=var_values[i]).pack(anchor=tk.W, padx=10, pady=5)

# Add Select button
ttk.Button(root, text="Select", command=on_select).pack(pady=20)

root.mainloop()
root.destroy()

loaded_landmark_data_path = os.path.join(raw_landmark_dir, selected_files[0])
loaded_landmark_data = np.load(loaded_landmark_data_path)

final_feature_matrix, num_landmarks, _ = calculate_feature_matrix(loaded_landmark_data)

# Example: Plotting the X coordinate of the first landmark over time
plt.plot(final_feature_matrix[0])
plt.title('X coordinate of 1st Landmark over Time')
plt.xlabel('Frame')
plt.ylabel('X Coordinate')
plt.show()

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
