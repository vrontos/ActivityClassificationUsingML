import os
import subprocess
import csv
import tkinter as tk
from tkinter import filedialog

def get_video_duration(video_path):
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
    duration = float(subprocess.check_output(cmd).decode('utf-8').strip())
    return duration

def create_labels_csv(video_directory, output_csv_path):
    files = [f for f in os.listdir(video_directory) if f.endswith('.mp4')]
    current_time = 0.0

    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['start_time', 'end_time', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for file in files:
            video_duration = get_video_duration(os.path.join(video_directory, file))
            label = os.path.splitext(file)[0]
            writer.writerow({'start_time': current_time, 'end_time': current_time + video_duration, 'label': label})
            current_time += video_duration

def merge_videos(video_directory, output_filename):
    files = [f for f in os.listdir(video_directory) if f.endswith('.mp4')]
    file_list_path = os.path.join(video_directory, 'file-list.txt')
    
    with open(file_list_path, 'w') as f:
        for file in files:
            f.write(f"file '{os.path.abspath(os.path.join(video_directory, file))}'\n")

    cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', file_list_path, '-c', 'copy', os.path.join(video_directory, output_filename)]
    subprocess.run(cmd)
    os.remove(file_list_path)

# Open a file explorer window to let the user select the directory
root = tk.Tk()
root.withdraw()
video_directory = filedialog.askdirectory(title="Please select the video directory")

# Normalize the path to use backslashes
video_directory = video_directory.replace("/", "\\")

# Check if the directory is not empty and process the videos
if video_directory:
    create_labels_csv(video_directory, os.path.join(video_directory, "labels.csv"))
    merge_videos(video_directory, "train_all_vids.mp4")
