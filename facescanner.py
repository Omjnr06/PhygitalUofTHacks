# imports for our datetime, os permissions to create/save file, ultralytics for our AI models
# we also use a cv2 import for computer vision to properly handle videos
import cv2
import torch
import os
from datetime import datetime
from ultralytics import YOLO, solutions

# This checks if the cpu that this code is running on has an NVIDIA chip and to run it on the NVIDIA, if not then the device we are running it on should just be the CPU of the computer
device = "cuda" if torch.cuda.is_available() else "cpu"

# Setting the model to YOLO11n, n stands for nano so we are running a full scale model on our computers
# YOLO11 was chosen because initially we had the thought of using faces to track 
model = YOLO("yolo11n.pt").to(device)

# If running on an NVIDIA chip, cut the model in half so it runs way faster and the video playback is better
if device == "cuda": model.half() 

# Setting variables for paths to where to find our video of contention and where to save heatmap images
video_path = r"assets\test_video_footage.mp4"
output_folder = r"assets\processed_heatmaps"

# We open a python window that will playback the video with the overlay
win_name = "Phygital Person Scanner"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Running the video from the path
cap = cv2.VideoCapture(video_path)

# we create a heatmap object from Ultralytics built in heatmap function using colormap and the model
heatmap_obj = solutions.Heatmap(
    model="yolo11n.pt",
    colormap=cv2.COLORMAP_JET
)


final_heatmap_img = None
frame_count = 0

# causes AI to run only on every 2nd frame
stride = 2 

# Following while loop seperates the playback and heatmap generation (had an issue of the heatmap generating on the playback)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame_count += 1
    

    if frame_count % stride == 0:
        heatmap_results = heatmap_obj.process(frame)
        final_heatmap_img = heatmap_results.plot_im

        track_results = model.track(frame, persist=True, classes=[0], verbose=False, vid_stride=stride)
        annotated_frame = track_results[0].plot()
        
        cv2.imshow(win_name, cv2.resize(annotated_frame, (1280, 720)))

# End key if we need to stop the video playing
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Generation of the heat map image
if final_heatmap_img is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_folder, f"heatmap_{timestamp}.png")
    cv2.imwrite(save_path, final_heatmap_img)
    print(f"Final Heatmap saved to: {save_path}")

# end of the video: closes the window and releases the footage
cap.release()
cv2.destroyAllWindows()