# imports for our datetime, os permissions to create/save file, ultralytics for our AI models
# we also use a cv2 import for computer vision to properly handle videos
import cv2
import torch
import os
import numpy as np
from datetime import datetime
from ultralytics import YOLO, solutions

# This checks if the cpu that this code is running on has an NVIDIA chip and to run it on the NVIDIA, if not then the device we are running it on should just be the CPU of the computer
device = "cuda" if torch.cuda.is_available() else "cpu"

# Setting the model to YOLO11n, n stands for nano so we are running a full scale model on our computers
# YOLO11 was chosen because initially we had the thought of using faces to track 
model = YOLO("yolo11n.pt").to(device)

# If running on an NVIDIA chip, cut the model in half so it runs way faster and the video playback is better
if device == "cuda": model.half() 

# Setting variables for paths to where to find our video of contention
video_path = r"assets\test_video_footage2.mp4"

# --- OUTPUT ADDRESSES ---
# Address 1: Standard heatmap of the actual store video
store_output_folder = r"assets\processed_heatmaps"

# Address 2: Heatmap layered specifically on your floor plan
floorplan_output_folder = r"assets\floor_plans\heatmap_layered_floor_plan"

# HARDCODED: The raw floor plan we are using as a base
floor_plan_path = r"assets\floor_plans\sample_floor_plan.png"

# We open a python window that will playback the video with the overlay
win_name = "Phygital Person Scanner"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Running the video from the path
cap = cv2.VideoCapture(video_path)

# Prepare floor plan and internal heat accumulation layer
floor_plan = cv2.imread(floor_plan_path)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Ensure floor plan matches video dimensions for 1:1 pixel mapping
if floor_plan is not None:
    floor_plan = cv2.resize(floor_plan, (w, h))
else:
    print("Error: Could not find sample_floor_plan.png at the specified path.")

heat_layer = np.zeros((h, w), dtype=np.float32)

# we create a heatmap object from Ultralytics built in heatmap function using colormap and the model
heatmap_obj = solutions.Heatmap(
    model="yolo11n.pt",
    colormap=cv2.COLORMAP_JET
)

final_store_heatmap = None
final_floorplan_heatmap = None
frame_count = 0

# causes AI to run only on every 2nd frame
stride = 2 

# Following while loop seperates the playback and heatmap generation
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame_count += 1
    
    if frame_count % stride == 0:
        # TASK 1: Process standard store heatmap (the .plot_im version)
        heatmap_results = heatmap_obj.process(frame)
        final_store_heatmap = heatmap_results.plot_im # Store the image of the actual store with heat

        # TASK 2: Track people to draw heat on the invisible floor plan layer
        track_results = model.track(frame, persist=True, classes=[0], verbose=False, vid_stride=stride)
        
        if track_results[0].boxes.id is not None:
            for box in track_results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                # Draw "heat" on our invisible layer
                cv2.circle(heat_layer, (cx, cy), 20, 1, -1)

        # Generate the visual overlay for the Floor Plan export
        heat_norm = cv2.normalize(heat_layer, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        heatmap_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
        final_floorplan_heatmap = cv2.addWeighted(floor_plan, 0.6, heatmap_color, 0.4, 0)

        # Show the playback (initial style with boxes)
        annotated_frame = track_results[0].plot()
        cv2.imshow(win_name, cv2.resize(annotated_frame, (1280, 720)))

# End key if we need to stop the video playing
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- FINAL GENERATION OF BOTH IMAGES ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save Output 1: The Store Heatmap
if final_store_heatmap is not None:
    store_save_path = os.path.join(store_output_folder, f"store_heat_{timestamp}.png")
    cv2.imwrite(store_save_path, final_store_heatmap)
    print(f"Store Heatmap saved: {store_save_path}")

# Save Output 2: The Floor Plan Heatmap
if final_floorplan_heatmap is not None:
    fp_save_path = os.path.join(floorplan_output_folder, f"floorplan_heat_{timestamp}.png")
    cv2.imwrite(fp_save_path, final_floorplan_heatmap)
    print(f"Floor Plan Heatmap saved: {fp_save_path}")

# end of the video: closes the window and releases the footage
cap.release()
cv2.destroyAllWindows()