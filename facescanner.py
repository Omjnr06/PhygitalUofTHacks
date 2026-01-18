import cv2
import torch
import os
from datetime import datetime
from ultralytics import YOLO, solutions

# 1. Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11n.pt").to(device)
video_path = r"assets\test_video_footage.mp4"
output_folder = r"assets\processed_heatmaps"

cap = cv2.VideoCapture(video_path)

# 2. Heatmap Init
heatmap_obj = solutions.Heatmap(
    model="yolo11n.pt",
    colormap=cv2.COLORMAP_JET
)

final_heatmap_img = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # --- STEP 1: Standard Video Playback (Boxes) ---
    # We run tracking and use .plot() to get the regular "initial style" boxes
    track_results = model.track(frame, persist=True, classes=[0], verbose=False)
    annotated_frame = track_results[0].plot() # This has the boxes/IDs

    # --- STEP 2: Background Heatmap (Invisible) ---
    # We pass the frame to the heatmap tool but we DON'T show this version yet
    heatmap_results = heatmap_obj.process(frame)
    final_heatmap_img = heatmap_results.plot_im # Store the heatmap state

    # --- STEP 3: Display the "Normal" Style ---
    cv2.imshow("Tracking View (Initial Style)", cv2.resize(annotated_frame, (1280, 720)))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- STEP 4: Save the Final Heatmap Photo ---
if final_heatmap_img is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_folder, f"heatmap_{timestamp}.png")
    cv2.imwrite(save_path, final_heatmap_img)
    print(f"Heatmap photo saved to: {save_path}")

cap.release()
cv2.destroyAllWindows()