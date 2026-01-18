# imports for our datetime, os permissions to create/save file, ultralytics for our AI models
# we also use a cv2 import for computer vision to properly handle videos
import cv2
import torch
import os
import json
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
video_path = r"assets/test_video_footage.mp4"

# --- OUTPUT ADDRESSES ---
# Address 1: Standard heatmap of the actual store video
store_output_folder = r"assets/processed_heatmaps"

# Address 2: Heatmap layered specifically on your floor plan
floorplan_output_folder = r"assets/floor_plans/heatmap_layered_floor_plan"

# HARDCODED: The raw floor plan we are using as a base
floor_plan_path = r"assets/floor_plans/sample_floor_plan.png"

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
        # TASK 1: Process heatmap with Ultralytics on the floor plan background
        # First, run detection on the actual video frame to get tracking data
        heatmap_results = heatmap_obj.process(frame)
        
        # Extract the heatmap overlay from the result
        # We'll recreate this on the floor plan instead of the video frame
        heatmap_only = heatmap_results.plot_im
        
        # Create a copy of the floor plan for the store heatmap
        floor_plan_copy = floor_plan.copy()
        
        # Overlay the Ultralytics heatmap on the floor plan
        # The heatmap_only contains the visualization, we blend it with floor plan
        final_store_heatmap = cv2.addWeighted(floor_plan_copy, 0.5, heatmap_only, 0.5, 0)

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

# --- EXTRACT HIGH TRAFFIC AREAS FROM STORE HEATMAP ---
if final_store_heatmap is not None:
    # Convert the heatmap to HSV to isolate red/yellow hot areas
    # The JET colormap: blue (cold) -> cyan -> green -> yellow -> red (hot)
    hsv_heatmap = cv2.cvtColor(final_store_heatmap, cv2.COLOR_BGR2HSV)
    
    # Create mask for red areas (high traffic)
    # Red wraps around in HSV: 0-10 and 170-180
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Yellow/orange areas (medium-high traffic)
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    
    # Combine masks
    mask_red1 = cv2.inRange(hsv_heatmap, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_heatmap, lower_red2, upper_red2)
    mask_yellow = cv2.inRange(hsv_heatmap, lower_yellow, upper_yellow)
    
    # Combine all high-traffic masks (red + yellow)
    binary_mask = cv2.bitwise_or(mask_red1, mask_red2)
    binary_mask = cv2.bitwise_or(binary_mask, mask_yellow)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours of high traffic regions
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract data for each high traffic region
    high_traffic_areas = []
    for idx, contour in enumerate(contours):
        # Filter out very small regions (noise)
        area = cv2.contourArea(contour)
        if area < 500:  # Minimum area threshold in pixels (increased from 100)
            continue
        
        # Get bounding box
        x, y, w_box, h_box = cv2.boundingRect(contour)
        
        # Calculate center point
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w_box // 2, y + h_box // 2
        
        # Calculate intensity metrics from the value channel (brightness)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        # Get average color in HSV to determine heat level
        region_hsv = cv2.mean(hsv_heatmap, mask=mask)
        heat_intensity = region_hsv[2]  # V channel (brightness/value)
        
        # Determine heat category based on hue
        avg_hue = region_hsv[0]
        if avg_hue <= 10 or avg_hue >= 170:
            heat_category = "very_high"  # Red
        elif 15 <= avg_hue <= 35:
            heat_category = "high"  # Yellow/Orange
        else:
            heat_category = "medium"  # Other
        
        # Store region data
        region_data = {
            "id": idx + 1,
            "center": {"x": int(cx), "y": int(cy)},
            "bounding_box": {
                "x": int(x),
                "y": int(y),
                "width": int(w_box),
                "height": int(h_box)
            },
            "area_pixels": int(area),
            "heat_intensity": float(heat_intensity),
            "heat_category": heat_category,
            "contour_points": contour.reshape(-1, 2).tolist()[:50]  # Limit points for JSON size
        }
        high_traffic_areas.append(region_data)
    
    # Sort by heat intensity (highest first)
    high_traffic_areas.sort(key=lambda x: x["heat_intensity"], reverse=True)
    
    # Create JSON output
    json_output = {
        "metadata": {
            "timestamp": timestamp,
            "video_dimensions": {"width": w, "height": h},
            "total_frames_processed": frame_count,
            "detection_method": "hsv_colormap_analysis",
            "total_high_traffic_regions": len(high_traffic_areas)
        },
        "high_traffic_areas": high_traffic_areas
    }
    
    # Save to JSON file
    json_output_path = os.path.join(floorplan_output_folder, f"high_traffic_data_{timestamp}.json")
    with open(json_output_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"High Traffic Analysis Complete!")
    print(f"{'='*60}")
    print(f"Found {len(high_traffic_areas)} high-traffic regions")
    print(f"Data saved to: {json_output_path}")
    if high_traffic_areas:
        print(f"\nTop 5 highest traffic areas:")
        for i, area in enumerate(high_traffic_areas[:5], 1):
            print(f"  {i}. Center: ({area['center']['x']}, {area['center']['y']}), "
                  f"Area: {area['area_pixels']}px, Heat: {area['heat_category']}")
    print(f"{'='*60}\n")
else:
    print("Warning: No store heatmap available for traffic analysis")

# end of the video: closes the window and releases the footage
cap.release()
cv2.destroyAllWindows()