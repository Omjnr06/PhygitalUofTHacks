import cv2
import torch
import os
import json
import numpy as np
from datetime import datetime
from ultralytics import YOLO, solutions

# --- SETUP DEVICES & MODELS ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11n.pt").to(device)
if device == "cuda": model.half() 

# --- PATHS ---
# Make sure these match your actual filenames!
video_path = r"assets\test_video_footage.mp4"
floor_plan_path = r"assets\floor_plans\sample_floor_plan.png"
zones_file = r"assets\zones.json"

# Outputs
analytics_output = r"assets\store_analytics.json"
heatmap_image_output = r"assets\floor_plans\final_heatmap.png"

# --- LOAD ZONES ---
if os.path.exists(zones_file):
    with open(zones_file, 'r') as f:
        aisle_zones = json.load(f)
    print(f"✅ Loaded {len(aisle_zones)} Aisles from zones.json")
else:
    print("❌ Warning: zones.json not found. You must run define_zones.py first!")
    aisle_zones = []

# --- VIDEO & FLOOR PLAN SETUP ---
cap = cv2.VideoCapture(video_path)
video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

floor_plan = cv2.imread(floor_plan_path)
if floor_plan is None:
    print(f"❌ Error: Could not find floor plan at {floor_plan_path}")
    exit()

# Get floor plan dimensions
h_fp, w_fp = floor_plan.shape[:2]

# Create a blank layer to draw the heat on (Float32 allows smooth accumulation)
heat_layer = np.zeros((h_fp, w_fp), dtype=np.float32)

# --- TRACKING LOOP ---
stride = 2 
frame_count = 0

print(f"Starting Scanner... (Video: {video_w}x{video_h} -> Plan: {w_fp}x{h_fp})")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame_count += 1
    if frame_count % stride == 0:
        # Run YOLO Tracking
        track_results = model.track(frame, persist=True, classes=[0], verbose=False, vid_stride=stride)
        
        if track_results[0].boxes.id is not None:
            boxes = track_results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                # 1. Find the person's feet in the video (x, y)
                vid_x = (box[0] + box[2]) / 2
                vid_y = box[3] 
                
                # 2. SIMPLE SCALING: Stretch video coordinate to floor plan size
                # New_X = Old_X * (Target_Width / Source_Width)
                fp_x = int(vid_x * (w_fp / video_w))
                fp_y = int(vid_y * (h_fp / video_h))

                # 3. Draw heat on the invisible floor plan layer
                # Clamp values to make sure we don't draw off the edge
                fp_x = max(0, min(fp_x, w_fp - 1))
                fp_y = max(0, min(fp_y, h_fp - 1))
                
                cv2.circle(heat_layer, (fp_x, fp_y), 15, 1, -1)

        # Show the video feed (Optional)
        annotated_frame = track_results[0].plot()
        cv2.imshow("Phygital Scanner", cv2.resize(annotated_frame, (1080, 720)))

    # Press 'q' to stop early and save results
    if cv2.waitKey(1) & 0xFF == ord("q"): break

# --- ANALYTICS GENERATION (NEW DENSITY LOGIC) ---
print("\nProcessing Analytics...")

# 1. Normalize heat layer to standard 0-255 image range
heat_norm = cv2.normalize(heat_layer, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# 2. Save the Visual Heatmap Image
heatmap_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
final_overlay = cv2.addWeighted(floor_plan, 0.6, heatmap_color, 0.4, 0)
cv2.imwrite(heatmap_image_output, final_overlay)
print(f"✅ Visual Heatmap saved to: {heatmap_image_output}")

# 3. Generate JSON Data for ChatGPT
final_data = {
    "timestamp": datetime.now().isoformat(),
    "store_name": "Phygital Demo Store",
    "total_aisles_tracked": len(aisle_zones),
    "aisle_analysis": []
}

for zone in aisle_zones:
    # Create a mask for just this aisle's box
    mask = np.zeros_like(heat_norm)
    pts = np.array(zone['coordinates'], np.int32)
    cv2.fillPoly(mask, [pts], 255)
    
    # Calculate Total Area of the Aisle (in pixels)
    total_area_pixels = cv2.countNonZero(mask)
    
    # Extract heat ONLY within this aisle
    zone_heat = cv2.bitwise_and(heat_norm, heat_norm, mask=mask)
    
    # --- DENSITY CALCULATION ---
    if total_area_pixels > 0:
        total_heat_score = np.sum(zone_heat)
        # Average heat across the WHOLE aisle (including empty space)
        traffic_density = total_heat_score / total_area_pixels
        
        # Calculate coverage (how much of the floor was actually walked on)
        active_pixels = cv2.countNonZero(zone_heat)
        coverage_percent = (active_pixels / total_area_pixels) * 100
    else:
        traffic_density = 0
        coverage_percent = 0
        
    # --- THRESHOLDS ---
    # Adjusted for density scoring (lower numbers are normal now)
    if traffic_density > 25: label = "HIGH_TRAFFIC"
    elif traffic_density > 5: label = "MEDIUM_TRAFFIC"
    else: label = "LOW_TRAFFIC"
    
    final_data["aisle_analysis"].append({
        "aisle_id": zone['id'],
        "product_category": zone['product'],
        "traffic_label": label,
        "density_score": round(float(traffic_density), 2),
        "floor_coverage": f"{round(coverage_percent, 1)}%"
    })

# 4. Save JSON
with open(analytics_output, 'w') as f:
    json.dump(final_data, f, indent=4)

print(f"✅ ChatGPT Data saved to: {analytics_output}")

cap.release()
cv2.destroyAllWindows()