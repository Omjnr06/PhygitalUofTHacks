import cv2
import torch
import os
import json
import numpy as np
from datetime import datetime
from ultralytics import YOLO, solutions
from ai_agent import get_store_insights 

def run_scanner():
    # --- SETUP DEVICES & MODELS ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolo11n.pt").to(device)
    if device == "cuda": model.half() 

    # --- PATHS ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(base_dir, "assets", "test_video_footage.mp4")
    floor_plan_path = os.path.join(base_dir, "assets", "floor_plans", "sample_floor_plan.png")
    zones_file = os.path.join(base_dir, "assets", "zones.json")
    
    analytics_output = os.path.join(base_dir, "assets", "store_analytics.json")
    heatmap_image_output = os.path.join(base_dir, "assets", "floor_plans", "final_heatmap.png")

    # --- LOAD ZONES ---
    if os.path.exists(zones_file):
        with open(zones_file, 'r') as f:
            aisle_zones = json.load(f)
        print(f"✅ Loaded {len(aisle_zones)} Aisles from zones.json")
    else:
        return {"error": "zones.json not found. Run define_zones.py first!"}

    # --- VIDEO & FLOOR PLAN SETUP ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file."}

    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    floor_plan = cv2.imread(floor_plan_path)
    if floor_plan is None:
        return {"error": "Could not find floor plan image."}

    h_fp, w_fp = floor_plan.shape[:2]
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
            track_results = model.track(frame, persist=True, classes=[0], verbose=False, vid_stride=stride)
            
            if track_results[0].boxes.id is not None:
                boxes = track_results[0].boxes.xyxy.cpu().numpy()
                for box in boxes:
                    # Feet position
                    vid_x = (box[0] + box[2]) / 2
                    vid_y = box[3] 
                    
                    # Simple Scaling
                    fp_x = int(vid_x * (w_fp / video_w))
                    fp_y = int(vid_y * (h_fp / video_h))

                    fp_x = max(0, min(fp_x, w_fp - 1))
                    fp_y = max(0, min(fp_y, h_fp - 1))
                    
                    cv2.circle(heat_layer, (fp_x, fp_y), 15, 1, -1)

            # Optional: Comment out imshow if running on a server without a screen
            annotated_frame = track_results[0].plot()
            cv2.imshow("Phygital Scanner", cv2.resize(annotated_frame, (1080, 720)))

        if cv2.waitKey(1) & 0xFF == ord("q"): break

    # --- ANALYTICS GENERATION ---
    print("\nProcessing Analytics...")
    cap.release()
    cv2.destroyAllWindows()

    heat_norm = cv2.normalize(heat_layer, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Save Visual Heatmap
    heatmap_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    final_overlay = cv2.addWeighted(floor_plan, 0.6, heatmap_color, 0.4, 0)
    cv2.imwrite(heatmap_image_output, final_overlay)

    # Generate JSON
    final_data = {
        "timestamp": datetime.now().isoformat(),
        "store_name": "Phygital Demo Store",
        "total_aisles_tracked": len(aisle_zones),
        "aisle_analysis": []
    }

    for zone in aisle_zones:
        mask = np.zeros_like(heat_norm)
        pts = np.array(zone['coordinates'], np.int32)
        cv2.fillPoly(mask, [pts], 255)
        
        total_area_pixels = cv2.countNonZero(mask)
        zone_heat = cv2.bitwise_and(heat_norm, heat_norm, mask=mask)
        
        if total_area_pixels > 0:
            total_heat_score = np.sum(zone_heat)
            traffic_density = total_heat_score / total_area_pixels
            active_pixels = cv2.countNonZero(zone_heat)
            coverage_percent = (active_pixels / total_area_pixels) * 100
        else:
            traffic_density = 0
            coverage_percent = 0
            
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

    with open(analytics_output, 'w') as f:
        json.dump(final_data, f, indent=4)

    print(f"✅ Data saved to: {analytics_output}")

    # --- TRIGGER AI ---
    print("🚀 Triggering OpenRouter AI...")
    ai_status = get_store_insights()
    
    return {
        "status": "complete", 
        "analytics_file": analytics_output, 
        "ai_status": ai_status
    }

if __name__ == "__main__":
    run_scanner()