import cv2
import json
import os
import numpy as np

def rebuild_visual():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    floor_plan_path = os.path.join(base_dir, "assets", "floor_plans", "sample_floor_plan.png")
    
    # Reads from the NEW folder
    new_json_path = os.path.join(base_dir, "assets", "ai_agent_returns", "optimized_zones.json")
    output_image_path = os.path.join(base_dir, "assets", "ai_agent_returns", "ai_proposed_layout.png")

    if not os.path.exists(floor_plan_path):
        return "Error: Floor plan not found."
    
    if not os.path.exists(new_json_path):
        return "Error: Optimized JSON not found. Run the scanner/AI first!"

    img = cv2.imread(floor_plan_path)
    
    with open(new_json_path, 'r') as f:
        new_zones = json.load(f)

    print(f"🔄 Rebuilding Layout with {len(new_zones)} optimized zones...")

    for zone in new_zones:
        product_name = zone['product']
        pts = np.array(zone['coordinates'], np.int32)

        # Draw Green Box
        cv2.polylines(img, [pts], True, (0, 200, 0), 3)

        # Calculate Center
        M = cv2.moments(pts)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = pts[0][0], pts[0][1]

        # Draw Text (White with Black Outline)
        text_size = cv2.getTextSize(product_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = cX - text_size[0] // 2
        text_y = cY + text_size[1] // 2

        cv2.putText(img, product_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(img, product_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(output_image_path, img)
    print(f"✅ Proposed Layout saved to: {output_image_path}")
    return "Layout Rebuild Complete"

if __name__ == "__main__":
    rebuild_visual()