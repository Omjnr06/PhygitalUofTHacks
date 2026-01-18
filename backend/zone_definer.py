import cv2
import json
import os
import numpy as np

# --- SETUP ---
# Path to your floor plan
floor_plan_path = r"assets\floor_plans\sample_floor_plan.png"
output_file = r"assets\zones.json"

# Check file exists
if not os.path.exists(floor_plan_path):
    print(f"❌ Error: Floor plan not found at {floor_plan_path}")
    exit()

img = cv2.imread(floor_plan_path)
zones = []
current_points = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Zone Definer", img)
        current_points.append([x, y])
        
        if len(current_points) == 4:
            # We have a full box (4 corners)
            pts = np.array(current_points, np.int32)
            cv2.polylines(img, [pts], True, (0, 0, 255), 2)
            cv2.imshow("Zone Definer", img)
            
            # Ask user for the product name
            zone_name = input(f"\nenter Product Name for Zone {len(zones)+1} (e.g., 'Cereal' or 'Dairy'): ")
            
            zones.append({
                "id": len(zones) + 1,
                "product": zone_name,
                "coordinates": current_points.copy() # Save the 4 points
            })
            print(f"✅ Saved Zone {len(zones)}: {zone_name}")
            current_points.clear() # Reset for next zone

cv2.imshow("Zone Definer", img)
cv2.setMouseCallback("Zone Definer", click_event)

print("--- INSTRUCTIONS ---")
print("1. Click 4 corners to define an Aisle.")
print("2. Look at the terminal -> Type the Product Name -> Hit Enter.")
print("3. Repeat for all 7 Aisles.")
print("4. Press 'q' when finished to save.")

while True:
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

# Save to JSON
with open(output_file, 'w') as f:
    json.dump(zones, f, indent=4)

print(f"\nSUCCESS: Saved {len(zones)} zones to {output_file}")
cv2.destroyAllWindows()