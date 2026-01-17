import cv2
from collections import deque
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# 1. Using the standard model for now so it auto-downloads
model = YOLO("yolo11n.pt") 

# 2. Raw string for Windows paths
video_path = r"assets\test_video_footage.mp4" 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}")
    exit()

track_history = {} 

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # We tell YOLO to only track 'person' (class 0)
    results = model.track(frame, persist=True, classes=[0]) 

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            radius = int((x2 - x1) / 4) # Smaller circle for testing

            cv2.circle(frame, center, radius, colors(track_id, True), 2)
            
            # Trajectory logic
            track = track_history.get(track_id, deque(maxlen=30))
            track.append(center)
            track_history[track_id] = track
            for i in range(1, len(track)):
                cv2.line(frame, track[i-1], track[i], colors(track_id, True), 2)

    cv2.imshow("Tracking Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()