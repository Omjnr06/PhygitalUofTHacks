import cv2
import torch
from ultralytics import YOLO
from ultralytics.solutions import heatmap

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11n.pt").to(device)

video_path = r"assets\test_video_footage.mp4"
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

heatmap_obj = heatmap.Heatmap(
    colormap=cv2.COLORMAP_JET,
    view_img=False,
    shape=(height, width),
    names=model.names,
)

final_frame = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    input_frame = cv2.resize(frame, (640, 640))
    results = model.track(input_frame, persist=True, classes=[0], verbose=False)

    frame = heatmap_obj.generate_heatmap(frame, tracks=results)
    final_frame = frame # Update the reference for the final export

    cv2.imshow("Heatmap", cv2.resize(frame, (1280, 720)))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Export the last state of the heatmap
if final_frame is not None:
    cv2.imwrite("heatmap_output.png", final_frame)
    print("Heatmap saved as heatmap_output.png")

cap.release()
cv2.destroyAllWindows()