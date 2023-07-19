import cv2
from ultralytics import YOLO
import numpy as np

cap = cv2.VideoCapture("NathanBBall.mov")
class_label_map = {
    0: "basketball",
    1: "rim"
}
model = YOLO("BeforeBest.pt")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device="mps")
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    scores = np.array(result.boxes.conf.cpu(), dtype="float")

    for cls, bbox, score in zip(classes, bboxes, scores):
        (x, y, x2, y2) = bbox
        if cls in class_label_map:
            label = class_label_map[cls]
            confidence = f"{score:.2f}"  # Convert confidence score to string with 2 decimal places
            text = f"{label} ({confidence})"  # Combine label and confidence score
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    out.write(frame)  # Write the frame with detections to the output video

    cv2.imshow("Imp", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
out.release()  # Release the VideoWriter
cv2.destroyAllWindows()
