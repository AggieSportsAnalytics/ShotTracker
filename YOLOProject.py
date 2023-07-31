import cv2
from ultralytics import YOLO
import numpy as np
import time

cap = cv2.VideoCapture("DairyCourts.MOV")
class_label_map = {
    0: "basketball",
    1: "rim"
}
model = YOLO("best.pt")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

start_time = time.time()
prevY = 0
prevX = 0
rimX = 0
rimY = 0
counter = 0
miss_counter = 0
while True:
    elapsed_time = time.time() - start_time
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device="mps")
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    scores = np.array(result.boxes.conf.cpu(), dtype="float")
    
    poslistX = []
    poslistY = []
    basketball_count = []
    for cls, bbox, score in zip(classes, bboxes, scores):
        (x, y, x2, y2) = bbox
        if cls in class_label_map and score > 0.6:
            label = class_label_map[cls]
            confidence = f"{score:.2f}"  # Convert confidence score to string with 2 decimal places
            
            #Checking for prediction
            if label == 'basketball':
                basketball_count.append([x,y,score])
                #if more than one basketball
                if len(basketball_count) > 1:
                    for var in basketball_count:
                        if var[2] > score:
                            print(x)
                            score = var[2]
                            x = var[0]
                            y = var[1]
                cv2.putText(frame, f"Rim Height: {rimY}", (800,500), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                cv2.putText(frame, f"Rim X: {rimX}", (800,400), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                cv2.putText(frame, f"Prev Ball X: {prevX}", (800,300), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                cv2.putText(frame, f"Prev ball: {prevY}", (800,200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                cv2.putText(frame, f"Shots Made: {counter}", (800,600), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                cv2.putText(frame, f"Shots Missed: {miss_counter}", (800,700), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                if counter > 0 or miss_counter > 0:
                    cv2.putText(frame, f"Percent Made: {round(counter / (miss_counter + counter), 2) * 100}%", (800,800), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                if y > rimY and prevY < rimY:
                    cv2.putText(frame, f"Ball height: {y}", (200,200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                    cv2.putText(frame, f"Rim height: {rimY}", (200,300), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                    if x > rimX - 100 and x < rimX + 100:
                        cv2.putText(frame, "IN", (500,500), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)  
                        if elapsed_time > 5:
                            counter += 1
                            start_time = time.time()
                        break  
                    else:
                        if elapsed_time > 4:
                            miss_counter += 1
                            start_time = time.time()
                        cv2.putText(frame, "OUT", (500,500), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)  
                prevX = x
                prevY = y
            else:
                rimX = x
                rimY = y
                
            text = f"{label} ({confidence}) {x}, {y}"  # Combine label and confidence score
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        

    out.write(frame)  # Write the frame with detections to the output video

    cv2.imshow("Imp", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

    time.sleep(0.000001)

cap.release()
out.release()  # Release the VideoWriter
cv2.destroyAllWindows()
