from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import math
from sort import Sort

# Initialize video capture
cap = cv2.VideoCapture(r"E:\Astudy\yolov8\ultralytics-main\10_output_video.mp4")  # For video

# Initialize YOLO model
model = YOLO(r"E:\Astudy\yolov8\ultralytics-main\weights\appletreet_Effient.pt")  # large model works better with the GPU

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define video writer to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for video
output_video_path = 'Sortoutput_video.avi'  # Output video file path
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (1280, 720))  # Create VideoWriter object

# Open a text file to save frame data
with open('sorttracking_ids.txt', 'w') as file:
    frame_number = 0  # Initialize frame counter

    while True:
        success, img = cap.read()
        if success:
            img = cv2.resize(img, (1280, 720))
            frame_number += 1  # Increment frame counter

            # Use YOLO for object detection
            results = model(img, stream=True)
            detections = np.empty((0, 6))
            apple_count = 0  # Initialize apple counter

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    bbox = (x1, y1, w, h)
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    currentClass = model.names[cls]

                    # Only keep detections for specified class (apple)
                    if currentClass == "apple":
                        currentArray = np.array([x1, y1, x2, y2, conf, cls])
                        detections = np.vstack((detections, currentArray))
                        apple_count += 1  # Increment apple counter

            # Use SORT for object tracking
            classes_array = detections[:, -1:]
            resultsTracker = tracker.update(detections)
            try:
                resultsTracker = np.hstack((resultsTracker, classes_array))
            except ValueError:
                classes_array = classes_array[:resultsTracker.shape[0], :]
                resultsTracker = np.hstack((resultsTracker, classes_array))

            # Collect IDs for the current frame
            frame_ids = []
            for result in resultsTracker:
                x1, y1, x2, y2, id, cls = result
                x1, y1, x2, y2, id, cls = int(x1), int(y1), int(x2), int(y2), int(id), int(cls)
                w, h = x2 - x1, y2 - y1

                # Draw tracking box and ID
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
                cvzone.putTextRect(img, text=f"{model.names[cls]} {id}", pos=(max(0, x1), max(35, y1)),
                                   scale=2, thickness=3, offset=10)

                cx, cy = x1 + w // 2, y1 + h // 2
                cv2.circle(img, (cx, cy), radius=5, color=(255, 0, 255), thickness=cv2.FILLED)

                currentClass = model.names[cls]
                frame_ids.append(float(id))  # Store the ID for this frame

            # Save frame IDs to the file
            if frame_ids:
                file.write(f"Frame {frame_number}.0: tensor([{', '.join(map(str, frame_ids))}])\n")

            # Display the count of apples
            cvzone.putTextRect(img, text=f"Apple Count: {apple_count}", pos=(50, 50),
                               scale=2, thickness=3, offset=10, colorR=(0, 255, 0))
            # Write the processed frame to the output video
            out.write(img)

            # Optional: Print frame number for monitoring
            print(f"Processed frame {frame_number}...")  # You can replace this with any other logging mechanism if needed.

        else:
            break

# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows (if any were opened)
cv2.destroyAllWindows()
