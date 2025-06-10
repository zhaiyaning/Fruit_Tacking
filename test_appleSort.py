from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import math
from sort import Sort

# Initialize video capture
cap = cv2.VideoCapture(r"E:\Astudy\yolov8\ultralytics-main\10_output_video.mp4")  # For video

# Initialize YOLO model
model = YOLO(r"E:\Astudy\yolov8\ultralytics-main\weights\appletree_v8.pt")  # large model works better with the GPU

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define video writer to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for video
output_video_path = r'E:\Astudy\mycode\test_track\all_trackresult\track_video\v8_Sort.avi'  # Output video file path
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (1280, 720))  # Create VideoWriter object

# Open a text file to save frame data
with open(r'E:\Astudy\mycode\test_track\all_trackresult\testtxt\v8_Sort_ids.txt', 'w') as file:
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

                # Draw tracking box with a thicker red border to avoid overlap with text
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red border with a thinner line width

                # Prepare the text to display ID and confidence
                text = f"ID: {id}, apple {conf:.2f}"  # Limit confidence to 2 decimal places for readability

                # Adjust the text position to ensure it doesn't overlap the bounding box
                text_offset_y = 10  # This will move the text slightly above the bounding box
                text_position = (max(0, x1), max(35, y1 - text_offset_y))  # Slightly above the box

                # Draw the text with a background (optional) for better readability
                cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Optionally, you can add a background for the text for better visibility
                # cv2.rectangle(img, (x1, y1 - 20), (x2, y1), (0, 0, 255), -1)  # Red background for text

                # Remove the center circle if it's not needed
                # cx, cy = x1 + w // 2, y1 + h // 2
                # cv2.circle(img, (cx, cy), radius=5, color=(255, 0, 255), thickness=cv2.FILLED)

                frame_ids.append(float(id))  # Store the ID for this frame

            # Update the apple count display
            cvzone.putTextRect(img, text=f"Apple Count: {apple_count}", pos=(50, 80),
                               # Change position to (50, 80) to avoid overlap with tracking info
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
