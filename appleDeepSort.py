# 导入所需库
from ultralytics import YOLO
import cv2
import cvzone
import math
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils import get_class_color, estimatedSpeed
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 允许重复导入 OpenCV 库的问题

# 选择视频源：摄像头或视频文件
# cap = cv2.VideoCapture(0)  # 对于摄像头
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("/root/autodl-tmp/ultralytics/10_output_video.mp4")  # 对于视频

# 加载 YOLO 模型
model = YOLO("/root/autodl-tmp/ultralytics/runs/detect/train2/weights/best.pt")  # 使用较大的模型以获得更好的 GPU 性能


# 初始化 DeepSort 跟踪器
tracker = DeepSort(
    max_iou_distance=0.7,
    max_age=2,
    n_init=3,
    nms_max_overlap=3.0,
    max_cosine_distance=0.2)

# Define video writer to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for video
output_video_path = 'deepsort_output_video.avi'  # Output video file path
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (1280, 720))  # Create VideoWriter object
coordinatesDict = dict()
while True:
    # 读取视频帧
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))

    # 对图像进行目标检测
    results = model(img, stream=True)
    detections = list()
    apple_count = 0  # Initialize apple counter

    # 处理检测结果
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

            # 仅考虑车辆类别并且置信度大于0.5的检测结果
            if currentClass == 'apple' and conf > 0.5:
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, cls))
                apple_count += 1  # Increment apple counter



    # 使用 DeepSort 进行目标跟踪
    tracks = tracker.update_tracks(detections, frame=img)

    # 处理跟踪结果
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltrb()
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        w, h = x2 - x1, y2 - y1

        co_ord = [x1, y1]

        # 更新坐标字典
        if track_id not in coordinatesDict:
            coordinatesDict[track_id] = co_ord
        else:
            if len(coordinatesDict[track_id]) > 2:
                del coordinatesDict[track_id][-3:-1]
            coordinatesDict[track_id].append(co_ord[0])
            coordinatesDict[track_id].append(co_ord[1])

        # 估计速度
        estimatedSpeedValue = 0
        if len(coordinatesDict[track_id]) > 2:
            location1 = [coordinatesDict[track_id][0], coordinatesDict[track_id][2]]
            location2 = [coordinatesDict[track_id][1], coordinatesDict[track_id][3]]
            estimatedSpeedValue = estimatedSpeed(location1, location2)

        # 获取对象类别和颜色
        cls = track.get_det_class()
        currentClass = model.names[cls]
        clsColor = get_class_color(currentClass)

        # 在帧上绘制跟踪信息
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=clsColor)
        cvzone.putTextRect(
            img,
            text=f"{model.names[cls]} {estimatedSpeedValue} km/h",
            pos=(max(0, x1), max(35, y1)),
            colorR=clsColor,
            scale=1,
            thickness=1,
            offset=2)
        cx, cy = x1 + w // 2, y1 + h // 2

        # 在帧上绘制速度信息
        cv2.circle(img, (cx, cy), radius=5, color=clsColor, thickness=cv2.FILLED)

    cvzone.putTextRect(img, text=f"Apple Count: {apple_count}", pos=(50, 50),
                           scale=2, thickness=3, offset=10, colorR=(0, 255, 0))
        # Write the processed frame to the output video
    out.write(img)

        # Optional: Print frame number for monitoring
    print("Processed frame...")  # You can replace this with any other logging mechanism if needed.



# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows (if any were opened)
cv2.destroyAllWindows()
