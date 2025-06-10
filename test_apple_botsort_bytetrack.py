from ultralytics import YOLO
import cv2

# 加载YOLOv8模型
model = YOLO(r'E:\Astudy\yolov8\ultralytics-main\weights\appletree_EMF.pt')

# 打开视频
video_path = r'E:\Astudy\yolov8\ultralytics-main\10_output_video.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 视频输出路径
output_video_path = r'E:\Astudy\mycode\test_track\all_trackresult\track_video\EMF_Botsort.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

output_file = r"E:\Astudy\mycode\test_track\all_trackresult\testtxt\EMF_Botsort.txt"

# 选择 Tracker 配置
tracker_config = "botsort.yaml"  # 或者改为 "bytetrack.yaml"

with open(output_file, 'w') as f:
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # 在帧上运行YOLOv8追踪，指定 tracker 参数
            results = model.track(frame, persist=True, tracker=tracker_config)
            boxes = results[0].boxes

            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy.cpu().numpy() if box.xyxy is not None else []
                    if len(xyxy) == 4:
                        x1, y1, x2, y2 = map(int, xyxy)
                        conf = box.conf[0].item()
                        track_id = int(box.id[0].item()) if box.id is not None else -1
                        label = f"ID: {track_id} Conf: {conf:.2f}"

                        # 绘制检测框和标签
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 保存当前帧的boxes信息到文件
                frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                frame_ids = [int(box.id[0].item()) for box in boxes if box.id is not None]
                frame_id_str = ','.join(map(str, frame_ids))
                f.write(f"Frame {frame_id}: {frame_id_str}\n")

            out.write(frame)
        else:
            break

cap.release()
out.release()
cv2.destroyAllWindows()
