import cv2
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO(r'E:\Astudy\yolov8\ultralytics-main\weights\appletree_EMF.pt')

# 打开视频文件
video_path = r'E:\Astudy\yolov8\ultralytics-main\10_output_video.mp4'
cap = cv2.VideoCapture(video_path)

# 获取视频的基本属性
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 设置输出视频文件路径
output_video_path = r'E:\Astudy\mycode\test_track\all_trackresult\track_video\EMF_Botsort.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4格式
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# 准备保存boxes的文件
output_file = r"E:\Astudy\mycode\test_track\all_trackresult\testtxt\EMF_Botsort.txt"

# 打开文件准备写入
with open(output_file, 'w') as f:
    # 循环遍历视频帧
    while cap.isOpened():
        # 从视频读取一帧
        success, frame = cap.read()

        if success:
            # 在帧上运行YOLOv8追踪，持续追踪帧间的物体
            results = model.track(frame, persist=True)
            # 输出每次追踪推理结果的boxes
            boxes = results[0].boxes

            # 检查是否存在检测框
            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy.cpu().numpy() if box.xyxy is not None else []
                    if len(xyxy) == 4:
                        x1, y1, x2, y2 = map(int, xyxy)
                        conf = box.conf[0].item()  # 置信度
                        track_id = int(box.id[0].item()) if box.id is not None else -1
                        label = f"ID: {track_id} Conf: {conf:.2f}"

                        # 绘制检测框和标签到帧上
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 保存当前帧的boxes信息到文件
                frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                frame_ids = [int(box.id[0].item()) for box in boxes if box.id is not None]
                frame_id_str = ','.join(map(str, frame_ids))
                f.write(f"Frame {frame_id}: {frame_id_str}\n")

            # 将当前帧写入输出视频文件
            out.write(frame)

        else:
            # 如果视频结束则退出循环
            break

# 关闭文件和视频写入器
cap.release()
out.release()
cv2.destroyAllWindows()
