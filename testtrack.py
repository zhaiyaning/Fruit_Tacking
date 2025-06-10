import cv2
import torch
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO(r'E:\Astudy\yolov8\ultralytics-main\weights\appletree_EMF.pt')

# 打开视频文件
video_path = r'E:\Astudy\yolov8\ultralytics-main\10_output_video.mp4'
cap = cv2.VideoCapture(video_path)

# 获取视频的帧率、宽度和高度
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 准备保存检测框视频的文件
output_video_path = r'E:\Astudy\mycode\test_track\all_trackresult\track_video\DynamticKF.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4格式
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# 准备保存boxes的文件
output_file =r"E:\Astudy\mycode\test_track\all_trackresult\testtxt\DynamticKF.txt"  # 请替换为你希望保存的文件路径

# 打开文件准备写入
with open(output_file, 'w') as f:
    # 循环遍历视频帧
    while cap.isOpened():
        # 从视频读取一帧
        success, frame = cap.read()

        if success:
            # 在帧上运行YOLOv8追踪，持续追踪帧间的物体
            results = model.track(frame, persist=True)

            # 获取检测框和标签
            boxes = results[0].boxes
            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # 获取当前帧的ID

            # 将框ID转化为tensor格式
            box_ids = torch.tensor([box.id[0] for box in boxes])

            # 在帧上绘制检测框和标签
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取坐标
                conf = box.conf[0]  # 获取置信度
                label = f"ID: {box.id[0]} Conf: {conf:.2f}"  # 构建标签

                # 绘制矩形框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 绿色框

                # 在框上绘制标签
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 将框ID数据保存为tensor格式，格式化为"tensor([...])"
            f.write(f"Frame {frame_id:.1f}: {box_ids}\n")

            # 将处理后的帧写入到输出视频
            out.write(frame)

        else:
            # 如果视频结束则退出循环
            break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
