import cv2
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO(r'E:\Astudy\yolov8\ultralytics-main\weights\appletree_EMF.pt')

# 打开视频文件
video_path = r'E:\Astudy\yolov8\ultralytics-main\10_output_video.mp4'
cap = cv2.VideoCapture(video_path)

# 准备保存boxes的文件
output_file =r"E:\Astudy\yolov8\ultralytics-main\runs\track\2024.12.2\EMF_botsort_不好的yaml.txt" # 请替换为你希望保存的文件路径

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
            frame_id=boxes.id
            print(frame_id)

            # 将boxes写入文件
            f.write("Frame {}: {}\n".format(cap.get(cv2.CAP_PROP_POS_FRAMES), frame_id))

        else:
            # 如果视频结束则退出循环
            break

# 关闭文件
cap.release()
cv2.destroyAllWindows()
