import cv2
import os

def extract_frames(video_path, output_folder, num_frames=250):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 确保不超过视频的总帧数
    num_frames = min(num_frames, frame_count)

    # 逐帧读取视频
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        # 保存帧为图片
        cv2.imwrite(os.path.join(output_folder, f"frame_{i}.jpg"), frame)

    # 释放资源
    cap.release()

# 设置视频文件路径和输出文件夹路径
video_path = r'E:\Astudy\mycode\test_track\all_trackresult\track_video\DynamicKF.mp4'  # 替换为你的视频文件路径
output_folder = r'E:\Astudy\mycode\test_track\all_trackresult\track_frame\DynamicKF'  # 替换为你想要保存帧的文件夹路径

# 提取前all帧
extract_frames(video_path, output_folder, num_frames=250)
