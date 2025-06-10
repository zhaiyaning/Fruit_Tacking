import cv2
import os

# 图片文件夹路径
image_folder = r'E:\Astudy\mycode\Synthetic-apples-1\annotated_frames'

# 视频输出路径和名称
video_name = '10_output_annotatedvideo.mp4'

# 获取图片文件夹中的所有图片文件名
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

# 按文件名排序，确保视频帧的顺序正确
images.sort()

# 图片文件路径
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# 视频编解码器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 创建视频写入对象
video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))

# 写入每一帧图片到视频
for image in images:
    print(image)
    video.write(cv2.imread(os.path.join(image_folder, image)))

# 释放资源
cv2.destroyAllWindows()
video.release()
