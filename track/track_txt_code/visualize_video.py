import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np

# JSON 文件路径
json_file = r'E:\Astudy\mycode\Synthetic-apples-1\ground_truth.json'

# 视频文件路径
video_file = r'E:\Astudy\mycode\24_output_video.mp4'

# 输出视频文件路径
output_video_file = 'draw_video.mp4'  # 修改输出视频文件为.mp4格式

# 读取 JSON 文件
with open(json_file, 'r') as f:
    data = json.load(f)

# 打开视频文件
cap = cv2.VideoCapture(video_file)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 获取视频帧的宽度和高度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建 VideoWriter 对象，使用 H264 编解码器以支持.mp4格式
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 修改为支持.mp4格式的编解码器
out = cv2.VideoWriter(output_video_file, fourcc, 30.0, (frame_width, frame_height))

# 创建 Matplotlib 画布和子图
fig, ax = plt.subplots(1)

# 读取视频的每一帧
frame_nr = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为 RGB 格式用于 Matplotlib 显示
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 绘制当前帧的图像
    ax.imshow(frame_rgb)

    # 查找当前帧中的目标并绘制矩形框和文本信息
    for obj in data:
        if obj['frameNr'] == frame_nr:
            x = obj['box.x']
            y = obj['box.y']
            width = obj['box.width']
            height = obj['box.height']
            detection_id = obj['detection_id']
            object_id = obj['object_id']

            # 绘制矩形框
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # 添加文本信息
            ax.text(x, y - 10, f"Object ID: {object_id}, Detection ID: {detection_id}", color='r')

    # 关闭坐标轴
    plt.axis('off')

    # 将 Matplotlib 图像保存为 OpenCV 图像格式
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    # 写入视频文件
    out.write(img)

    # 清除当前帧以准备绘制下一帧
    ax.clear()

    frame_nr += 1

# 释放视频文件、关闭窗口和释放 VideoWriter
cap.release()
out.release()
plt.close()
