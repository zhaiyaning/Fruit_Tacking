import cv2
import os
import json

# JSON 文件路径
json_file = r'E:\Astudy\mycode\Synthetic-apples-1\ground_truth.json'
# 图片文件夹路径
image_folder = 'E:/Astudy/mycode/Synthetic-apples-1/frames'
# 新文件夹路径，用于保存绘制后的图片
output_folder = 'E:/Astudy/mycode/Synthetic-apples-1/annotated_frames'
# 确保输出文件夹存在，如果不存在则创建它
os.makedirs(output_folder, exist_ok=True)

# 读取 JSON 文件
with open(json_file, 'r') as f:
    data = json.load(f)

# 字典用于存储每个frameNr对应的检测结果
frame_data = {}
# 假设data是你的检测数据列表
for detection in data:
    frameNr = detection.get('frameNr')
    detection_id=detection.get('detection_id')
    object_id = detection.get('object_id')
    box_x = detection.get('box.x')
    box_y = detection.get('box.y')
    box_width = detection.get('box.width')
    box_height = detection.get('box.height')
    # 如果frameNr不在字典中，初始化一个空列表
    if frameNr not in frame_data:
        frame_data[frameNr] = []
    # 将当前检测结果添加到对应frameNr的列表中
    frame_data[frameNr].append((detection_id,object_id, box_x, box_y, box_width, box_height))

# 遍历frame_data中的每个frameNr及其对应的检测结果
for frameNr, detections in frame_data.items():
    # 构建当前帧对应的图片路径
    image_path = os.path.join(image_folder, f'frame_{frameNr:04d}.png')  # 根据实际情况修改图片文件名格式
    # 加载图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        continue
    #遍历当前帧的所有检测结果
    for detection in detections:
        detection_id,object_id, box_x, box_y, box_width, box_height = detection

        # 绘制矩形框
        pt1 = (int(box_x), int(box_y))
        pt2 = (int(box_x + box_width), int(box_y + box_height))
        cv2.rectangle(image, pt1, pt2, (0, 255, 0), 1)

        # 可选：在矩形框上标注物体ID等信息
        text = f'Object ID: {object_id},detection_id:{detection_id}'
        cv2.putText(image, text, (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # 黑色文本
        # 构建保存绘制后图片的文件路径
        output_path = os.path.join(output_folder, f'annotated_frame_{frameNr}.png')  # 根据实际情况修改保存文件名格式

        # 保存绘制后的图片
        cv2.imwrite(output_path, image)

        print(f"Processed annotations for frame {frameNr} and saved to {output_path}")

    print("All frames processed and saved.")






'''
# 遍历图片文件夹中的每一张图片
for filename in os.listdir(image_folder):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # 构建完整的图片文件路径
        image_path = os.path.join(image_folder, filename)
        # 加载图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image {image_path}")
            continue



        # 在图片上绘制示例矩形框（这里仅作示范，实际操作根据需求进行更改）
        cv2.rectangle(image, (50, 50), (200, 200), (0, 255, 0), 2)
        cv2.putText(image, 'Example Text', (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 构建保存绘制后图片的文件路径
        output_path = os.path.join(output_folder, filename)

        # 保存绘制后的图片
        cv2.imwrite(output_path, image)

        print(f"Processed {filename} and saved to {output_path}")

print("All images processed and saved.")
'''




'''
# 写入每一帧图片到视频
for detection in data:
    frameNr = detection.get('frameNr')
    object_id = detection.get('object_id')
    box_x = detection.get('box.x')
    box_y = detection.get('box.y')
    box_width = detection.get('box.width')
    box_height = detection.get('box.height')

    # 构建图片文件路径
    image_path = os.path.join(image_folder, f'frame_{frameNr:04d}.png')

    # 加载图片
    image = cv2.imread(image_path)


    if image is None:
        print(f"Error: Unable to load image {image_path}")
        continue

    #绘制检测框
    cv2.rectangle(image, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 255, 0), 2)

    # 在检测框上标注对象信息
    text = f'Object ID: {object_id}'
    cv2.putText(image, text, (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


#video.write(cv2.imread(os.path.join(image_folder, image)))

# 显示所有帧的图片
for frameNr, image in frames_images.items():
    cv2.imshow(f'Frame {frameNr}', image)

cv2.waitKey(0)
cv2.destroyAllWindows()

'''