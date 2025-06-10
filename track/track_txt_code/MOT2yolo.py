import os
import pandas as pd

def convert_mot_csv_to_yolo(csv_file, output_dir, class_mapping, img_width=1920, img_height=1080):
    # 创建YOLO格式的标签文件存放目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 遍历每一行数据
    for index, row in df.iterrows():
        frame_id = row[0] # 帧ID
        object_id = row[1] # 物体ID
        class_label = 0  # MOT中的物体ID映射到YOLO类别
        x, y, w, h = row[2], row[3], row[4], row[5]  # 包围框坐标和尺寸

        # 计算YOLO格式的中心坐标和相对宽度、高度
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height

        # 构建YOLO格式标签内容
        label_content = f"{class_label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

        # 写入YOLO格式的标签文件
        label_file = os.path.join(output_dir, f"frame_{frame_id}.txt")
        with open(label_file, 'a') as f:
            f.write(label_content + '\n')

# 示例用法
csv_file = r'E:\Astudy\mycode\test_track\144.csv'  # 替换为你的MOT数据集CSV文件路径
output_dir = r'E:\Astudy\mycode\test_track\labels'  # 替换为你想要保存YOLO格式标签文件的目录路径
class_mapping = {0: 'apple'}  # MOT物体ID到YOLO类别的映射

convert_mot_csv_to_yolo(csv_file, output_dir, class_mapping)
