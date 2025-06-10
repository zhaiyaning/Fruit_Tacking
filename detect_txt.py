from ultralytics import YOLO
import cv2  # OpenCV用于图像处理
import os

if __name__ == '__main__':
    # 加载模型
    model = YOLO(r'E:\Astudy\yolov8\ultralytics-main\weights\apple_v8.pt')  # 加载模型
    img_folder = r'E:\Astudy\yolov8\ultralytics-main\detect_images\images'  # 输入图像文件夹路径
    output_txt_folder = r'E:\Astudy\yolov8\ultralytics-main\detect_images\txt\YOLOv8n-txt'  # 输出结果txt文件夹路径

    # 确保输出文件夹存在
    os.makedirs(output_txt_folder, exist_ok=True)

    # 遍历输入图像文件夹中的所有图像文件
    for img_file in os.listdir(img_folder):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):  # 判断是否为图片文件
            img_path = os.path.join(img_folder, img_file)  # 输入图像路径
            txt_path = os.path.join(output_txt_folder, img_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt'))  # 输出txt路径

            # 读取图像
            image = cv2.imread(img_path)

            # 进行预测
            results = model(img_path)
            predictions = results[0]  # 获取预测结果

            # 创建文件保存结果
            with open(txt_path, 'w') as f:
                # 遍历所有检测到的目标
                for bbox in predictions.boxes:
                    x1, y1, x2, y2 = bbox.xyxy[0].tolist()  # 获取边界框坐标
                    conf = bbox.conf[0].item()  # 置信度
                    cls = 0  # apple类别标识为0

                    # 计算中心坐标和宽高
                    img_h, img_w = image.shape[:2]
                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h

                    # 写入txt文件
                    f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")

            print(f"检测结果已保存到: {txt_path}")

    print("所有图像处理完成。")
