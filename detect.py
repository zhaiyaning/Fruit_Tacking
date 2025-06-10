from ultralytics import YOLO
import cv2  # OpenCV用于图像处理

if __name__ == '__main__':
    # 加载模型
    model = YOLO(r'E:\Astudy\yolov8\ultralytics-main\weights\apple_EMF.pt')  # 加载模型
    img_path = r'E:\Astudy\yolov8\ultralytics-main\detect_images\a-1.jpg'  # 输入图像路径
    out_path = r'E:\Astudy\yolov8\ultralytics-main\detect_images\EMF-YOLO\a-1_detected.jpg'  # 输出图像路径

    # 读取图像
    image = cv2.imread(img_path)

    # 进行预测
    results = model(img_path)
    predictions = results[0]  # 获取预测结果

    # 遍历所有检测到的目标
    for bbox in predictions.boxes:
        x1, y1, x2, y2 = map(int, bbox.xyxy[0])  # 获取边界框坐标（转换为整数）
        conf = bbox.conf[0].item()  # 置信度
        cls = int(bbox.cls[0])  # 类别
        label = f"apple {conf:.2f}"

        # 计算文本大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_w, text_h = text_size

        # 底色框（白色）
        text_bg_x1 = x1
        text_bg_y1 = y1 - text_h - 5  # 上方留5像素间距
        text_bg_x2 = x1 + text_w + 6  # 右侧留6像素间距
        text_bg_y2 = y1  # 高度与文本对齐

        # 确保底色框不会超出图像上边界
        if text_bg_y1 < 0:
            text_bg_y1 = y1
            text_bg_y2 = y1 + text_h + 5

        # 绘制底色框
        cv2.rectangle(image, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), (255, 255, 255), -1)

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

        # 绘制文本（黑色）
        cv2.putText(image, label, (x1 + 2, text_bg_y2 - 5), font, font_scale, (0, 0, 0), thickness)

    # 保存可视化结果
    cv2.imwrite(out_path, image)

    # 显示结果
    cv2.imshow('Detected Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
