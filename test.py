from ultralytics import YOLO
import cv2  # OpenCV用于图像处理

if __name__ == '__main__':
    # Load a model
    model = YOLO(r'E:\Astudy\yolov8\ultralytics-main\Effient.pt')  # 加载一个官方的检测模型
    results = model(r'E:\Astudy\yolov8\ultralytics-main\appletest_1000\images\20150919_174151_image1_png.rf.1ce28835b0d18af2dcf62db5ca89ae67.jpg')  # predict on an image

    # 获取预测结果
    predictions = results[0]  # 获取第一个图像的结果

    # 打印预测结果（边界框、类别、置信度）
    for bbox in predictions.boxes:  # 遍历所有边界框
        x1, y1, x2, y2 = bbox.xyxy[0]  # 获取边界框坐标
        conf = bbox.conf[0]  # 置信度
        cls = int(bbox.cls[0])  # 类别
        print(f"Detected class: {cls}, Confidence: {conf:.2f}, Box coordinates: ({x1}, {y1}), ({x2}, {y2})")

    # 获取带有检测框的图像并进行可视化
    result_img = predictions.plot()  # 获取带有检测框的图像

    # 将结果图像从 RGB 转换为 BGR
    #result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)

    # 使用 OpenCV 显示结果
    cv2.imshow('Detected Image', result_img)  # 显示带有检测框的图像
    cv2.waitKey(0)  # 等待用户按键
    cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
