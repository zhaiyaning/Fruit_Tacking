import cv2

# 读取图像
img = cv2.imread('car_img.jpg')


# 鼠标事件的回调函数
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 获取点击坐标
        xy = "%d,%d" % (x, y)

        # 在点击的点上画一个蓝色的圆
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)

        # 将坐标显示为文本
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)

        # 显示修改后的图像
        cv2.imshow("image", img)


# 创建一个窗口
cv2.namedWindow("image")

# 设置鼠标回调函数
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)

# 进入交互循环
while True:
    # 在窗口中显示图像
    cv2.imshow("image", img)

    # 等待按键事件，如果按下 'Esc' 键则退出循环
    if cv2.waitKey(0) & 0xFF == 27:
        break

# 关闭所有OpenCV窗口
cv2.destroyAllWindows()
