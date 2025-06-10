import math

# 定义颜色调色板
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def get_class_color(cls):
    """
    获取对象类别对应的颜色

    Args:
    - cls (str): 对象的类别

    Returns:
    - tuple: RGB颜色值
    """
    if cls == 'car':
        color = (204, 51, 0)
    elif cls == 'truck':
        color = (22, 82, 17)
    elif cls == 'motorbike':
        color = (255, 0, 85)
    else:
        # 对于未知类别，使用调色板生成动态颜色
        color = [int((p * (2 ** 2 - 14 + 1)) % 255) for p in palette]
    return tuple(color)


def estimatedSpeed(location1, location2):
    """
    估算目标速度
    Args:
    - location1 (list): 起始位置坐标 [x1, y1]
    - location2 (list): 结束位置坐标 [x2, y2]
    Returns:
    - int: 估算速度（千米/小时）
    """
    # 计算两点之间的欧几里得距离（像素）
    d_pixel = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))

    # 设置每米的像素数（像素每米）
    ppm = 4  # 可以根据对象离摄像头的距离动态调整这个值

    # 将像素距离转换为实际距离（米）
    d_meters = d_pixel / ppm

    # 时间常数，用于速度估算
    time_constant = 15 * 3.6  # 这个值可以根据实际情况调整

    # 估算速度（千米/小时）
    speed = (d_meters * time_constant) / 100

    return int(speed)
