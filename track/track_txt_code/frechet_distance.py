import numpy as np

def read_tum_file(filepath):
    """从TUM格式的文件中读取路径数据，提取每个位姿的x和y坐标。"""
    path = []
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue  # 忽略注释行
            parts = line.strip().split()
            if len(parts) >= 8:  # 确保是完整的数据行
                x = float(parts[1])
                y = float(parts[2])
                path.append((x, y))
    return path

def calculate_euclid(point_a, point_b):
    """计算两点之间的欧几里得距离。"""
    return np.sqrt((point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2)

def calculate_frechet_distance(curve_a, curve_b):
    """迭代方式计算两条路径之间的弗雷歇距离。"""
    n = len(curve_a)
    m = len(curve_b)
    dp = [[float('inf')] * m for _ in range(n)]

    dp[0][0] = calculate_euclid(curve_a[0], curve_b[0])

    for i in range(1, n):
        dp[i][0] = max(dp[i-1][0], calculate_euclid(curve_a[i], curve_b[0]))

    for j in range(1, m):
        dp[0][j] = max(dp[0][j-1], calculate_euclid(curve_a[0], curve_b[j]))

    for i in range(1, n):
        for j in range(1, m):
            dp[i][j] = max(min(dp[i-1][j], dp[i-1][j-1], dp[i][j-1]), calculate_euclid(curve_a[i], curve_b[j]))

    return dp[n-1][m-1]

def get_similarity(curve_a, curve_b):
    """计算两条路径之间的弗雷歇距离。"""
    return calculate_frechet_distance(curve_a, curve_b)

# 示例用法
curve_a = read_tum_file('SCI_Experiment/SlowTurn_Motion/wo_pose.tum')
curve_b = read_tum_file('SCI_Experiment/SlowTurn_Motion/state_pose.tum')
curve_c = read_tum_file('SCI_Experiment/SlowTurn_Motion/wo_sync.tum')
similarity_AC = get_similarity(curve_a, curve_c)
similarity_BC = get_similarity(curve_b, curve_c)
print("Fréchet Distance between FIS-IESKF AND Standard Pose:", similarity_AC)
print("Fréchet Distance between EKF AND Standard Pose:", similarity_BC)

