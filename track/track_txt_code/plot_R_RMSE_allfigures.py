import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from get_gt_trackdata import process_mot_dataset
from get_yolo_trackdata import get_track_data
from scipy.signal import savgol_filter
import seaborn as sns  # 用于绘制KDE曲线
from matplotlib.lines import Line2D



# 设置全局字体为支持中文的字体，例如 SimHei
#plt.rcParams['font.family'] = ['SimHei']  # Windows 下可使用 SimHei 或 Microsoft YaHei
# 假设tracking_results和ground_truth已经正确定义和填充
n = 250
tracking_results = get_track_data(n, r"E:\Astudy\mycode\test_track\all_trackresult\track_txt\botsort_EMF.txt")
ground_truth = process_mot_dataset(n)

# 计算每个列表中的ID数量
tracking_counts = [len(frame) for frame in tracking_results]
ground_truth_counts = [len(frame) for frame in ground_truth]
min_len = min(len(tracking_counts), len(ground_truth_counts))
tracking_counts = tracking_counts[:min_len]
ground_truth_counts = ground_truth_counts[:min_len]

# 绘制散点图
plt.figure(figsize=(8, 6))
plt.scatter(ground_truth_counts, tracking_counts, color='blue', label='Data points')
plt.plot([0, max(ground_truth_counts)], [0, max(ground_truth_counts)], color='red', linestyle='--', label='Ideal line')
plt.title('Scatter plot of ID counts')
plt.xlabel('Ground Truth ID counts')
plt.ylabel('Tracking Results ID counts')
plt.legend()

plt.show()

# 计算R²和RMSE
r2 = r2_score(ground_truth_counts, tracking_counts)
rmse = np.sqrt(mean_squared_error(ground_truth_counts, tracking_counts))

print(f'R² score: {r2:.4f}')
print(f'RMSE: {rmse:.4f}')


# ------------------------- 额外的可视化扩展 -------------------------

# 1. 绘制柱状图 - 比较每帧的ID数量
def plot_bar_chart():
    # 计算差异（误差）
    frame_indices = np.arange(len(tracking_counts))
    errors = np.array(tracking_counts) - np.array(ground_truth_counts)

    # 使用Savitzky-Golay滤波器进行平滑
    smoothed_errors = savgol_filter(errors, window_length=11, polyorder=2)

    plt.figure(figsize=(10, 6))

    # 绘制误差条形图
    plt.bar(frame_indices, errors, color='purple', alpha=0.7, label='Error (Tracking - Ground Truth)')

    # 绘制平滑的误差趋势线
    plt.plot(frame_indices, smoothed_errors, color='red', linewidth=2, label='Smoothed Error Trend')

    plt.title('Difference between Tracking and Ground Truth ID counts')
    plt.xlabel('Frame Index')
    plt.ylabel('Error (Tracking - Ground Truth)')
    plt.legend()

    plt.show()


plot_bar_chart()


# 2. 绘制线性回归拟合线
def plot_linear_regression():
    # 使用线性回归拟合
    X = np.array(ground_truth_counts).reshape(-1, 1)
    y = np.array(tracking_counts)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # 获取线性回归模型的系数和截距
    slope = model.coef_[0]
    intercept = model.intercept_

    # 创建方程的字符串形式
    equation = f'Linear Regression: y = {slope:.4f}x + {intercept:.4f}'

    # 绘制散点图和线性回归拟合线
    plt.figure(figsize=(8, 6))
    plt.scatter(ground_truth_counts, tracking_counts, color='blue', label='Data points')
    plt.plot(ground_truth_counts, y_pred, color='green', linestyle='-', label=f'Linear Regression Line\n({equation})')
    plt.plot([0, max(ground_truth_counts)], [0, max(ground_truth_counts)], color='red', linestyle='--',
             label='Ideal line')
    plt.title('Linear Regression fit on ID counts')
    plt.xlabel('Ground Truth ID counts')
    plt.ylabel('Tracking Results ID counts')
    plt.legend()

    plt.show()
plot_linear_regression()


# 3. 绘制误差图 - 显示ID差异的分布
def plot_error_distribution():
    # 计算误差
    errors = np.array(tracking_counts) - np.array(ground_truth_counts)

    # 绘制误差直方图
    plt.figure(figsize=(8, 6))

    # 绘制直方图
    plt.hist(errors, bins=20, color='orange', edgecolor='black', alpha=0.7, density=True)

    # 使用Seaborn绘制KDE曲线（密度估计曲线）
    sns.kdeplot(errors, color='b', lw=2, fill=True,label='Density Estimation')

    # 添加标题和标签
    plt.title('Error Distribution (Tracking - Ground Truth)')
    plt.xlabel('Error')
    plt.ylabel('Density')
    plt.legend()

    plt.show()

    # ------------------------------- 饼图部分 -------------------------------

    # 定义误差区间，分为四种情况
    error_labels = ['Large Negative', 'Small Negative', 'Zero', 'Positive']
    error_counts = [
        np.sum(np.abs(errors) > 4),  # Large Negative
        np.sum((np.abs(errors) >= 2) & (np.abs(errors) < 4)),  # Small Negative
        np.sum(errors == 0),  # Zero
        np.sum(np.abs(errors) <= 2)  # Positive
    ]

    # 设置不同程度绿色的颜色
    green_shades = ['#004d00', '#007300', '#4cbb17', '#80e0a7']  # 深绿到浅绿

    # 绘制饼图
    plt.figure(figsize=(6, 6))
    plt.pie(error_counts, labels=error_labels, autopct='%1.1f%%', startangle=90, colors=green_shades,
            wedgeprops={'width': 0.5})  # 设置宽度为0.5，产生空心效果

    plt.title('Error Distribution (Tracking - Ground Truth) - Doughnut Chart')
    plt.axis('equal')  # 保证饼图为圆形
    plt.show()


# 调用函数来绘制误差分布图
plot_error_distribution()

# 4. 绘制直方图 - Ground Truth 和 Tracking Counts 的分布
def plot_histogram():
    plt.figure(figsize=(10, 6))
    plt.hist(ground_truth_counts, bins=20, alpha=0.5, label='Ground Truth', color='blue', edgecolor='black')
    plt.hist(tracking_counts, bins=20, alpha=0.5, label='Tracking Results', color='green', edgecolor='black')
    plt.title('Distribution of Ground Truth and Tracking Counts')
    plt.xlabel('ID Counts')
    plt.ylabel('Frequency')
    plt.legend()

    plt.show()


plot_histogram()
