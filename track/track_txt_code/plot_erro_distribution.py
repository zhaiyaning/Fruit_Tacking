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
# plt.rcParams['font.family'] = ['SimHei']  # Windows 下可使用 SimHei 或 Microsoft YaHei
# 假设tracking_results和ground_truth已经正确定义和填充
n = 250
tracking_results = get_track_data(n, r"E:\Astudy\mycode\test_track\all_trackresult\track_txt\dynamatic_KF.txt")
ground_truth = process_mot_dataset(n)

# 计算每个列表中的ID数量
tracking_counts = [len(frame) for frame in tracking_results]
ground_truth_counts = [len(frame) for frame in ground_truth]
min_len = min(len(tracking_counts), len(ground_truth_counts))
tracking_counts = tracking_counts[:min_len]
ground_truth_counts = ground_truth_counts[:min_len]

# 3. 绘制误差图 - 显示ID差异的分布
def plot_error_distribution():
    # 计算误差
    errors = np.array(tracking_counts) - np.array(ground_truth_counts)

    # 绘制误差直方图
    plt.figure(figsize=(8, 6))

    # 绘制直方图
    plt.hist(errors, bins=20, color='#ff6347', edgecolor='black', alpha=0.7, density=True)

    # 使用Seaborn绘制KDE曲线（密度估计曲线）
    sns.kdeplot(errors, color='b', lw=2,  label='Density Estimation')

    # 设置x轴为整数刻度
    plt.xticks(np.arange(int(np.min(errors)), int(np.max(errors)) + 1, 1))  # 设置整数刻度

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
    #green_shades = ['#004d00', '#007300', '#4cbb17', '#80e0a7']  # 深绿到浅绿

    #yellow_shades = ['#b8860b', '#daa520', '#f0e68c', '#ffffe0']  # 黄色渐变：从深黄到浅黄

    #blue_shades = ['#00008b', '#0000cd', '#4682b4', '#87cefa']    # 蓝色渐变：从深蓝到浅蓝

    red_shades = ['#8b0000', '#b22222', '#ff6347', '#ff9999']   # 红色渐变：从深红到浅红



    # 绘制饼图
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        error_counts,
        labels=error_labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=red_shades,
        wedgeprops={'width': 0.5},  # 设置宽度为0.5，产生空心效果
        pctdistance=1.5,  # 设置百分比文本离圆心的距离（0.85表示较远的外部）
        labeldistance=1.1  # 设置标签离圆心的距离（1.1表示较远的外部）
    )

    # 设置autotext文本的颜色和大小
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('black')

    # 设置文本字体
    for text in texts:
        text.set_fontsize(12)
        text.set_color('black')

    plt.title('Error Distribution (Tracking - Ground Truth) - Doughnut Chart')
    plt.axis('equal')  # 保证饼图为圆形

    # 显示图表
    plt.show()

# 调用函数来绘制误差分布图
plot_error_distribution()
