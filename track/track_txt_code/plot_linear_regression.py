import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from get_gt_trackdata import process_mot_dataset
from get_yolo_trackdata import get_track_data

# 假设ground_truth已经正确定义和填充
n = 250
tracking_results_1 = get_track_data(n, r"E:\Astudy\mycode\test_track\all_trackresult\track_txt\botsort_v8.txt")
tracking_results_2 = get_track_data(n, r"E:\Astudy\mycode\test_track\all_trackresult\track_txt\botsort_EMF.txt")
tracking_results_3 = get_track_data(n, r"E:\Astudy\mycode\test_track\all_trackresult\track_txt\dynamatic_KF.txt")
ground_truth = process_mot_dataset(n)

# 计算每个列表中的ID数量
tracking_counts_1 = [len(frame) for frame in tracking_results_1]
tracking_counts_2 = [len(frame) for frame in tracking_results_2]
tracking_counts_3 = [len(frame) for frame in tracking_results_3]
ground_truth_counts = [len(frame) for frame in ground_truth]

# 确保所有列表的长度一致
min_len = min(len(tracking_counts_1), len(tracking_counts_2), len(tracking_counts_3), len(ground_truth_counts))
tracking_counts_1 = tracking_counts_1[:min_len]
tracking_counts_2 = tracking_counts_2[:min_len]
tracking_counts_3 = tracking_counts_3[:min_len]
ground_truth_counts = ground_truth_counts[:min_len]


def plot_linear_regression():
    # 使用线性回归拟合
    def fit_and_plot(X, y, label, color, linestyle='-', show_equation=True):
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # 获取线性回归模型的系数和截距
        slope = model.coef_[0]
        intercept = model.intercept_

        # 创建方程的字符串形式
        equation = f'{label}: y = {slope:.4f}x + {intercept:.4f}'

        # 绘制拟合线
        plt.plot(X, y_pred, color=color, linestyle=linestyle, label=f'{equation}')
        return y_pred

    # 重新定义 X 和 y
    X = np.array(ground_truth_counts).reshape(-1, 1)

    # 创建图形
    plt.figure(figsize=(10, 8))

    # 绘制第一个tracking_results
    fit_and_plot(X, np.array(tracking_counts_1), 'Tracking 1', color='green', linestyle='-')
    plt.scatter(ground_truth_counts, tracking_counts_1, color='green', label='Data points', s=15, cmap='summer')

    # 绘制第二个tracking_results
    fit_and_plot(X, np.array(tracking_counts_2), 'Tracking 2', color='y', linestyle='--')
    plt.scatter(ground_truth_counts, tracking_counts_2, color='y', label='Data points', s=15, cmap='summer')

    # 绘制第三个tracking_results
    fit_and_plot(X, np.array(tracking_counts_3), 'Tracking 3', color='red', linestyle=':')
    plt.scatter(ground_truth_counts, tracking_counts_3, color='red', label='Data points', s=15, cmap='summer')

    # 绘制理想线
    plt.plot([0, max(ground_truth_counts)], [0, max(ground_truth_counts)], color='black', linestyle='--', label='Ideal line')

    # 添加标题和标签
    plt.title('Linear Regression fit on ID counts (Multiple Tracking Results)')
    plt.xlabel('Ground Truth ID counts')
    plt.ylabel('Tracking Results ID counts')
    plt.legend()

    # 显示图形
    plt.show()

# 调用绘图函数
plot_linear_regression()
