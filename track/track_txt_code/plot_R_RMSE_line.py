import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from get_gt_trackdata import process_mot_dataset
from  get_yolo_trackdata import get_track_data

# Assuming tracking_results and ground_truth are already defined and populated correctly
n=250
tracking_results = get_track_data(n)
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
plt.grid(True)
plt.show()


# 计算R²和RMSE
r2 = r2_score(ground_truth_counts, tracking_counts)
rmse = np.sqrt(mean_squared_error(ground_truth_counts, tracking_counts))

print(f'R² score: {r2:.4f}')
print(f'RMSE: {rmse:.4f}')
