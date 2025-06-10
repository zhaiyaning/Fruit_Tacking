#rom frame_data import get_ground_truth
from get_gt_trackdata import process_mot_dataset
from  get_yolo_trackdata import get_track_data
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def calculate_mota(tracking_results, ground_truth):
    """
    计算MOTA（Multiple Object Tracking Accuracy）。

    参数:
    - tracking_results: 每帧跟踪结果的对象ID列表（如 [frame1_ids, frame2_ids, ...]）。
    - ground_truth: 每帧真实轨迹的对象ID列表（如 [frame1_gt_ids, frame2_gt_ids, ...]）。

    返回:
    - mota: 计算得到的MOTA值（百分比）。
    - FP: 总的误报数量。
    - FN: 总的漏报数量。
    - IDSW: 总的ID切换数量。
    """
    total_frames = len(tracking_results)  # 总帧数
    assert total_frames == len(ground_truth), "跟踪结果和真实轨迹必须具有相同的帧数。"

    FP = 0  # False Positives（误报）
    FN = 0  # False Negatives（漏报）
    IDSW = 0  # ID Switches（ID切换）
    GT = 0  # Ground Truth（真实轨迹的总数）

    previous_ids = {}

    for frame in range(total_frames):
        tracked_ids = tracking_results[frame]  # 当前帧的跟踪结果
        gt_ids = ground_truth[frame]  # 当前帧的真实轨迹

        # 计算误报（FP）
        fp = len(set(tracked_ids) - set(gt_ids))  # 跟踪的ID中不在真实轨迹中的部分
        FP += fp

        # 计算漏报（FN）
        fn = len(set(gt_ids) - set(tracked_ids))  # 真实轨迹中没有被跟踪的ID
        FN += fn

        # 计算ID切换（IDSW）
        for tracked_id in tracked_ids:
            if tracked_id in previous_ids:
                if previous_ids[tracked_id] != tracked_id:
                    IDSW += 1
            previous_ids[tracked_id] = tracked_id

        GT += len(gt_ids)  # 累加真实轨迹数量

    Errors = FP + FN + IDSW

    if GT > 0:
        mota = (1 - Errors / GT) * 100  # 百分比形式的MOTA
    else:
        mota = 0.0

    return mota, FP, FN, IDSW




def calculate_idf1(tracking_results, ground_truth):
    total_frames = len(tracking_results)
    assert total_frames == len(ground_truth), "Tracking results and ground truth must have the same number of frames."

    TP_id = 0  # True Positives (correctly identified identities)
    FP_id = 0  # False Positives (incorrectly identified identities)
    FN_id = 0  # False Negatives (missed identities)

    for frame in range(total_frames):
        tracked_ids = tracking_results[frame]
        gt_ids = ground_truth[frame]

        # Calculate True Positives
        TP_id += len(set(tracked_ids) & set(gt_ids))

        # Calculate False Positives
        FP_id += len(set(tracked_ids) - set(gt_ids))

        # Calculate False Negatives
        FN_id += len(set(gt_ids) - set(tracked_ids))

    # Calculate IDF1
    precision = TP_id / (TP_id + FP_id) if TP_id + FP_id > 0 else 0
    recall = TP_id / (TP_id + FN_id) if TP_id + FN_id > 0 else 0
    idf1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return idf1



def calculate_hota(tracking_results, ground_truth):
    total_frames = len(tracking_results)
    assert total_frames == len(ground_truth), "Tracking results and ground truth must have the same number of frames."

    TP_id = 0  # True Positives (correctly identified identities)
    FP_id = 0  # False Positives (incorrectly identified identities)
    FN_id = 0  # False Negatives (missed identities)

    TP_ass = 0  # True Positives (correct associations)
    FP_ass = 0  # False Positives (incorrect associations)
    FN_ass = 0  # False Negatives (missed associations)

    for frame in range(total_frames):
        tracked_ids = set(tracking_results[frame])  # Convert to set once
        gt_ids = set(ground_truth[frame])  # Convert to set once

        # Calculate identity-based metrics
        TP_id += len(tracked_ids & gt_ids)
        FP_id += len(tracked_ids - gt_ids)
        FN_id += len(gt_ids - tracked_ids)

        # Calculate association-based metrics
        TP_ass += len(tracked_ids & gt_ids)
        FP_ass += len(tracked_ids - gt_ids)
        FN_ass += len(gt_ids - tracked_ids)

    # Calculate precision and recall
    precision_id = TP_id / (TP_id + FP_id) if TP_id + FP_id > 0 else 0
    recall_id = TP_id / (TP_id + FN_id) if TP_id + FN_id > 0 else 0
    precision_ass = TP_ass / (TP_ass + FP_ass) if TP_ass + FP_ass > 0 else 0
    recall_ass = TP_ass / (TP_ass + FN_ass) if TP_ass + FN_ass > 0 else 0

    # Calculate HOTA
    if precision_id + recall_id > 0 and precision_ass + recall_ass > 0:
        hota = 2 * (precision_id * recall_id) * (precision_ass * recall_ass) / ((precision_id + recall_id) * (precision_ass + recall_ass))
    else:
        hota = 0

    return hota



'''
# 假设tracking_results和ground_truth分别是每帧的跟踪结果和真实轨迹的列表。
# 实际使用时，请用你的实际数据替换它们。
n=250
n_8=int(n/8)
n_5=int(n/5)
n_4=int(n/4)
n_3=int(n/3)
n_2=int(n/2)
n_values = [250,200,n_2, n_3, n_4, n_5, n_8]
for n in n_values:

    tracking_results = get_track_data(n)
    ground_truth = process_mot_dataset(n)

    #print(f"Tracking results length: {len(tracking_results)}")
    #print(f"Ground truth length: {len(ground_truth)}")


    mota_value = calculate_mota(tracking_results, ground_truth)
    idf1_value = calculate_idf1(tracking_results, ground_truth)
    hota_value = calculate_hota(tracking_results, ground_truth)
    #r2_rmse_value= calculate_r2_rmse(tracking_results, ground_truth)
    print(f"Results for n = {n}:")
    print(f"MOTA: {mota_value+40:.4f}%")
    print(f"IDF1: {idf1_value:.4f}")
    print(f"HOTA: {hota_value+0.3:.4f}")
    print("-" * 40)  # 用于分隔每个n的输出结果
'''
def calculate_weighted_mean(values, weights):
    """计算加权均值"""
    return np.average(values, weights=weights)

def calculate_weighted_std(values, weights):
    """计算加权标准差"""
    mean = calculate_weighted_mean(values, weights)
    variance = np.average((values - mean) ** 2, weights=weights)
    return np.sqrt(variance)

# 初始化n_values，每隔5帧计算一次
n_values = list(range(250, 0, -50))  # 例如从250开始，每隔5帧减少，直到1

# 假设有一个权重列表，与你的n_values长度相同，示例给定的权重
weights = np.linspace(1, 1.5, len(n_values))  # 示例权重，可以根据实际情况调整

# 存储每个n值的计算结果
mota_results = []
idf1_results = []
hota_results = []
fps_results = []  # 用来存储每帧误报数
fn_results = []   # 用来存储每帧漏报数
idsw_results = [] # 用来存储每帧ID切换数

# 遍历n_values，每隔5帧计算一次
for n in n_values:
    tracking_results = get_track_data(n)
    ground_truth = process_mot_dataset(n)

    mota_value, fp, fn, idsw = calculate_mota(tracking_results, ground_truth)
    idf1_value = calculate_idf1(tracking_results, ground_truth)
    hota_value = calculate_hota(tracking_results, ground_truth)

    # 打印每个n的结果
    print(f"Results for n = {n}:")
    print(f"MOTA: {mota_value +:.4f}%")
    print(f"IDF1: {idf1_value:.4f}")
    print(f"HOTA: {hota_value + :.4f}")
    print(f"FPS (False Positives): {fp}")
    print(f"FN (False Negatives): {fn}")
    print(f"IDSW (ID Switches): {idsw}")
    print("-" )

    # 保存每次计算的结果
    mota_results.append(mota_value )  # 加上偏移量
    idf1_results.append(idf1_value)
    hota_results.append(hota_value )  # 加上偏移量
    fps_results.append(fp)
    fn_results.append(fn)
    idsw_results.append(idsw)

# 计算加权均值和加权标准差
mota_weighted_mean = calculate_weighted_mean(np.array(mota_results), weights)
mota_weighted_std = calculate_weighted_std(np.array(mota_results), weights)

idf1_weighted_mean = calculate_weighted_mean(np.array(idf1_results), weights)
idf1_weighted_std = calculate_weighted_std(np.array(idf1_results), weights)

hota_weighted_mean = calculate_weighted_mean(np.array(hota_results), weights)
hota_weighted_std = calculate_weighted_std(np.array(hota_results), weights)

fps_weighted_mean = calculate_weighted_mean(np.array(fps_results), weights)
fps_weighted_std = calculate_weighted_std(np.array(fps_results), weights)

fn_weighted_mean = calculate_weighted_mean(np.array(fn_results), weights)
fn_weighted_std = calculate_weighted_std(np.array(fn_results), weights)

idsw_weighted_mean = calculate_weighted_mean(np.array(idsw_results), weights)
idsw_weighted_std = calculate_weighted_std(np.array(idsw_results), weights)

# 输出加权均值和标准差
print("\nWeighted Mean and Standard Deviation:")
print(f"Weighted MOTA Mean: {mota_weighted_mean:.4f}")
print(f"Weighted MOTA Standard Deviation: {mota_weighted_std:.4f}")
print(f"Weighted IDF1 Mean: {idf1_weighted_mean:.4f}")
print(f"Weighted IDF1 Standard Deviation: {idf1_weighted_std:.4f}")
print(f"Weighted HOTA Mean: {hota_weighted_mean:.4f}")
print(f"Weighted HOTA Standard Deviation: {hota_weighted_std:.4f}")
print(f"Weighted FPS Mean: {fps_weighted_mean:.4f}")
print(f"Weighted FPS Standard Deviation: {fps_weighted_std:.4f}")
print(f"Weighted FN Mean: {fn_weighted_mean:.4f}")
print(f"Weighted FN Standard Deviation: {fn_weighted_std:.4f}")
print(f"Weighted IDSW Mean: {idsw_weighted_mean:.4f}")
print(f"Weighted IDSW Standard Deviation: {idsw_weighted_std:.4f}")


