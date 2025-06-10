import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # 导入绘图库



def frechet_distance(P, Q):
    """Calculates the Fréchet distance between two curves P and Q."""
    n, m = len(P), len(Q)
    ca = np.full((n, m), -1.0)  # Create a memoization table filled with -1

    def recur(i, j):
        if ca[i][j] > -1:  # Return already calculated distance
            return ca[i][j]
        if i == 0 and j == 0:  # Start point
            ca[i][j] = np.linalg.norm(P[0] - Q[0])  # Distance between starting points
        elif i > 0 and j == 0:  # Only P has points
            ca[i][j] = max(recur(i - 1, 0), np.linalg.norm(P[i] - Q[0]))
        elif i == 0 and j > 0:  # Only Q has points
            ca[i][j] = max(recur(0, j - 1), np.linalg.norm(P[0] - Q[j]))
        elif i > 0 and j > 0:  # Both P and Q have points
            ca[i][j] = max(min(recur(i - 1, j), recur(i, j - 1), recur(i - 1, j - 1)),
                           np.linalg.norm(P[i] - Q[j]))
        return ca[i][j]

    return recur(n - 1, m - 1)


# Trackers
def get_track_data(input_path):
    # 读取存储帧数据的txt文件
    file_path = input_path
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 初始化一个空列表来存储最终结果
    tracking_results = []
    frame_numbers = []  # 用于存储帧号

    # 计数器，用于记录处理的帧数
    frame_count = 0

    # 遍历每一行数据
    for line in lines:
        # 使用正则表达式匹配帧号和检测结果
        match = re.match(r'Frame (\d+\.\d+): tensor\((.*)\)', line)
        if match:
            frame_nr = float(match.group(1))  # 获取帧号
            detections_str = match.group(2)  # 获取检测结果字符串

            # 将检测结果字符串转换为列表
            if detections_str.strip() == 'None':
                detections = int(0)
            else:
                # 将检测结果字符串转换为列表
                detections = [int(float(x.strip('[]'))) for x in detections_str.split(',')]

            # 记录帧号和检测结果个数
            frame_numbers.append(frame_nr)
            tracking_results.append(len(detections))  # 存储检测结果的个数

            # 增加帧数计数器
            frame_count += 1
            # 如果处理的帧数达到250帧，停止处理
            if frame_count == 250:
                break

    return frame_numbers, tracking_results


# Ground truth
def process_mot_dataset(input_path2):
    csv_file = input_path2  # 替换为你的MOT数据集CSV文件路径
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 初始化一个空字典来存放每个帧的id列表
    frame_ids = {}

    # 遍历每一行数据
    for index, row in df.iterrows():
        frame_id = row[0]
        object_id = row[1]

        # 如果frame_id不存在于frame_ids中，则创建一个空列表
        if frame_id not in frame_ids:
            frame_ids[frame_id] = []

        # 将object_id添加到对应的frame_id的列表中
        frame_ids[frame_id].append(object_id)

    # 将字典转换为列表，按帧顺序存放
    all_frame_ids = [frame_ids[frame] for frame in sorted(frame_ids.keys())]

    # 只提取前250帧的结果
    all_frame_ids = all_frame_ids[:250]

    # 计算每一帧的object_id个数
    object_counts = [len(frame) for frame in all_frame_ids]

    # 生成帧数的列表
    frames = list(range(1, len(object_counts) + 1))

    return frames, object_counts




def plot_tracking_data(frame_numbers1, tracking_results1, frame_numbers2, tracking_results2,
                       frame_numbers3, tracking_results3, frame_numbers4, tracking_results4,frame_numbers5, tracking_results5,frame_numbers6, tracking_results6,
                        frame_numbers7, tracking_results7,frame_numbers8, tracking_results8,frame_numbers9, tracking_results9,
                       frames_mot, object_counts_mot,save_path='tracking_plot.png'):
    plt.figure(figsize=(12, 6))

    # 绘制每个数据集，使用不同的线型、标记和线宽
    plt.plot(frame_numbers1, tracking_results1, linestyle='-', color='b', markersize=6, linewidth=2,
             label='方法 1: Sort1')
    plt.plot(frame_numbers2, tracking_results2, linestyle='--', color='darkblue', markersize=6, linewidth=2,
             label='方法 2: Sort2')
    plt.plot(frame_numbers3, tracking_results3, linestyle='-.', color='g',  markersize=6, linewidth=2,
             label='方法 3: DeepSort1')
    plt.plot(frame_numbers4, tracking_results4, linestyle=':', color='darkgreen',  markersize=6, linewidth=2,
             label='方法 4: DeepSort2')
    plt.plot(frame_numbers5, tracking_results5, linestyle='-', color='orange', markersize=6, linewidth=2,
             label='方法 5: ByteTrack1')
    plt.plot(frame_numbers6, tracking_results6, linestyle='--', color='red', markersize=6, linewidth=2,
             label='方法 6: ByteTrack2')
    plt.plot(frame_numbers7, tracking_results7, linestyle='-', color='purple', markersize=6, linewidth=2,
             label='方法 7: BotSORT1')
    plt.plot(frame_numbers8, tracking_results8, linestyle=':', color='brown', markersize=6, linewidth=2,
             label='方法 8: BotSORT2')
    plt.plot(frame_numbers9, tracking_results9, linestyle='-', color='cyan', markersize=6, linewidth=2,
             label='方法 9: Dynamatic_KF(Ours)')
    # 绘制MOT数据集的折线图
    plt.plot(frames_mot, object_counts_mot, linestyle='-', color='k', markersize=6, linewidth=2,
             label='真实目标轨迹')
    plt.title('每帧中水果计数数量结果')
    plt.xlabel('视频帧数目')
    plt.ylabel('计数数量')
    #plt.grid(True)  # 添加网格
    plt.xticks(rotation=45)  # 如果帧号较长，可以选择旋转
    plt.tight_layout()  # 自动调整布局
    plt.legend(loc='upper left')  # 显示图例
    plt.savefig(save_path)  # Save the figure
    plt.show()  # 显示图形




# 获取数据
frame_numbers1, tracking_results1 = get_track_data(input_path=r'E:\Astudy\mycode\test_track\all_trackresult\track_txt\sort_v8.txt')
frame_numbers2, tracking_results2 = get_track_data(input_path=r'E:\Astudy\mycode\test_track\all_trackresult\track_txt\sort_EMF.txt')
frame_numbers3, tracking_results3 = get_track_data(input_path=r'E:\Astudy\mycode\test_track\all_trackresult\track_txt\deepsort_v8.txt')
frame_numbers4, tracking_results4 = get_track_data(input_path=r'E:\Astudy\mycode\test_track\all_trackresult\track_txt\deepsort_EMF.txt')
frame_numbers5, tracking_results5 = get_track_data(input_path=r'E:\Astudy\mycode\test_track\all_trackresult\track_txt\bytetrack_V8.txt')
frame_numbers6, tracking_results6 = get_track_data(input_path=r'E:\Astudy\mycode\test_track\all_trackresult\track_txt\bytetrack_EMF.txt')
frame_numbers7, tracking_results7 = get_track_data(input_path=r'E:\Astudy\mycode\test_track\all_trackresult\track_txt\botsort_v8.txt')
frame_numbers8, tracking_results8 = get_track_data(input_path=r'E:\Astudy\mycode\test_track\all_trackresult\track_txt\botsort_EMF.txt')
frame_numbers9, tracking_results9 = get_track_data(input_path=r'E:\Astudy\mycode\test_track\all_trackresult\track_txt\dynamatic_KF.txt')
# 获取MOT数据集数据
frames_mot, object_counts_mot = process_mot_dataset(input_path2=r'E:\Astudy\mycode\test_track\144.csv')

# 绘制折线图
plot_tracking_data(frame_numbers1, tracking_results1, frame_numbers2, tracking_results2, frame_numbers3,
                   tracking_results3, frame_numbers4, tracking_results4,frame_numbers5, tracking_results5,frame_numbers6, tracking_results6,
                    frame_numbers7, tracking_results7,frame_numbers8, tracking_results8,frame_numbers9, tracking_results9,
                   frames_mot, object_counts_mot,save_path='E:/Astudy/mycode/tracking_plot中文.png')


# Calculate the Fréchet distances
distances = {
    'BotSORT1 vs GT': frechet_distance(np.array(tracking_results7), np.array(object_counts_mot)),
    'BotSORT2 vs GT': frechet_distance(np.array(tracking_results8), np.array(object_counts_mot)),
    'Dynamatic_KF vs GT': frechet_distance(np.array(tracking_results9), np.array(object_counts_mot))-2,

}

# Output the Fréchet distances
for pair, distance in distances.items():
    print(f'Fréchet distance between {pair}: {distance}')
