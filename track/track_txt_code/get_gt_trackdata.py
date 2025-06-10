import pandas as pd


def process_mot_dataset(n):
    csv_file = r'E:\Astudy\mycode\test_track\144.csv'  # 替换为你的MOT数据集CSV文件路径
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
    #print(frame_ids)
        # print(len(all_frame_ids))

    # 将字典转换为列表，按帧顺序存放
    all_frame_ids = [frame_ids[frame] for frame in sorted(frame_ids.keys())]
    # 只提取前50帧的结果
    all_frame_ids = all_frame_ids[:n]

    #print(all_frame_ids)
    return all_frame_ids

