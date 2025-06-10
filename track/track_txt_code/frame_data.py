import cv2
import os
import json
def get_ground_truth():
    # JSON 文件路径
    json_file = r'E:\Astudy\mycode\Synthetic-apples-1\ground_truth.json'
    # 读取 JSON 文件
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 字典用于存储每个frameNr对应的检测结果
    frame_data = {}
    # 假设data是你的检测数据列表
    for detection in data:
        frameNr = detection.get('frameNr')
        object_id = detection.get('object_id')
        # 如果frameNr不在字典中，初始化一个空列表
        if frameNr not in frame_data:
            frame_data[frameNr] = []
        # 将当前检测结果添加到对应frameNr的列表中
        frame_data[frameNr].append((object_id))
        #print(frame_data[frameNr])
    ground_truth = [list(detections) for frameNr, detections in frame_data.items()]
    # 打印ground_truth列表，检查结果
    #print(ground_truth)
    return  ground_truth
