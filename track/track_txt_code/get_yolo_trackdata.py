import re

def get_track_data(n,input_path):
    # 读取存储帧数据的txt文件
    #file_path = r'E:\Astudy\mycode\test_track\all_trackresult\track_txt\dynamatic_KF.txt'
    file_path=input_path
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 初始化一个空列表来存储最终结果
    tracking_results = []

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


            # 添加到tracking_results中
            tracking_results.append(detections)

            # 增加帧数计数器
            frame_count += 1
            # 如果处理的帧数达到50帧，停止处理
            if frame_count == n:
                break
    # 打印tracking_results列表，检查结果
    #print(tracking_results)
    #print(len(tracking_results))
    return tracking_results
    

