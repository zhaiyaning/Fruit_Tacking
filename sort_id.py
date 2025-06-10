import ast
import re

output_file = r"E:\Astudy\yolov8\ultralytics-main\runs\track\2024.12.2\2.txt"

# 读取文件中的数据
with open(output_file, 'r') as file:
    lines = file.readlines()

# 初始化一个空的列表来存储所有的数值
all_numbers = []

# 遍历文件的每一行
for line in lines:
    # 检查每一行是否包含 tensor 的形式
    match = re.search(r"tensor\((\[[^\]]+\])\)", line)
    if match:
        # 提取 tensor 中的数字部分
        numbers_str = match.group(1)  # 提取括号中的内容
        numbers = ast.literal_eval(numbers_str)  # 使用 literal_eval 转换为列表
        all_numbers.extend(numbers)

# 去重并排序
unique_sorted_numbers = sorted(set(all_numbers))

# 输出排序后的结果
print("Sorted and Unique Numbers:", unique_sorted_numbers)
print(len(unique_sorted_numbers))

# 可选：将结果保存回文件
with open('sorted_unique_data.txt', 'w') as output_file:
    output_file.write("Sorted and Unique Numbers: " + str(unique_sorted_numbers))
