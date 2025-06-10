def read_and_inspect_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            print(f"Inspecting file: {file_path}")
            print("")

            # 逐行读取文件内容
            for line in file:
                # 去除行尾的换行符
                line = line.strip()
                # 分割每行数据
                data_list = line.split()

                # 打印每行数据及其类型
                for data in data_list:
                    data_type = type(eval(data))
                    print(f"Data: {data}, Type: {data_type}")

                print("")  # 换行

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"Error: {e}")

# 请替换为你的文件路径
file_path = r'E:\Astudy\mycode\txt\00b5fefed_jpg.rf.e0000563d76086104f6da9f777bf3b61.jpg.txt'

read_and_inspect_file(file_path)
