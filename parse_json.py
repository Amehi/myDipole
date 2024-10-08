import os
import json
import pandas as pd

def find_json_files(root_folder):
    """递归查找根目录下的所有JSON文件"""
    json_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('4.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def read_json_data(json_files):
    """读取所有JSON文件的数据并汇总成列表"""
    data_list = []
    keys = set()  # 用于存储所有文件中的key
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data_list.append(data)
            keys.update(data.keys())  # 收集所有文件中的key
    return data_list, sorted(keys)

def write_to_excel(data_list, keys, output_file):
    """将数据写入Excel文件"""
    rows = []
    for data in data_list:
        row = [data.get(key, None) for key in keys]  # 按照keys的顺序取出value
        rows.append(row)
    
    # 创建DataFrame并写入Excel
    df = pd.DataFrame(rows, columns=keys)
    df.to_excel(output_file, index=False)

def main():
    # 定义你的根目录和输出文件
    root_folder = 'result'  # 替换为你的文件夹路径
    output_file = 'output_dipole.xlsx'  # 结果保存的Excel文件

    # 查找所有的JSON文件
    json_files = find_json_files(root_folder)
    
    if not json_files:
        print("未找到任何JSON文件")
        return

    # 读取所有JSON文件的数据
    data_list, keys = read_json_data(json_files)

    # 将数据写入Excel
    write_to_excel(data_list, keys, output_file)
    print(f"数据已成功保存到 {output_file}")

if __name__ == "__main__":
    main()