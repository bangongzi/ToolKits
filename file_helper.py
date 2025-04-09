import json
import os
import numpy as np
import yaml
import zipfile


def compress_to_zip(files, zip_path):
    """将给定的文件列表压缩到ZIP文件中"""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            file_name = os.path.basename(file)
            zipf.write(file, arcname=file_name)


def distribute_files_to_zip(input_dir, output_dir, limit):
    """遍历目录下的文件，分堆并压缩
    
    Args:
        limit: 分堆文件大小限制，单位为字节（Byte）。如limit = 1024 * 1024 * 1024，每一堆文件压缩前存储占用不超过1 GB。
    """
    # 获取目录中的所有文件
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    # 计算每个文件的存储空间占用
    file_sizes = [(f, os.path.getsize(f)) for f in files]

    # 分堆
    heap = []
    current_size = 0
    heap_count = 0
    for f, size in file_sizes:
        if current_size + size <= limit:
            heap.append(f)
            current_size += size
        else:
            # 创建新的堆并压缩
            zip_path = os.path.join(output_dir, f'{os.path.basename(input_dir)}_{heap_count}.zip')
            print(f"zipping file heap {heap_count}, files range from: {heap[0]} to {heap[-1]}")
            compress_to_zip(heap, zip_path)
            heap = [f]
            current_size = size
            heap_count += 1

    # 处理剩余的文件
    if heap:
        zip_path = os.path.join(output_dir, f'{os.path.basename(input_dir)}_{heap_count}.zip')
        print(f"zipping file heap {heap_count}, files range from: {heap[0]} to {heap[-1]}")
        compress_to_zip(heap, zip_path)
        heap_count += 1
    print(f"Successfully zip {input_dir} to {heap_count} zips")


def find_all_target_file(input_dir, endmark_list):
    res = list()
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            for endmark in endmark_list:
                if file.endswith(endmark):
                    res.append((root, file))
                    break
    return res

def save_to_json_file(json_path, data):
  class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
      if isinstance(obj, np.ndarray):
        return obj.tolist()
      return json.JSONEncoder.default(self, obj)

  with open(json_path, 'w') as f:
    f.write(json.dumps(data, indent=4, cls=NumpyEncoder))


def load_json_file(json_path):
    with open(json_path, 'r') as f:
        content = f.read()
        data = json.loads(content)
    return data


def load_yaml_file(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as file:
        loaded_data = yaml.safe_load(file)
    return loaded_data


def makedir_if_not_exist(input_dir):
    """注意，该工具函数不建议使用。该函数功能可用os.makedirs(input_dir, exist_ok=True)代替"""
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
