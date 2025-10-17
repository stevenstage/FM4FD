import numpy as np
import scipy.io
import os

def combine_time_series(data_dict):
    """ 合并多个时间序列数据 """
    shapes = [data.shape for data in data_dict.values()]
    unique_shapes = set(shapes)
    if len(unique_shapes) > 1:
        raise ValueError("Not all data arrays have the same shape")
    
    data_arrays = list(data_dict.values())
    combined_data = np.stack(data_arrays, axis=-1)
    
    return combined_data

def load_data_and_labels(mat_file_paths, labels):
    """
    加载多个 .mat 文件的数据并分配标签
    - mat_file_paths: .mat 文件的路径列表
    - labels: 与每个文件对应的标签列表
    """
    if len(mat_file_paths) != len(labels):
        raise ValueError("Number of files must match the number of labels.")
    
    data_list = []
    labels_list = []
    variable_names = [
        'P3504_S101_U_phase_current_instantaneous_value',
        'P3505_S101_V_phase_current_instantaneous_value',
        'P3506_S101_W_phase_current_instantaneous_value',
        'S101_TorqueCurrentPct',
        'S101_FluxCurrentPct',
        'S101_UL3L1Act',
        'S101_UL2L3Act',
        'S101_UL1L2Act',
        'P9009_S101_Motor_Torque',
        'P9015_S101_Output_Frequency'
    ]

    for file_path, label in zip(mat_file_paths, labels):
        mat_data = scipy.io.loadmat(file_path)
        data_dict = {name: mat_data[name] for name in variable_names}
        
        combined_data = combine_time_series(data_dict)
        second_column_data = combined_data[:, 1, :]
        
        data_list.append(second_column_data)
        labels_list.append(label)
    
    return data_list, labels_list  # 返回数据块列表和标量标签列表

# 工具函数：搜索文件夹中的所有 .mat 文件
def search_mat_files(folder_list):
    """
    搜索多个文件夹中的所有 .mat 文件
    - folder_list: 文件夹路径列表
    """
    mat_files = []
    for folder in folder_list:
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.mat'):
                    mat_files.append(os.path.join(root, file))
    return mat_files

# 动态生成标签函数
def extract_label_from_filename(filename):
    """
    根据文件名提取标签
    - filename: 文件名（包含路径）
    """
    basename = os.path.basename(filename)
    parts = basename.split('_')
    last_part = parts[-1].split('.')[0]
    
    fault_level_mapping = {
        '0': 0,
        '0v41': 1,
        '0v83': 2,
        '1v25': 3,
        '1v66': 4
    }
    
    return fault_level_mapping.get(last_part, -1)

# 数据分割函数
def segment(data_list, label_list):
    split_data_list = []
    split_label_list = []

    for index, data in enumerate(data_list):
        step_size = 512  # 窗口大小
        move_size = 30   # 步长

        for i in range(0, data.shape[0] - step_size + 1, move_size):
            split_data_list.append(data[i:i + step_size, :].T)
            split_label_list.append(label_list[index])

    split_data = np.stack(split_data_list, axis=0)
    split_label = np.array(split_label_list)
    
    indices = np.arange(split_data.shape[0])
    np.random.shuffle(indices)
    split_data = split_data[indices]
    split_label = split_label[indices]

    return split_data, split_label

def data_load_d(train_ratio=0.8):
    data_folder_list = [
        '/root/autodl-tmp/f_10',
        '/root/autodl-tmp/f_25',
        '/root/autodl-tmp/f_35',
        '/root/autodl-tmp/f_50'
    ]

    # 替换为你的多数据文件夹路径列表
    mat_file_paths = search_mat_files(data_folder_list)
    print(f"Found {len(mat_file_paths)} .mat files in total.")
    
    # 根据文件名生成标签
    labels = [extract_label_from_filename(file_path) for file_path in mat_file_paths]
    
    # 过滤掉未匹配的文件（如果有的话）
    valid_indices = [i for i, label in enumerate(labels) if label != -1]
    valid_file_paths = [mat_file_paths[i] for i in valid_indices]
    valid_labels = [labels[i] for i in valid_indices]
    
    print(f"Valid files: {len(valid_file_paths)}")
    print("Valid labels:", valid_labels)
    
    if not valid_file_paths:
        print("No valid files found. Exiting.")
        exit()

    # 加载所有数据
    data_list, labels_list = load_data_and_labels(valid_file_paths, valid_labels)
    X, y = segment(data_list, labels_list)
    print("Data shape after segmentation:", X.shape, y.shape)

    # 按train_ratio分割训练集和测试集
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":
    train_set, test_set = data_load_d(train_ratio=0.8)
    print("Train data:", train_set[0].shape, train_set[1].shape)
    print("Test data:", test_set[0].shape, test_set[1].shape)