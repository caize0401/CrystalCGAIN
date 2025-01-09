from crystal_utils import binary_sampler_coords
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import random

def data_loader(data_folder, data_dis_folder):
    # Load all .npy files in the folder as samples
    data_x_list = []
    for file_name in sorted(os.listdir(data_folder)):
        if file_name.endswith('.npy'):
            sample = np.load(os.path.join(data_folder, file_name))
            data_x_list.append(sample)

    # Stack all samples into a single dataset
    data_x = np.stack(data_x_list, axis=0)  # Shape: (num_samples, features)
    data_dis = data_distance_matrix_loader(data_dis_folder)
    return data_x, data_dis


def data_loader_follow(data_x, miss_number, data_dis):
    # Parameters
    num_samples, n, feature_dim = data_x.shape

    # Initialize list to store the missing data and masks
    miss_data_x_list = []
    data_m_list = []

    # Ensure each sample has the same missing rate
    for i in range(num_samples):
        # For each sample, create a binary mask with the desired missing rate
        data_m = binary_sampler_coords(miss_number, n, feature_dim)  # Shape: (1, n * feature_dim)
        data_m = data_m.reshape(-1)
        data_x_flattened = data_x.reshape(num_samples, -1)  # Shape: (num_samples, n * feature_dim)
        miss_data_x = data_x_flattened[i].copy()
        miss_data_x[data_m == 0] = 0  # Assign missing values to 0
        miss_data_x_list.append(miss_data_x)
        data_m_list.append(data_m)

    # Stack the list of arrays into tensors
    miss_data_x = np.vstack(miss_data_x_list)  # Shape: (num_samples, n * feature_dim)
    data_m = np.vstack(data_m_list)  # Shape: (num_samples, n * feature_dim)

    # Convert to PyTorch tensors
    data_x = torch.tensor(data_x_flattened, dtype=torch.float32)
    miss_data_x = torch.tensor(miss_data_x, dtype=torch.float32)
    data_m = torch.tensor(data_m, dtype=torch.float32)
    data_adj = torch.tensor(data_dis, dtype=torch.float32)

    return data_x, miss_data_x, data_m, data_adj

def train_test_split_data(data_x, miss_data_x, data_m, adj_data, test_size=0.1, random_state=42):
    # Convert PyTorch tensors to numpy for splitting
    data_x_np = data_x.numpy()
    miss_data_x_np = miss_data_x.numpy()
    data_m_np = data_m.numpy()
    adj_data_np = adj_data.reshape(adj_data.shape[0], -1)

    # Use train_test_split to split the data into training and testing sets
    train_x, test_x, train_miss_x, test_miss_x, train_m, test_m, train_adj, test_adj = train_test_split(
        data_x_np, miss_data_x_np, data_m_np, adj_data_np, test_size=test_size, random_state=random_state)

    # Convert back to PyTorch tensors
    train_x = torch.tensor(train_x, dtype=torch.float32)
    test_x = torch.tensor(test_x, dtype=torch.float32)
    train_miss_x = torch.tensor(train_miss_x, dtype=torch.float32)
    test_miss_x = torch.tensor(test_miss_x, dtype=torch.float32)
    train_m = torch.tensor(train_m, dtype=torch.float32)
    test_m = torch.tensor(test_m, dtype=torch.float32)
    train_adj = torch.tensor(train_adj, dtype=torch.float32)
    test_adj = torch.tensor(test_adj, dtype=torch.float32)


    return train_x, test_x, train_miss_x, test_miss_x, train_m, test_m, train_adj, test_adj

def data_distance_matrix_loader(data_folder):
    # Load all .npy files in the folder as samples
    data_dis_list = []
    for file_name in sorted(os.listdir(data_folder)):
        if file_name.endswith('.npy'):
            sample = np.load(os.path.join(data_folder, file_name))
            data_dis_list.append(sample)

    # Stack all samples into a single dataset
    data_dis = np.stack(data_dis_list, axis=0)  # Shape: (num_samples, features)
    data_dis_flattened = data_dis.reshape(data_dis.shape[0], -1)
    return data_dis_flattened


def prepare_training_data_from_folders(folder_paths):
    folder_no = 0
    for folder_path in folder_paths:
        all_data = []
        # 遍历当前文件夹中的所有 .npy 文件
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".npy"):
                file_path = os.path.join(folder_path, file_name)
                # 加载 .npy 文件
                data = np.load(file_path)
                all_data.append(data)

        # 将所有数据堆叠为一个 NumPy 数组
        all_data_array = np.stack(all_data)  # 形状为 (样本数, 27, 3)
        # 转换为 PyTorch 张量
        training_tensor = torch.tensor(all_data_array, dtype=torch.float32)
        folder_no += 1
        if folder_no == 1:
            new_data_all = training_tensor
        if folder_no == 2:
            new_data_x = training_tensor
        if folder_no == 3:
            new_data_adj = training_tensor
        if folder_no == 4:
            new_data_m = training_tensor
        if folder_no == 5:
            new_data_x_z = training_tensor

    return new_data_all, new_data_x, new_data_adj, new_data_m, new_data_x_z

if __name__ == '__main__':
    # 测试正常输入
    '''data_folder = "origin_data/PCOD_npy_sort_folder_Concatenated_npy"
    data_dis_folder = "origin_data/PCOD_npy_sort_folder_distance_matrix_npy"
    # 定义 miss_number 的取值范围
    min_miss = 6  # 范围的最小值
    max_miss = 12  # 范围的最大值
    # 生成范围内的随机整数
    miss_number = random.randint(min_miss, max_miss)
    data_x, data_dis = data_loader(data_folder, data_dis_folder)
    print(data_x.shape)
    print(data_dis.shape)

    data_x, miss_data_x, data_m, data_adj = data_loader_follow(data_x, miss_number, data_dis)
    print(data_adj.shape)'''
    # 测试新结构输入
    folders = [
        "origin_data/new_mse_data/new_concatenated",
        "origin_data/new_mse_data/new_miss",
        "origin_data/new_mse_data/new_distance_martix",
        "origin_data/new_mse_data/new_mask",
        "origin_data/new_mse_data/new_miss_mask_z"
    ]

    new_data_all, new_data_x, new_data_adj, new_data_m, new_data_x_z = prepare_training_data_from_folders(folders)

    print(new_data_all.shape)
    print(new_data_x.shape)
    print(new_data_adj.shape)
    print(new_data_m.shape)
    print(new_data_x_z.shape)






