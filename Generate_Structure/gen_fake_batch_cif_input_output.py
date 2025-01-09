import os

import torch
from torch import nn
import numpy as np
from crystal_utils import uniform_sampler
import random
from crystal_data_loader import data_loader, data_loader_follow
from pymatgen.core import Lattice

# 假设 Generator 模型的定义和之前相同
class Generator(nn.Module):
    def __init__(self, number):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(number * number + (number + 3) * 3 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512 + number * number, 256)
        self.fc4 = nn.Linear(256, (number + 3) * 3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_mb, M_mb, adj):
        x = torch.cat([X_mb, M_mb, adj], dim=-1)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.cat([x, adj], dim=-1)
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(number=24).to(device)

# 请确保这里的路径是正确的，指向你保存的 generator.pth 文件
generator.load_state_dict(torch.load('gan_model_step_2/generator_5500.pth'))
generator.eval()  # 设置为评估模式

def gen_fake(batch_size=1501, cycle=71):
    # 读取数据
    data_name = 'origin_data/PCOD_npy_sort_folder_Concatenated_npy'
    adj_name = 'origin_data\PCOD_npy_sort_folder_distance_matrix_npy'
    data_x, data_dis = data_loader(data_name, adj_name)
    # 定义 miss_number 的取值范围
    min_miss = 6  # 范围的最小值
    max_miss = 20  # 范围的最大值

    input_folder_path = 'fake_data/gen_fake_inputs'
    output_folder_path = 'fake_data/gen_fake_outputs'

    # 创建文件夹
    os.makedirs(input_folder_path, exist_ok=True)
    os.makedirs(output_folder_path, exist_ok=True)

    for i in range(cycle):
        # 生成范围内的随机整数
        miss_number = random.randint(min_miss, max_miss)

        data_x_all, miss_data_x, data_m, data_adj = data_loader_follow(data_x, miss_number, data_dis)

        # Sample batch
        X_mb = miss_data_x[:batch_size, :].to(device)
        M_mb = data_m[:batch_size, :].to(device)
        adj = data_adj[:batch_size, :].to(device)
        print(X_mb.shape, M_mb.shape, adj.shape)

        # z parameter
        low, high = 0, 0.001
        data_x_copy = data_x.reshape(data_x.shape[0], -1)
        no, dim = data_x_copy.shape
        # Sample random vectors and hint vectors
        Z_mb = torch.tensor(uniform_sampler(low, high, batch_size, dim), dtype=torch.float32).to(device)
        # Combine random vectors with observed data
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        # 通过生成器生成结果
        with torch.no_grad():
            generated_data = generator(X_mb, M_mb, adj)

        # 处理生成的数据，生成器的输出假设为[batch_size, (number + 3) * 3]，我们需要 reshape 成[27, 3]格式
        generated_data_reshaped = generated_data.view(-1, 27, 3).cpu().numpy()

        # 遍历生成的样本并保存
        for idx, tensor in enumerate(generated_data_reshaped):
            sample_idx = idx + (i * batch_size)

            # 保存输入数据
            input_filename = f"{input_folder_path}/sample_{sample_idx}.npy"
            np.save(input_filename, {
                "X_mb": X_mb[idx].cpu().numpy(),
                "M_mb": M_mb[idx].cpu().numpy(),
                "adj": adj[idx].cpu().numpy()
            })

            # 保存生成的 CIF 文件
            cif_filename = f"{output_folder_path}/sample_{sample_idx}.cif"
            process_sample(tensor, cif_filename)

    return generated_data_reshaped  # 如果需要返回生成的数据，可以在这里返回

def process_sample(tensor, filename):
    lattice_matrix = tensor[0:3, :]  # 获取前 3 行作为晶格参数矩阵
    lattice_matrix = lattice_matrix * 64.0 - 32.0  # 还原晶格参数矩阵
    lattice = Lattice(lattice_matrix)

    # 提取晶格参数
    a, b, c = lattice.abc  # 边长
    alpha, beta, gamma = lattice.angles  # 角度
    lattice_params = np.array([a, b, c, alpha, beta, gamma])

    # 获取 Si 和 O 的原子分数坐标
    si_coords = tensor[3:11, :]  # Si 原子坐标
    o_coords = tensor[11:27, :]  # O 原子坐标

    # 写入CIF文件
    with open(filename, 'w') as f:
        f.write("data_generated\n")
        f.write("_cell_length_a {:.6f}\n".format(lattice_params[0]))
        f.write("_cell_length_b {:.6f}\n".format(lattice_params[1]))
        f.write("_cell_length_c {:.6f}\n".format(lattice_params[2]))
        f.write("_cell_angle_alpha {:.6f}\n".format(lattice_params[3]))
        f.write("_cell_angle_beta {:.6f}\n".format(lattice_params[4]))
        f.write("_cell_angle_gamma {:.6f}\n".format(lattice_params[5]))

        f.write("loop_\n")
        f.write("_atom_site_label\n")
        f.write("_atom_site_type_symbol\n")
        f.write("_atom_site_fract_x\n")
        f.write("_atom_site_fract_y\n")
        f.write("_atom_site_fract_z\n")

        for i, coord in enumerate(si_coords):
            f.write("Si{} Si {:.6f} {:.6f} {:.6f}\n".format(i + 1, coord[0], coord[1], coord[2]))

        for i, coord in enumerate(o_coords):
            f.write("O{} O {:.6f} {:.6f} {:.6f}\n".format(i + 1, coord[0], coord[1], coord[2]))

if __name__ == "__main__":
    generated_data_reshaped = gen_fake()
