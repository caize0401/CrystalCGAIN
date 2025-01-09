import numpy as np
from pymatgen.core import Lattice
import time
import subprocess
import os
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed


# 函数1：处理（27, 3）张量并生成临时的CIF文件
def generate_cif_from_tensor(tensor, filename):
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


# 函数2：使用 zeo++ 计算并打印孔隙率
def calculate_porosity(cif_filename, cygwin_bash, zeo_path, folder_cygwin_path, probe_radius=0.05, chan_radius=0.05):
    cif_path = "/cygdrive/" + folder_cygwin_path + "/" + cif_filename
    zeo_command = f'{zeo_path} -ha -vol {probe_radius} {chan_radius} 5000 {cif_path}'

    error_value = np.nan

    try:
        # 运行 zeo++ 命令以计算孔隙率
        subprocess.run(f'"{cygwin_bash}" --login -c "{zeo_command}"',
                       shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

        # 读取 .vol 文件以获取孔隙率结果
        filename_without_extension = os.path.splitext(cif_filename)[0]
        vol_file = f'E:/GAIN/Porosity_prediction/{filename_without_extension}.vol'

        with open(vol_file, 'r') as file:
            lines = file.read().splitlines()
        try:
            av_line = lines[0]
            av_volume_fraction = av_line.split()[9]
            # 将 av_volume_fraction 转换为浮点数
            av_volume_fraction = float(av_volume_fraction)
            porosity = av_volume_fraction
        except:
            porosity = error_value

    except:
        porosity = error_value


    return porosity


# 批量处理样本
def process_sample(idx, tensor, cygwin_bash, zeo_path, folder_cygwin_path):
    cif_filename = f"g_sample/sample_{idx}.cif"
    # 生成 CIF 文件
    generate_cif_from_tensor(tensor, cif_filename)
    # 计算并打印孔隙率
    porosity = calculate_porosity(cif_filename, cygwin_bash, zeo_path, folder_cygwin_path)

    return porosity


def process_batch(batch_data):
    cygwin_bash = r'D:\Software\Software_Cygwin\bin\bash.exe'
    zeo_path = r'/cygdrive/d/Software/Software_zeo++/zeo++-0.3/network'
    folder_cygwin_path = r'e/GAIN/Porosity_prediction'
    porosities = []

    # 使用多线程并行处理批次中的每个样本
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_sample, idx, tensor, cygwin_bash, zeo_path, folder_cygwin_path)
                   for idx, tensor in enumerate(batch_data)]

        # 收集结果
        for future in as_completed(futures):
            porosity = future.result()  # 获取每个样本的孔隙率
            porosities.append(porosity)
    # 将结果转换为张量，并返回
    porosity_tensor = torch.tensor(porosities)
    return porosity_tensor

"""不保证多线程并行的时候，样本顺序和孔隙率张量保持一致"""
# 测试主函数
if __name__ == '__main__':
    # 记录开始时间
    start_time = time.time()
    # 设置随机种子，保证每次运行的随机结果都相同
    np.random.seed(42)
    # 输入的批量（64, 27, 3）张量样例
    batch_data = np.random.rand(2, 27, 3)  # 示例数据，可替换为你的实际数据

    # 设置 Cygwin Bash 和 zeo++ 路径
    '''放入process_batch函数中了'''

    # 处理批次并获取孔隙率张量
    porosity_tensor = process_batch(batch_data)
    print(porosity_tensor.shape)
    # 打印孔隙率张量
    print("孔隙率张量: \n", porosity_tensor)

    # 记录结束时间
    end_time = time.time()

    # 打印运行时间
    print("程序运行时间: {:.6f} 秒".format(end_time - start_time))
