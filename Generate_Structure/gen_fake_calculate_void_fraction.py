import torch
from torch import nn
import numpy as np
from crystal_utils import uniform_sampler
from zeo_utils import process_batch
import random
from crystal_data_loader import data_loader, data_loader_follow
import matplotlib.pyplot as plt

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

def gen_random(batch_size=100):
    # 生成随机噪声作为输入
    dim = 27 * 3  # 假设每个样本的维度为27x3
    adj_dim = 24 * 24  # 假设 adjacency matrix 的维度为 24x24

    # 模拟一些随机输入
    X_mb = torch.tensor(uniform_sampler(0, 1.0, batch_size, dim), dtype=torch.float32).to(device)
    M_mb = torch.ones_like(X_mb)  # Mask 矩阵，用 1 表示观测值
    adj = torch.tensor(uniform_sampler(0, 1.0, batch_size, adj_dim), dtype=torch.float32).to(device)
    print(X_mb.shape, M_mb.shape, adj.shape)

    # 通过生成器生成结果
    with torch.no_grad():
        generated_data = generator(X_mb, M_mb, adj)

    # 处理生成的数据，生成器的输出假设为[batch_size, (number + 3) * 3]，我们需要 reshape 成[27, 3]格式
    generated_data_reshaped = generated_data.view(-1, 27, 3).cpu().numpy()

    # 计算孔隙率
    predicted_porosity = process_batch(generated_data_reshaped)

    count = 0
    # 打印每个样本的孔隙率
    for i, porosity in enumerate(predicted_porosity):
        '''print(f"Sample {i + 1} Porosity: {porosity:.4f}")'''
        if porosity >= 0.15 and porosity <=0.25:
            count += 1
    print(count)
    return predicted_porosity

def gen_fake(batch_size=100):
    # 读取数据
    data_name = 'origin_data/PCOD_npy_sort_folder_Concatenated_npy'
    adj_name = 'origin_data\PCOD_npy_sort_folder_distance_matrix_npy'
    data_x, data_dis = data_loader(data_name, adj_name)
    # 定义 miss_number 的取值范围
    min_miss = 8  # 范围的最小值
    max_miss = 20  # 范围的最大值
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
    Z_mb = torch.tensor(uniform_sampler(low, high, len(batch_size), dim), dtype=torch.float32).to(device)
    # Combine random vectors with observed data
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

    # 通过生成器生成结果
    with torch.no_grad():
        generated_data = generator(X_mb, M_mb, adj)

    # 处理生成的数据，生成器的输出假设为[batch_size, (number + 3) * 3]，我们需要 reshape 成[27, 3]格式
    generated_data_reshaped = generated_data.view(-1, 27, 3).cpu().numpy()

    # 计算孔隙率
    predicted_porosity = process_batch(generated_data_reshaped)

    count = 0
    # 打印每个样本的孔隙率
    for i, porosity in enumerate(predicted_porosity):
        '''print(f"Sample {i + 1} Porosity: {porosity:.4f}")'''
        if porosity >= 0.15 and porosity <= 0.25:
            count += 1
    print(count)
    return predicted_porosity

# 定义画柱状图的函数
# 定义画柱状图的函数，允许自定义某个区间颜色
def plot_porosity_distribution(porosity_list, range_min=0, range_max=1.0, num_bins=20, highlight_range=None,
                               save_path=None):
    # 创建自定义的区间
    bins = np.linspace(range_min, range_max, num_bins + 1)

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 计算每个区间的频率
    counts, _, patches = ax.hist(porosity_list, bins=bins, color='skyblue', edgecolor='black')

    # 如果设置了需要高亮显示的区间
    if highlight_range:
        low, high = highlight_range
        for count, patch, bin_left, bin_right in zip(counts, patches, bins[:-1], bins[1:]):
            if low <= bin_left <= high or low <= bin_right <= high:
                patch.set_facecolor('red')  # 将这些区间的柱状颜色设置为红色

    # 设置标题和标签
    ax.set_title('Void Fraction Distribution', fontsize=14)
    ax.set_xlabel('Void Fraction', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(True)

    # 如果指定了保存路径，则保存图片
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    # 显示图像
    plt.show()


if __name__ == '__main__':
    batch =1000
    # 生成数据
    '''porosity_data = gen_fake(batch)'''
    porosity_data = gen_random(batch)

    # 绘制孔隙率分布柱状图，定义自定义区间 [0, 100] 并将数据分成 20 组
    # 并将 [0.15, 0.25] 的区间颜色设置为红色，保存图片到指定路径
    plot_porosity_distribution(porosity_data, range_min=0, range_max=1, num_bins=100, highlight_range=[0.15, 0.25],
                               save_path='void_fraction_distribution_random.png')


