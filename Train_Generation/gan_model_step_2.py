import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import MSELoss
from tqdm import tqdm
from crystal_utils import normalization, renormalization, rounding
from crystal_utils import binary_sampler, uniform_sampler, binary_sampler_coords
from crystal_utils import xavier_init
from crystal_utils import rmse_loss, rmse_obse_loss
from zeo_utils import generate_cif_from_tensor, calculate_porosity, process_sample, process_batch
from crystal_data_loader import data_loader, data_loader_follow
import random
import os
from tqdm import tqdm
import time

# 定义根文件目录
root_dir = "model/gan_model_step_2"

# 如果目录不存在，则创建
if not os.path.exists(root_dir):
    os.makedirs(root_dir)


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
        # 将这些输入数据拼接在一起，例如沿着最后一个维度
        x = torch.cat([X_mb, M_mb, adj], dim=-1)
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.cat([x, adj], dim=-1)
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim, h_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, input_dim // 2)  # Output size matches original data
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h):
        inputs = torch.cat([x, h], dim=1)
        d_h1 = self.relu(self.fc1(inputs))
        d_h2 = self.relu(self.fc2(d_h1))
        d_prob = self.sigmoid(self.fc3(d_h2))
        return d_prob

def load_model_if_exists(generator, discriminator, G_optimizer, D_optimizer, epoch_num):
    gen_path = f'{root_dir}/generator_{epoch_num}.pth'
    dis_path = f'{root_dir}/discriminator_{epoch_num}.pth'
    G_opt_path = f'{root_dir}/G_optimizer_{epoch_num}.pth'
    D_opt_path = f'{root_dir}/D_optimizer_{epoch_num}.pth'

    if os.path.exists(gen_path) and os.path.exists(dis_path):
        generator.load_state_dict(torch.load(gen_path))
        discriminator.load_state_dict(torch.load(dis_path))

        if os.path.exists(G_opt_path) and os.path.exists(D_opt_path):
            G_optimizer.load_state_dict(torch.load(G_opt_path))
            D_optimizer.load_state_dict(torch.load(D_opt_path))
        print(f"Successfully loaded model and optimizer from epoch {epoch_num}")
        return True
    else:
        print(f"No pre-trained model found for epoch {epoch_num}")
        return False


def gan_step_2(gain_parameters, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # System parameters
    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    beta = gain_parameters['beta']
    epochs = gain_parameters['epochs']

    # 读取数据
    data_name = args.data_name
    adj_name = args.adj_name
    data_x, data_dis = data_loader(data_name, adj_name)


    # Other parameters
    data_x_copy = data_x.reshape(data_x.shape[0], -1)
    no, dim = data_x_copy.shape
    h_dim = dim

    # Initialize generator and discriminator
    generator = Generator(number=24).to(device)
    discriminator = Discriminator(input_dim=dim * 2, h_dim=h_dim).to(device)

    # Optimizers
    G_optimizer = optim.Adam(generator.parameters())
    D_optimizer = optim.Adam(discriminator.parameters())
    MSE = nn.MSELoss()

    use_pretrained = input("Do you want to load a pre-trained model? (yes/no): ").strip().lower()

    if use_pretrained == 'yes':
        epoch_num = int(input("Enter the epoch number of the model to load: "))
        if load_model_if_exists(generator, discriminator, G_optimizer, D_optimizer, epoch_num):
            print(f"Successfully loaded model and optimizer from epoch {epoch_num}")
            start_epoch = epoch_num  # 从加载的epoch继续训练
        else:
            print("No pre-trained model found. Starting training from scratch...")
            start_epoch = 0  # 从头开始训练
    else:
        start_epoch = 0  # 从头开始训练\

    # Training loop by epoch
    for epoch in tqdm(range(start_epoch, epochs)):
        generator.train()


        # 定义 miss_number 的取值范围
        min_miss = 6  # 范围的最小值
        max_miss = 20  # 范围的最大值
        # 生成范围内的随机整数
        miss_number = random.randint(min_miss, max_miss)

        data_x_all, miss_data_x, data_m, data_adj = data_loader_follow(data_x, miss_number, data_dis)

        # 与下面保持一致
        train_miss_x = miss_data_x
        train_m = data_m
        train_adj = data_adj
        train_x = data_x_all

        # Shuffle indices once at the start of each epoch
        total_idx = np.random.permutation(no)
        # z parameter
        low, high = 0, 0.001
        porosity_loss = 0

        # Split the data into batches
        for i in range(0, no, batch_size):
            batch_idx = total_idx[i:i + batch_size]

            # Sample batch
            X_mb = train_miss_x[batch_idx, :]
            M_mb = train_m[batch_idx, :]
            adj = train_adj[batch_idx, :]
            X = train_x[batch_idx, :]

            # Sample random vectors and hint vectors
            Z_mb = torch.tensor(uniform_sampler(low, high, len(batch_idx), dim), dtype=torch.float32).to(device)
            H_mb_temp = torch.tensor(binary_sampler(hint_rate, len(batch_idx), dim), dtype=torch.float32).to(device)

            X_mb = X_mb.to(device)
            M_mb = M_mb.float().to(device)
            adj = adj.to(device)
            X = X.to(device)

            H_mb = M_mb * H_mb_temp

            # Combine random vectors with observed data
            X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

            # Discriminator step
            D_optimizer.zero_grad()
            G_sample = generator(X_mb, M_mb, adj).detach().to(device)
            Hat_X = X_mb * M_mb + G_sample * (1 - M_mb)
            D_prob = discriminator(Hat_X, H_mb).to(device)
            D_loss = -torch.mean(M_mb * torch.log(D_prob + 1e-8) + (1 - M_mb) * torch.log(1. - D_prob + 1e-8))
            D_loss.backward()
            D_optimizer.step()

            # Generator step
            for _ in range(1):
                G_optimizer.zero_grad()
                G_sample = generator(X_mb, M_mb, adj).to(device)

                '''孔隙率计算'''
                batch_data = G_sample.view(-1, 27, 3).cpu().detach().numpy()
                # 处理批次并获取孔隙率张量
                predicted_porosity = process_batch(batch_data)
                # 过滤无效的 NaN 值
                valid_indices = ~torch.isnan(predicted_porosity)
                # 创建 true_porosity
                fixed_value = 0.20  # 定义你希望每个元素的固定值
                true_porosity = torch.full((G_sample.size(0),), fixed_value)  # 确保 batch_size 对应
                valid_pred = predicted_porosity[valid_indices].to(device)
                valid_true = true_porosity[valid_indices].to(device)
                loss_p = MSE(valid_pred, valid_true)

                # 用于打印
                porosity_loss = loss_p

                '''GAN计算'''
                Hat_X = X_mb * M_mb + G_sample * (1 - M_mb)
                D_prob = discriminator(Hat_X, H_mb).to(device)
                G_loss_temp = -torch.mean((1 - M_mb) * torch.log(D_prob + 1e-8))
                MSE_loss = torch.mean((M_mb.float() * X_mb - M_mb.float() * G_sample) ** 2) / torch.mean(M_mb.float())
                loss_g = G_loss_temp + alpha * MSE_loss

                '''总误差'''
                G_loss = beta * loss_p + loss_g
                G_loss.backward()
                G_optimizer.step()


        # 验证模型
        generator.eval()
        with torch.no_grad():
            # train
            Z_mb = torch.tensor(uniform_sampler(low, high, no, dim), dtype=torch.float32).to(device)
            X_mb = train_miss_x
            M_mb = train_m
            X_mb = X_mb.to(device)
            M_mb = M_mb.float().to(device)
            adj = train_adj.to(device)

            X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
            imputed_data = generator(X_mb, M_mb, adj).detach()

            # rmse miss data
            train_rmse_miss_loss = rmse_loss(train_x, imputed_data.detach(), X_mb)
            # rmse obse data
            train_rmse_ob_loss = rmse_obse_loss(train_x, imputed_data.detach(), X_mb)

        # 打开文件，并将数据追加到文件中
        with open(f'{root_dir}/training_log.txt', 'a') as f:
            # 打印到控制台并保存到文件
            print(f"Epoch {epoch + 1}/{epochs},"
                  f" Train miss rmse:{train_rmse_miss_loss:.4f}, obse rmse:{train_rmse_ob_loss:.4f}, porosity:{porosity_loss:.4f}"
                  f" G loss:{G_loss:.4f}, d loss:{D_loss:.4f}, g loss:{loss_g:.4f}, g loss temp:{G_loss_temp:.4f}, MSE loss:{MSE_loss:.4f}")

            # 保存到文件中
            f.write(f"Epoch {epoch + 1}/{epochs},"
                    f" Train miss rmse:{train_rmse_miss_loss:.4f}, obse rmse:{train_rmse_ob_loss:.4f}, porosity:{porosity_loss:.4f}"
                  f" G loss:{G_loss:.4f}, d loss:{D_loss:.4f}, g loss:{loss_g:.4f}, g loss temp:{G_loss_temp:.4f}, MSE loss:{MSE_loss:.4f}\n")

    # Save models after training
    torch.save(generator.state_dict(), f'{root_dir}/generator_5500.pth')
    torch.save(discriminator.state_dict(), f'{root_dir}/discriminator_5500.pth')

    # Save optimizers
    torch.save(G_optimizer.state_dict(), f'{root_dir}/G_optimizer_5500.pth')
    torch.save(D_optimizer.state_dict(), f'{root_dir}/D_optimizer_5500.pth')

    return train_rmse_miss_loss

