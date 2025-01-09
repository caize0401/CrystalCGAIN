import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from crystal_utils import normalization, renormalization, rounding
from crystal_utils import binary_sampler, uniform_sampler, binary_sampler_coords
from crystal_utils import xavier_init
from crystal_utils import rmse_loss, rmse_obse_loss
from torch.optim.lr_scheduler import StepLR

class Generator(nn.Module):
    def __init__(self, number):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(number * number + (number + 3) * 3 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512 + number * number, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, (number + 3) * 3)
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
        x = torch.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
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

def reconstruction(train_x, test_x, train_miss_x, test_miss_x, train_m, test_m, train_adj, test_adj, gain_parameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 打开一个文件保存训练结果
    with open('reconstruction.txt', 'w') as f:
        # System parameters
        batch_size = gain_parameters['batch_size']
        hint_rate = gain_parameters['hint_rate']
        alpha = gain_parameters['alpha']
        epochs = gain_parameters['epochs']

        # Other parameters
        no, dim = train_miss_x.shape
        no_test, dim_test = test_miss_x.shape
        h_dim = dim

        # Initialize generator and discriminator
        generator = Generator(number=24).to(device)
        discriminator = Discriminator(input_dim=dim * 2, h_dim=h_dim).to(device)

        # Optimizers
        G_optimizer = optim.Adam(generator.parameters(), lr=0.0001)
        D_optimizer = optim.Adam(discriminator.parameters())

        MSE = nn.MSELoss()

        # Training loop by epoch
        for epoch in range(epochs):
            generator.train()

            # Shuffle indices once at the start of each epoch
            total_idx = np.random.permutation(no)

            # z parameter
            low, high = 0, 0.001
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

                X_mb = X_mb.to(device)
                M_mb = M_mb.float().to(device)
                adj = adj.to(device)
                X = X.to(device)

                # Combine random vectors with observed data
                X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

                # Discriminator step
                G_optimizer.zero_grad()
                G_sample = generator(X_mb, M_mb, adj).to(device)
                G_loss = MSE(G_sample, X)
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

                # test
                Z_mb = torch.tensor(uniform_sampler(low, high, no_test, dim_test), dtype=torch.float32).to(device)
                X_mb = test_miss_x
                M_mb = test_m
                X_mb = X_mb.to(device)
                M_mb = M_mb.float().to(device)
                adj = test_adj.to(device)

                X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
                imputed_data = generator(X_mb, M_mb, adj).detach()

                # rmse miss data
                test_rmse_miss_loss = rmse_loss(test_x, imputed_data.detach(), X_mb)
                # rmse obse data
                test_rmse_ob_loss = rmse_obse_loss(test_x, imputed_data.detach(), X_mb)

                # 打印并保存结果到文件
                result = (f"Epoch {epoch + 1}/{epochs},"
                          f" Train miss rmse:{train_rmse_miss_loss:.4f}, obse rmse:{train_rmse_ob_loss:.4f},"
                          f" Test miss rmse:{test_rmse_miss_loss:.4f}, obse rmse:{test_rmse_ob_loss:.4f}\n")
                print(result)
                f.write(result)
    return train_rmse_miss_loss, test_rmse_miss_loss

