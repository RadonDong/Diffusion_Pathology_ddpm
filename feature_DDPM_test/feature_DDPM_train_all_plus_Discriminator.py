import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

class SimpleDDPM(nn.Module):
    def __init__(self, input_dim):
        super(SimpleDDPM, self).__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim)
        )

    def forward(self, x, t):
        t_embedding = self._get_timestep_embedding(t, self.input_dim).to(x.device)
        t_embedding = t_embedding.unsqueeze(1)  # 增加维度以匹配 x
        return self.net(x + t_embedding)

    def _get_timestep_embedding(self, t, dim):
        half_dim = dim // 2
        emb = torch.arange(half_dim, dtype=torch.float32, device=t.device)
        emb = torch.exp(-torch.log(torch.tensor(10000.0, device=t.device)) * emb / (half_dim - 1))
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

class EmbeddingDataset(Dataset):
    def __init__(self, folder_path):
        self.filepaths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pt')]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        embedding = torch.load(filepath)
        return embedding

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=1, ndf=64, n_layers=3):
        super(PatchGANDiscriminator, self).__init__()
        model = []
        model += [nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1)]
        model += [nn.LeakyReLU(0.2, inplace=True)]
        
        for i in range(1, n_layers):
            mult = 2**(i-1)
            model += [nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=1)]
            model += [nn.BatchNorm2d(ndf * mult * 2)]
            model += [nn.LeakyReLU(0.2, inplace=True)]
        
        mult = 2**(n_layers-1)
        model += [nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=1)]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# 超参数
batch_size = 128
learning_rate = 1e-4
num_epochs = 50000
timesteps = 1000
base_save_dir = './training'
data_dir = './MIL_diff_LR_features'

def train_model_for_folder(folder_name, subfolder_name):
    train_folder = os.path.join(data_dir, folder_name, subfolder_name)
    save_dir = os.path.join(base_save_dir, folder_name, subfolder_name)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 定义模型、优化器和损失函数
    model = SimpleDDPM(768).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    # 定义PatchGAN辨别器及其优化器和损失函数
    discriminator = PatchGANDiscriminator(input_channels=1).cuda()
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=learning_rate)
    bce_loss = nn.BCEWithLogitsLoss()

    # 读取训练数据
    dataset = EmbeddingDataset(train_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 扩散过程参数
    beta_start = 0.0001
    beta_end = 0.02
    beta = torch.linspace(beta_start, beta_end, timesteps).cuda()
    alpha = 1 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0).cuda()

    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir=save_dir)

    def q_sample(x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0).cuda()
        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod[t]).view(-1, 1, 1)
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

    # 训练循环
    for epoch in range(num_epochs):
        epoch_loss = 0
        for x_0 in dataloader:
            x_0 = x_0.cuda()
            t = torch.randint(0, timesteps, (x_0.size(0),), device=x_0.device)
            noise = torch.randn_like(x_0).cuda()
            x_t = q_sample(x_0, t, noise)

            optimizer.zero_grad()
            optimizer_D.zero_grad()

            predicted_noise = model(x_t, t)
            mse_loss_value = mse_loss(predicted_noise, noise)
            
            # Reshape predicted_noise and noise for PatchGAN
            predicted_noise_reshaped = predicted_noise.view(predicted_noise.size(0), 1, 32, 24)
            noise_reshaped = noise.view(noise.size(0), 1, 32, 24)
            
            # Train PatchGAN Discriminator
            real_labels = torch.ones((noise.size(0), 1, 30, 22)).cuda()
            fake_labels = torch.zeros((noise.size(0), 1, 30, 22)).cuda()
            
            real_loss = bce_loss(discriminator(noise_reshaped), real_labels)
            fake_loss = bce_loss(discriminator(predicted_noise_reshaped.detach()), fake_labels)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # Train DDPM Model with additional PatchGAN loss
            gan_loss = bce_loss(discriminator(predicted_noise_reshaped), real_labels)
            total_loss = mse_loss_value + gan_loss
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar('Loss/train', avg_loss, epoch)

        print(f"Epoch {epoch+1} Loss: {avg_loss}")
        if (epoch+1) % 2000 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'simple_ddpm_epoch_{epoch+1}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(save_dir, f'discriminator_epoch_{epoch+1}.pth'))

    # 关闭 TensorBoard
    writer.close()

# 获取所有子文件夹
main_folders = [f.name for f in os.scandir(data_dir) if f.is_dir()]

# 逐个训练模型
for main_folder in main_folders:
    subfolders = [sf.name for sf in os.scandir(os.path.join(data_dir, main_folder)) if sf.is_dir()]
    for subfolder in subfolders:
        print(f"Training model for folder: {main_folder}/{subfolder}")
        train_model_for_folder(main_folder, subfolder)
        print(f"Finished training for folder: {main_folder}/{subfolder}")