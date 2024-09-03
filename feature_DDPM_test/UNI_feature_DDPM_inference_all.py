import torch
import torch.nn as nn
import os

class SimpleDDPM(nn.Module):
    def __init__(self, input_dim):
        super(SimpleDDPM, self).__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, input_dim)
        )

    def forward(self, x, t):
        t_embedding = self._get_timestep_embedding(t, self.input_dim).to(x.device)
        t_embedding = t_embedding.unsqueeze(1)  # 增加这一行
        return self.net(x + t_embedding)

    def _get_timestep_embedding(self, t, dim):
        half_dim = dim // 2
        emb = torch.arange(half_dim, dtype=torch.float32, device=t.device)
        emb = torch.exp(-torch.log(torch.tensor(10000.0, device=t.device)) * emb / (half_dim - 1))
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

class EmbeddingDataset:
    def __init__(self, folder_path):
        self.filepaths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pt')]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        embedding = torch.load(filepath)
        return embedding

# 超参数
timesteps = 1000
data_dir = './MIL_diff_LR_features'
training_dir = './training'
inference_base_dir = './inference'

# 创建推理输出目录
if not os.path.exists(inference_base_dir):
    os.makedirs(inference_base_dir)

def q_sample(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0).cuda()

    sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod[t]).view(-1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod[t]).view(-1, 1, 1)
    return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

def p_sample(model, x_t, t, t_index):
    betas_t = beta[t_index].view(-1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod[t_index]).view(-1, 1, 1)
    sqrt_recip_alpha_t = torch.sqrt(1 / alpha[t_index]).view(-1, 1, 1)

    predicted_noise = model(x_t, t)
    mean = sqrt_recip_alpha_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t)
    if t_index == 0:
        return mean
    noise = torch.randn_like(x_t).cuda()
    return mean + torch.sqrt(betas_t) * noise

def p_sample_loop(model, latent, shape):
    device = next(model.parameters()).device

    t = torch.full((latent.size(0),), 100 - 1, device=device, dtype=torch.long)
    noise = torch.randn_like(latent).cuda()
    x_t = q_sample(latent, t, noise)

    for t_index in reversed(range(100)):
        t = torch.full((shape[0],), t_index, device=device, dtype=torch.long)
        x_t = p_sample(model, x_t, t, t_index)
    return x_t

# 获取所有主文件夹
main_folders = [f.name for f in os.scandir(data_dir) if f.is_dir()]

# 扩散过程参数
beta_start = 0.0001
beta_end = 0.02
beta = torch.linspace(beta_start, beta_end, timesteps).cuda()
alpha = 1 - beta
alpha_cumprod = torch.cumprod(alpha, dim=0).cuda()

# 逐个处理每个主文件夹
for main_folder in main_folders:
    training_sub_dir = os.path.join(training_dir, main_folder)
    data_sub_dir = os.path.join(data_dir, main_folder)
    inference_sub_dir = os.path.join(inference_base_dir, main_folder)
    
    if not os.path.exists(inference_sub_dir):
        os.makedirs(inference_sub_dir)

    subfolders = [sf.name for sf in os.scandir(training_sub_dir) if sf.is_dir()]
    
    for subfolder in subfolders:
        model_dir = os.path.join(training_sub_dir, subfolder)
        data_folder = os.path.join(data_sub_dir, subfolder)
        # inference_folder = os.path.join(inference_sub_dir, subfolder)

        # if not os.path.exists(inference_folder):
        #     os.makedirs(inference_folder)

        # 加载模型
        model = SimpleDDPM(1024).cuda()
        model.load_state_dict(torch.load(os.path.join(model_dir, 'simple_ddpm_epoch_50000.pth')))
        model.eval()

        # 读取资料集
        dataset = EmbeddingDataset(data_folder)

        # 推理过程
        for idx in range(len(dataset)):
            latent = dataset[idx].cuda()

            # 扩展维度为 (1, 1, 1024) 并进行推理
            latent = latent.unsqueeze(0)

            for i in range(1):
                generated_vector = p_sample_loop(model, latent, (1, 1, 1024)).squeeze(0)

                # 确保生成的向量不需要梯度
                generated_vector = generated_vector.detach()

                # 创建对应的推理输出目录
                inference_step_dir = os.path.join(inference_sub_dir, f'inference_step100_{i}')
                if not os.path.exists(inference_step_dir):
                    os.makedirs(inference_step_dir)

                output_folder = os.path.join(inference_step_dir, subfolder)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                # 保存新生成的嵌入
                filename = os.path.basename(dataset.filepaths[idx])
                output_filepath = os.path.join(output_folder, filename)
                torch.save(generated_vector.cpu(), output_filepath)
                print(f"Saved generated embeddings to {output_filepath}")