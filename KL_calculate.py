import os
import numpy as np
import torch
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity

def load_tensors_from_folder(folder_path):
    tensors = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        tensor = torch.load(file_path)
        tensors.append(tensor.numpy())  # 将tensor转换为numpy数组
    return np.vstack(tensors)

def compute_kde(tensors, bandwidth=0.1):
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(tensors)
    return kde

def compute_kl_divergence(tensors1, tensors2, kde1, kde2):
    log_density1 = kde1.score_samples(tensors1)
    log_density2 = kde2.score_samples(tensors1)
    kl_div1 = np.mean(log_density1 - log_density2)
    
    log_density1 = kde1.score_samples(tensors2)
    log_density2 = kde2.score_samples(tensors2)
    kl_div2 = np.mean(log_density2 - log_density1)
    
    return kl_div1, kl_div2

# 文件夹路径
folder1_path = './feature_DDPM_test/MIL_diff_LR_features/06_KIRC_Detection_age_2024-06-25_01-14-26/0_1'
folder2_path = './feature_DDPM_test/inference/06_KIRC_Detection_age_2024-06-25_01-14-26/inference_step50_0/0_1'

# 载入tensor
tensors1 = load_tensors_from_folder(folder1_path)
tensors2 = load_tensors_from_folder(folder2_path)

# 计算核密度估计
kde1 = compute_kde(tensors1)
kde2 = compute_kde(tensors2)

# 计算KL散度
kl_div1, kl_div2 = compute_kl_divergence(tensors1, tensors2, kde1, kde2)

print(f'KL散度 from folder1 to folder2: {kl_div1}')
print(f'KL散度 from folder2 to folder1: {kl_div2}')