import os
import numpy as np
import torch
from sklearn.neighbors import KernelDensity

def load_tensors_from_folder(folder_path):
    tensors = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        tensor = torch.load(file_path)
        tensors.append(tensor.numpy())  # 将tensor转换为numpy数组
    return np.vstack(tensors)

# Scott, D.W. (1992) "Multivariate Density Estimation: Theory, Practice, and Visualization." John Wiley & Sons.
def scott_bandwidth(data):
    n = data.shape[0]  # 数据点数量
    return n ** (-1 / (data.shape[1] + 4))  # 计算带宽

def compute_kde(tensors):
    bandwidth = scott_bandwidth(tensors)  # 计算带宽
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

def compute_mean_variance(tensor):
    mean = np.mean(tensor, axis=0)
    variance = np.var(tensor, axis=0)
    return mean, variance

def mse(a, b):
    return np.mean((a - b) ** 2)

def combined_score(folder1_path, folder2_path, alpha=0.001, beta=100.0, gamma=1.0):
    # 载入tensor
    tensors1 = load_tensors_from_folder(folder1_path)
    tensors2 = load_tensors_from_folder(folder2_path)
    
    # 计算均值和方差
    mu_X, sigma_X = compute_mean_variance(tensors1)
    mu_Y, sigma_Y = compute_mean_variance(tensors2)
    
    # 计算均值和方差的MSE
    mse_mean = mse(mu_X, mu_Y)
    mse_variance = mse(sigma_X, sigma_Y)

    # 对方差进行对数变换以减少极端值的影响
    log_sigma_X = np.log(sigma_X + 1e-10)
    log_sigma_Y = np.log(sigma_Y + 1e-10)
    log_mse_variance = mse(log_sigma_X, log_sigma_Y)
    
    # 计算核密度估计
    kde1 = compute_kde(tensors1)
    kde2 = compute_kde(tensors2)
    
    # 计算KL散度
    kl_div1, kl_div2 = compute_kl_divergence(tensors1, tensors2, kde1, kde2)
    kl_div = (kl_div1 + kl_div2) / 2  # 平均KL散度
    
    # 对KL散度、均值的MSE和方差的MSE进行适当变换
    transformed_kl_div = np.log(kl_div + 1e-10)
    transformed_mse_mean = np.log(mse_mean + 1e-10)
    transformed_mse_variance = np.log(mse_variance)
    
    # 计算综合得分
    score = -alpha * transformed_kl_div - beta * transformed_mse_mean + gamma * transformed_mse_variance
    return score, kl_div, mse_mean, mse_variance

# 文件夹路径
folder1_path = './feature_DDPM_test/inference/04_LUAD_CSMD3_gender_2024-07-02_01-00-54/inference_step1000_0/0_0'
folder2_path = './nas/Nick/fairness_data_new/TCGA/MIL_diff_LR_features_mutation/gender/04_LUAD_CSMD3_gender_2024-07-02_01-00-54/0_0'

# 计算综合得分
score, kl_div, mse_mean, mse_variance = combined_score(folder1_path, folder2_path)

print("Combined Score:", score)
print("KL Divergence:", kl_div)
print("MSE Mean:", mse_mean)
print("MSE Variance:", mse_variance)