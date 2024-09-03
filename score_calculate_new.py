import os
import numpy as np
import torch
from sklearn.neighbors import KernelDensity, NearestNeighbors

def load_tensors_from_folder(folder_path):
    tensors = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        tensor = torch.load(file_path)
        tensors.append(tensor.numpy())  # 将tensor转换为numpy数组
    return np.vstack(tensors)

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

def compute_mean_nearest_distance(tensors_generated, tensors_train):
    # 使用NearestNeighbors来计算每个生成样本与训练样本的最近距离
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(tensors_train)
    distances, _ = nbrs.kneighbors(tensors_generated)
    
    mean_distance = np.mean(distances)
    return mean_distance

def combined_score(folder1_path, folder2_path, alpha=0.05, beta=1.0):
    # 载入tensor
    tensors1 = load_tensors_from_folder(folder1_path)  # 训练数据
    tensors2 = load_tensors_from_folder(folder2_path)  # 生成数据
    
    # 计算核密度估计
    kde1 = compute_kde(tensors1)
    kde2 = compute_kde(tensors2)
    
    # 计算KL散度
    kl_div1, kl_div2 = compute_kl_divergence(tensors1, tensors2, kde1, kde2)
    kl_div = (kl_div1 + kl_div2) / 2  # 平均KL散度
    
    # 计算生成数据与训练数据的对应距离
    diversity_score = compute_mean_nearest_distance(tensors2, tensors1)
    
    # 计算综合得分
    score = -alpha * kl_div + beta * diversity_score
    return score, kl_div, diversity_score

# 文件夹路径
folder1_path = './nas/Nick/fairness_data_new/TCGA/MIL_diff_LR_features_mutation/gender/04_LUAD_CSMD3_gender_2024-07-02_01-00-54/DDPM_sample/inference_step400_0/0_0'
folder2_path = './nas/Nick/fairness_data_new/TCGA/MIL_diff_LR_features_mutation/gender/04_LUAD_CSMD3_gender_2024-07-02_01-00-54/0_0'

# 计算综合得分
score, kl_div, diversity_score = combined_score(folder1_path, folder2_path)

print("Combined Score:", score)
print("KL Divergence:", kl_div)
print("Diversity Score:", diversity_score)