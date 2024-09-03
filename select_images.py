import os
import random
import shutil
import pandas as pd

# 指定文件夹路径
folder_path = './nas/TCGA/Patches_20x/01_BRCA'

# 读取.pkl文件内容
data = pd.read_pickle('./Mutation_BRCA/CDH1/GEN/old 01_BRCA v3 40 Formalin mutation1_GEN.pkl')

# 获取子文件夹名称列表
subfolders = data[4].tolist()

# 定义函数来筛选指定大小的.jpg文件并复制到目标文件夹
def select_and_copy_files(folder, min_size, max_size, num_files, dest_folder):
    selected_files = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        try:
            if filename.endswith('.jpg') and os.path.getsize(filepath) >= min_size and os.path.getsize(filepath) <= max_size:
                selected_files.append(filepath)
        except OSError as e:
            print(f"I/O error({e.errno}): {e.strerror} - {filepath}")
    selected_files = random.sample(selected_files, min(num_files, len(selected_files)))
    for file in selected_files:
        shutil.copy(file, dest_folder)

# 目标文件夹路径
dest_folder = './BRCA_Program/lora-test/lora-test/BRCA_Common_CDH1_formalin_withtumor_Old_mutation1'
os.makedirs(dest_folder, exist_ok=True)  # 创建目标文件夹

# 遍历每个子文件夹，选择符合条件的文件并复制到目标文件夹
for subfolder in subfolders:
    subfolder_path = os.path.join(folder_path, subfolder)
    if os.path.isdir(subfolder_path):
        select_and_copy_files(subfolder_path, 40*1024, 70*1024, 20, dest_folder)