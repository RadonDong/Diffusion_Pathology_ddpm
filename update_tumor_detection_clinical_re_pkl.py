import os
import pandas as pd
import numpy as np

def process_age_column(base_dir, target_base_dir):
    # 遍历所有子目录
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'clinical.pkl':
                file_path = os.path.join(root, file)
                
                # 读取pkl文件
                df = pd.read_pickle(file_path)
                
                # 计算age_at_index栏位的中位数，忽略NaN值
                median_age = df['age_at_index'].median(skipna=True)
                
                # 添加age栏位，根据age_at_index与中位数的比较结果填值，忽略NaN值
                df['age'] = df['age_at_index'].apply(lambda x: 'young' if pd.notna(x) and x < median_age else ('old' if pd.notna(x) and x >= median_age else np.nan))
                
                # 构建新的文件路径
                relative_path = os.path.relpath(root, base_dir)
                new_dir = os.path.join(target_base_dir, relative_path)
                
                # 创建目标目录（如果不存在）
                os.makedirs(new_dir, exist_ok=True)
                
                # 保存修改后的DataFrame到新的pkl文件
                new_file_path = os.path.join(new_dir, 'clinical.pkl')
                df.to_pickle(new_file_path)
                
                print(f'Processed and saved: {new_file_path}')

# 指定原始文件夹路径和目标文件夹路径
base_dir = './Tumor_detection_clinical_tsv'
target_base_dir = './Tumor_detection_clinical_pkl'

# 运行处理函数
process_age_column(base_dir, target_base_dir)