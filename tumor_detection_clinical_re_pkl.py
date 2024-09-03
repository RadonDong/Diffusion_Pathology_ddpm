import os
import pandas as pd
import numpy as np
import re

def process_clinical_files(base_dir):
    # 遍历所有子目录
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'clinical.tsv':
                file_path = os.path.join(root, file)
                
                # 读取tsv文件
                df = pd.read_csv(file_path, sep='\t')
                
                # 删除重复的行
                df = df.drop_duplicates()
                
                # 替换race栏位为not reported的值为NaN
                df['race'] = df['race'].replace('not reported', np.nan)
                
                # 替换race栏位中不为字母开头的值为NaN
                df['race'] = df['race'].apply(lambda x: x if isinstance(x, str) and re.match("^[a-zA-Z]", x.strip()) else np.nan)
                
                # 替换age_at_index不为数字的值为NaN
                df['age_at_index'] = pd.to_numeric(df['age_at_index'], errors='coerce')
                
                # 替换gender不为male或female的值为NaN
                df['gender'] = df['gender'].apply(lambda x: x if x in ['male', 'female'] else np.nan)
                
                # 重置索引，并删除原有索引
                df.reset_index(drop=True, inplace=True)
                
                print(df.shape)

                # 生成新的pkl文件路径
                pkl_file_path = os.path.join(root, 'clinical.pkl')
                
                # 保存为pkl文件
                df.to_pickle(pkl_file_path)
                
                print(f'Processed and saved: {pkl_file_path}')

# 指定文件夹路径
base_dir = './Tumor_detection_clinical_tsv'

# 运行处理函数
process_clinical_files(base_dir)