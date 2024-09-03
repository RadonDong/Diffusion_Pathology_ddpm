import os
import pandas as pd

# 定义根目录和目标目录
root_dir = './nas/Sophie/TCGA/Preprocessing'
target_dir = './Tumor_detection_clinical_tsv'

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)

# 遍历根目录下的所有子目录
for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)
    
    if os.path.isdir(subdir_path):
        # 构建clinical.tsv文件路径
        clinical_file = os.path.join(subdir_path, 'clinical.tsv')
        
        if os.path.exists(clinical_file):
            # 读取clinical.tsv文件
            df = pd.read_csv(clinical_file, sep='\t')
            
            # 提取所需字段
            required_columns = ['case_submitter_id', 'age_at_index', 'gender', 'race']
            if all(col in df.columns for col in required_columns):
                extracted_df = df[required_columns]
                
                # 构建目标子目录路径
                target_subdir = os.path.join(target_dir, subdir)
                os.makedirs(target_subdir, exist_ok=True)
                
                # 构建目标文件路径
                target_file = os.path.join(target_subdir, 'clinical.tsv')
                
                # 保存提取后的数据到目标文件
                extracted_df.to_csv(target_file, sep='\t', index=False)
                print(f'Successfully processed and saved: {target_file}')
            else:
                print(f'Missing columns in {clinical_file}. Skipping...')
        else:
            print(f'{clinical_file} does not exist. Skipping...')