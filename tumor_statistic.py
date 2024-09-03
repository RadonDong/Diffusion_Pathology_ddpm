import os
import pandas as pd
import pickle

def count_combinations(folder_path):
    all_combinations = {}
    
    # 先遍历一遍，找到所有的组合情况
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)
        
        if os.path.isdir(subdir_path):
            clinical_pkl_path = os.path.join(subdir_path, 'clinical.pkl')
            
            if os.path.exists(clinical_pkl_path):
                # 读取 clinical.pkl 文件
                with open(clinical_pkl_path, 'rb') as file:
                    data = pickle.load(file)
                
                # 检查需要的栏位是否存在
                if 'gender' in data and 'tumor' in data:
                    # 创建一个 DataFrame 用于统计组合
                    df = pd.DataFrame(data)
                    df['gender'] = df['gender'].fillna('NaN')
                    df['tumor'] = df['tumor'].fillna('NaN')
                    combinations = df.groupby(['gender', 'tumor']).size().reset_index(name='count')
                    
                    # 收集所有组合情况
                    for index, row in combinations.iterrows():
                        key = (row['gender'], row['tumor'])
                        if key not in all_combinations:
                            all_combinations[key] = 0

    # 准备结果字典
    result = {}

    # 遍历主文件夹中的每一个子文件夹
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)
        
        if os.path.isdir(subdir_path):
            clinical_pkl_path = os.path.join(subdir_path, 'clinical.pkl')
            
            if os.path.exists(clinical_pkl_path):
                # 读取 clinical.pkl 文件
                with open(clinical_pkl_path, 'rb') as file:
                    data = pickle.load(file)
                
                # 检查需要的栏位是否存在
                if 'gender' in data and 'tumor' in data:
                    # 创建一个 DataFrame 用于统计组合
                    df = pd.DataFrame(data)
                    df['gender'] = df['gender'].fillna('NaN')
                    df['tumor'] = df['tumor'].fillna('NaN')
                    combination_counts = df.groupby(['gender', 'tumor']).size().reset_index(name='count')
                    
                    # 初始化所有组合为0
                    result[subdir] = {key: 0 for key in all_combinations.keys()}
                    
                    # 更新当前子文件夹的组合计数
                    for index, row in combination_counts.iterrows():
                        key = (row['gender'], row['tumor'])
                        result[subdir][key] = row['count']
    
    # 转换结果为 DataFrame
    final_result = pd.DataFrame(result).T.fillna(0).astype(int)
    
    # 更新列名为字符串格式
    final_result.columns = [f"{gender}_{tumor}" for gender, tumor in final_result.columns]
    
    # 重置索引，将子文件夹名称作为第一列
    final_result.reset_index(inplace=True)
    final_result.rename(columns={'index': 'subdir'}, inplace=True)
    
    # 输出到 CSV 文件
    final_result.to_csv('tumor_frozen_gender_statistic.csv', index=False)

# 设置主文件夹路径
folder_path = './Tumor_detection_clinical_pkl'

# 调用函数
count_combinations(folder_path)