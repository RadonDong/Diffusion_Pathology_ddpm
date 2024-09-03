import os
import pandas as pd

# 定义目录路径
clinical_encode_dir = './clinical_encode'

# 获取clinical_encode目录中的所有.pkl文件
pkl_files = [f for f in os.listdir(clinical_encode_dir) if f.endswith('.pkl')]

# 初始化统计结果
conflicting_cases_count = 0

for pkl_file in pkl_files:
    # 定义文件路径
    encode_pkl_path = os.path.join(clinical_encode_dir, pkl_file)
    
    if not os.path.exists(encode_pkl_path):
        print(f"文件缺失: {encode_pkl_path}")
        continue

    # 读取数据
    encode_df = pd.read_pickle(encode_pkl_path)

    # 只考虑frozen值为0的行
    filtered_df = encode_df[encode_df['frozen'] == 0]

    # 检查同一个case_submitter_id是否存在两个不同的tumor值
    conflicting_cases = filtered_df.groupby('case_submitter_id')['tumor'].nunique()
    count_conflicts = (conflicting_cases > 1).sum()

    # 打印每个文件的统计结果
    print(f"{pkl_file} 文件中存在 {count_conflicts} 个case_submitter_id对应不同的tumor值")

    # 累加总的冲突数
    conflicting_cases_count += count_conflicts

# 打印总的统计结果
print(f"总共有 {conflicting_cases_count} 个case_submitter_id在所有文件中对应不同的tumor值（frozen值为0）")