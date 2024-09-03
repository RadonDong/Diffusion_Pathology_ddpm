import os
import pandas as pd

# 定义目录路径
tumor_detection_dir = './Tumor_detection_clinical_pkl'
clinical_encode_dir = './clinical_encode'

# 获取子目录列表
subdirs = [d for d in os.listdir(tumor_detection_dir) if os.path.isdir(os.path.join(tumor_detection_dir, d))]

for subdir in subdirs:
    # 定义文件路径
    clinical_pkl_path = os.path.join(tumor_detection_dir, subdir, 'clinical.pkl')
    encode_pkl_path = os.path.join(clinical_encode_dir, f'{subdir}_clinical_encode.pkl')

    if not os.path.exists(clinical_pkl_path) or not os.path.exists(encode_pkl_path):
        print(f"文件缺失: {clinical_pkl_path} 或 {encode_pkl_path}")
        continue

    try:
        # 读取数据
        clinical_df = pd.read_pickle(clinical_pkl_path)
        encode_df = pd.read_pickle(encode_pkl_path)

        # 保留 frozen 为 0 的数据，并按 case_submitter_id 分组，选择每组中第一个出现的 tumor 值
        encode_df = encode_df[encode_df['frozen'] == 0]
        encode_first_tumor = encode_df.groupby('case_submitter_id').first()['tumor']

        # 过滤并更新临床数据
        clinical_df['tumor'] = clinical_df['case_submitter_id'].map(encode_first_tumor)
        clinical_df = clinical_df[clinical_df['tumor'].notna()]

        # 保存修改后的DataFrame
        clinical_df.to_pickle(clinical_pkl_path)
        print(f"已更新文件: {clinical_pkl_path}")
        # print(encode_df[encode_df['case_submitter_id'] == "TCGA-W5-AA2U"])
        # print(encode_first_tumor)
        print(clinical_df)
    except Exception as e:
        print(f"处理文件时出错: {clinical_pkl_path} 或 {encode_pkl_path}，错误信息: {e}")