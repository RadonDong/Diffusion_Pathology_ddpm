import os
import pandas as pd

def extract_clinical_data(file_name):
    folder_id = file_name[:-3]  # 去掉 '.pt' 后缀
    parts = file_name.split('-')
    case_submitter_id = '-'.join(parts[:3])
    frozen = 1 if file_name[20:22] == 'DX' else 0
    tumor = 0 if file_name[13] == '0' else 1
    return folder_id, case_submitter_id, frozen, tumor

def process_folder(folder_path):
    clinical_data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pt'):
            folder_id, case_submitter_id, frozen, tumor = extract_clinical_data(file_name)
            clinical_data.append({
                'folder_id': folder_id,
                'case_submitter_id': case_submitter_id,
                'frozen': frozen,
                'tumor': tumor
            })
    return clinical_data

def save_clinical_data(base_path, sub_folder_name, clinical_data):
    output_folder = os.path.join("./", 'clinical_encode')
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"{sub_folder_name}_clinical_encode.pkl")
    
    df = pd.DataFrame(clinical_data)
    df.to_pickle(output_file)

def main(base_path):
    for sub_folder_name in os.listdir(base_path):
        sub_folder_path = os.path.join(base_path, sub_folder_name)
        if os.path.isdir(sub_folder_path):
            clinical_data = process_folder(sub_folder_path)
            save_clinical_data(base_path, sub_folder_name, clinical_data)

if __name__ == '__main__':
    base_path = './nas/Sophie/TCGA/CHIEF_features'
    main(base_path)