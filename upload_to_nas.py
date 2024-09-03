import os
import shutil
import filecmp

def copy_directory(source_path, destination_path):
    """复制整个目录到目标路径"""
    try:
        shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        print(f"Copied {source_path} to {destination_path}")
        return True
    except Exception as e:
        print(f"Error copying {source_path} to {destination_path}: {e}")
        return False

def verify_directory(source_path, destination_path):
    """验证目标目录下的文件和目录是否与源目录一致"""
    if not os.path.exists(destination_path):
        print(f"Destination path {destination_path} does not exist.")
        return False
    comparison = filecmp.dircmp(source_path, destination_path)
    if comparison.left_only or comparison.right_only or comparison.diff_files:
        print(f"Mismatch found: {comparison.report()}")
        return False
    return True

# 需要複製的文件夾名稱
folder_names = [
    "06_KIRC_Detection_age_2024-06-25_01-14-26",
    "14_STAD_Detection_age_2024-06-25_04-07-45",
    "16_LIHC_Detection_age_2024-06-25_05-03-14",
    "21_ESCA_Detection_age_2024-06-25_06-09-44",
    "22_PAAD_Detection_age_2024-06-25_06-47-26",
]

# 路徑設置
source_base_path = './feature_DDPM_test/inference'
destination_base_path = './nas/Nick/fairness_data_new/TCGA/MIL_diff_LR_features_detection/age'

# 要複製的文件列表（文件夹）
items_to_copy = [
    "inference_step200_0", "inference_step200_1", "inference_step200_2", "inference_step200_3", "inference_step200_4",
    "inference_step300_0", "inference_step300_1", "inference_step300_2", "inference_step300_3", "inference_step300_4",
    "inference_step400_0", "inference_step400_1", "inference_step400_2", "inference_step400_3", "inference_step400_4"
]

# 重試未成功的文件
max_retries = 3

# 複製文件的主過程
for folder_name in folder_names:
    source_folder_path = os.path.join(source_base_path, folder_name)
    destination_folder_path = os.path.join(destination_base_path, folder_name, 'DDPM_sample')
    os.makedirs(destination_folder_path, exist_ok=True)

    for item_name in items_to_copy:
        source_item_path = os.path.join(source_folder_path, item_name)
        destination_item_path = os.path.join(destination_folder_path, item_name)
        if not os.path.exists(source_item_path):
            print(f"{source_item_path} does not exist.")
            continue
        
        success = False
        for attempt in range(max_retries):
            print(f"Attempt {attempt + 1} for {source_item_path}")
            
            # 复制指定的文件夹
            if os.path.isdir(source_item_path):
                success = copy_directory(source_item_path, destination_item_path)
            else:
                print(f"{source_item_path} is not a directory.")
                break
            
            if not success:
                print(f"Copy failed for {source_item_path}. Retrying...")
                continue

            # 验证复制是否成功
            success = verify_directory(source_item_path, destination_item_path)
            if success:
                print(f"Successfully verified {source_item_path}")
                break
            else:
                print(f"Verification failed for {source_item_path}. Retrying...")

        if not success:
            print(f"Failed to copy {source_item_path} after {max_retries} attempts.")
        else:
            print(f"Item {source_item_path} copied and verified successfully.")
