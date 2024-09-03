import torch

def get_tensor_size(file_path, tensor_name=None):
    # Load the .pt file
    data = torch.load(file_path)
    
    if isinstance(data, dict):
        # If the data is a dictionary, find the specified tensor or list all tensors
        if tensor_name:
            if tensor_name in data:
                tensor = data[tensor_name]
                print(f"Size of tensor '{tensor_name}': {tensor.size()}")
            else:
                print(f"Tensor '{tensor_name}' not found in the file.")
        else:
            print("Available tensors in the file:")
            for key, value in data.items():
                if torch.is_tensor(value):
                    print(f"Tensor '{key}': {value.size()}")
    elif torch.is_tensor(data):
        # If the data is a single tensor, print its size
        print(f"Size of the tensor: {data.size()}")
    else:
        print("The .pt file does not contain any tensor.")

# Example usage
file_path = './inference/01_BRCA_CDH1_age_2024-07-19_07-11-29/inference_step50_0/0_0/TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291.pt'  # Replace with your .pt file path
get_tensor_size(file_path, tensor_name='your_tensor_name')  # Replace 'your_tensor_name' with the actual tensor name, or leave it as None to list all