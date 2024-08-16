import torch
import os

def find_files(directory, filename):
    matches = []  # 用于存储找到的文件路径
    # 遍历目录和子目录
    for root, dirs, files in os.walk(directory):
        if filename in files:
            # 如果找到文件，添加完整路径到列表中
            matches.append(os.path.join(root, filename))
    return matches

def sum_parameters(path):
    
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    total_params = 0
    for key, value in state_dict.items():
        total_params += value.numel()
    
    print(f"The Model:{path.split('/')[-2]} and total parameters: {round((total_params/1024/1024),2)} MB")


# 使用示例，假设我们在当前目录及其子目录中搜索 'best.pth'
directory_to_search = '/groups/lmm2024/home/share/Sat_Pretrain_xmq/cloud_cs0502/H8_output_cs0502/05dlinkvit_sst'  # 替换为实际目录路径
found_files = find_files(directory_to_search, 'best_model_exp1.pth')

# 打印找到的所有文件路径
# path = '/groups/lmm2024/home/share/Sat_Pretrain_xmq/cloud_cs0502/H8_output_cs0502/01unet/best_model_exp1.pth'
for path in found_files:
    # print(path)
    sum_parameters(path)


