import torch
import torch.nn as nn
import numpy as np
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 计算Hessian矩阵
def compute_hessian(dataloader, model, criterion):
    """
    计算模型在给定数据集上的Hessian矩阵
    
    参数:
    - dataloader: 数据加载器
    - model: 神经网络模型
    - criterion: 损失函数
    
    返回:
    - hessian: Hessian矩阵
    - gradient: 梯度向量
    """
    print("Step 1: 开始计算Hessian矩阵...")
    start_time = time.time()
    
    model.eval()
    model.zero_grad(set_to_none=True)
    
    # 内部函数：计算Hessian矩阵的一行
    def compute_hessian_row(g_tensor):
        g_tensor = g_tensor.cuda()
        total_params = g_tensor.size(0)
        hessian_list = []
        
        for d in range(total_params):
            # 创建单位向量
            unit_vector = torch.zeros(total_params)
            unit_vector[d] = 1
            unit_vector = unit_vector.cuda()
            
            # 计算梯度的方向导数
            l = torch.sum(g_tensor * unit_vector)
            l.backward(retain_graph=True)
            
            # 收集Hessian的一行
            hessian_row = []
            for name, param in model.named_parameters():
                if 'ln' in name or 'bias' in name or 'wte' in name or 'wpe' in name:
                    continue
                if param.requires_grad:
                    hessian_row.append(param.grad.double().data.clone())
            
            model.zero_grad(set_to_none=True)
            hessian_row = [g.flatten() for g in hessian_row]
            hessian_row = [g.cpu() for g in hessian_row]
            hessian_row = torch.cat(hessian_row)
            
            hessian_list.append(hessian_row)
            
            # 打印进度
            if d % 500 == 0 and d > 0:
                print(f"  计算Hessian: 已完成 {d}/{total_params} 行, 用时 {time.time() - start_time:.2f}秒")
                
        hessian = torch.stack(hessian_list, dim=1)
        return hessian
    
    # 计算完整的Hessian矩阵
    full_hessian = 0
    for images, labels in dataloader:
        labels = labels.cuda()
        images = images.reshape(-1, 28 * 28).cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 缩放损失并计算梯度
        scaled_loss = loss / len(dataloader)
        scaled_loss.backward(create_graph=True)
        
        # 收集梯度
        g_list = []
        for name, param in model.named_parameters():
            if 'ln' in name or 'bias' in name or 'wte' in name or 'wpe' in name:
                continue
            if param.requires_grad:
                g_list.append(torch.flatten(param.grad.double()))
        
        g_tensor = torch.cat(g_list, dim=0)
        model.zero_grad(set_to_none=True)
        
        # 计算Hessian矩阵
        H = compute_hessian_row(g_tensor)
        full_hessian += H
    
    # 处理数值问题
    full_hessian = torch.nan_to_num(full_hessian, nan=0, posinf=0, neginf=0)
    
    # 确保Hessian矩阵是对称的
    full_hessian = full_hessian.numpy().astype(np.float64)
    full_hessian = (full_hessian + full_hessian.T) / 2
    
    print(f"Hessian矩阵计算完成，用时 {time.time() - start_time:.2f}秒")
    return full_hessian, g_tensor

# 通过SVD计算特征值
def compute_eigenvalues_svd(hessian):
    """
    通过SVD分解计算Hessian矩阵的特征值
    
    参数:
    - hessian: Hessian矩阵
    
    返回:
    - eigenvalues: 特征值
    - u: 左奇异向量
    - v: 右奇异向量
    """
    print("\nStep 2: 开始通过SVD计算特征值...")
    start_time = time.time()
    
    # 将Hessian矩阵转换为PyTorch张量并移至GPU
    hessian_tensor = torch.tensor(hessian).cuda()
    
    # 执行SVD分解
    u, sigma, v = torch.svd(hessian_tensor)
    
    print(f"SVD分解完成，用时 {time.time() - start_time:.2f}秒")
    print(f"特征值形状: {sigma.shape}")
    
    # 打印前10个最大的特征值
    print("\n前10个最大的特征值:")
    for i, val in enumerate(sigma[:10]):
        print(f"  λ_{i+1} = {val.item():.6f}")
    
    return sigma, u, v

# 加载预训练模型并计算Hessian和特征值
def main():
    # 设置随机种子
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载数据集
    print("加载MNIST数据集...")
    kwargs = {'num_workers': 4, 'pin_memory': True}
    try:
        # 尝试使用用户本地路径
        data_path = os.path.join(os.getcwd(), 'data')
        train_dataset = datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
    except Exception as e:
        print(f"加载数据出错: {e}")
        print("尝试使用默认路径...")
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    
    # 创建数据加载器
    train_loader = DataLoader(dataset=train_dataset, batch_size=60000, shuffle=False, **kwargs)
    
    # 模型参数
    input_size = 784  # 28x28
    hidden_size = 5
    output_size = 10  # 0-9的数字
    
    # 创建模型
    model = MLP(input_size, hidden_size, output_size)
    
    # 尝试加载预训练模型
    print("\n尝试加载预训练模型...")
    try:
        script_dir = os.getcwd()
        ckpt_dir = "bs256_lr0.05_e30_hidden5_SGD_m0.9_seed0"
        model_path = f"{script_dir}/trained_MLP/{ckpt_dir}/model_para_e_0.1_decaying_2.0_pipecheck_w.pt"
        
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict['state_dict'])
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("使用随机初始化的模型继续...")
    
    # 将模型移至GPU
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    
    # 计算Hessian矩阵
    hessian, gradient = compute_hessian(train_loader, model, criterion)
    
    # 计算特征值
    eigenvalues, u, v = compute_eigenvalues_svd(hessian)
    
    # 计算Rayleigh商 (g^T H g) / ||g||^2
    gradient_cuda = gradient.cuda()
    hessian_cuda = torch.tensor(hessian).cuda()
    rayleigh = torch.matmul(gradient_cuda.reshape(-1,1).T, 
                           torch.matmul(hessian_cuda, 
                                      gradient_cuda.reshape(-1,1))) / torch.norm(gradient_cuda)**2
    
    print(f"\nRayleigh商 (g^T H g) / ||g||^2 = {rayleigh.item():.6f}")
    print("\n计算完成!")

if __name__ == "__main__":
    main()