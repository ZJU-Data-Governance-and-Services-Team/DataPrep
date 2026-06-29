import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, List, Optional, Union
import time

class InfluenceEstimator:
    """
    影响力估计器，使用Full Influence Function估计样本对验证集性能的影响
    """
    
    def __init__(self, classifier, processor, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化影响力估计器
        
        Args:
            classifier: 分类器模型
            processor: 数据处理器
            device: 计算设备
        """
        self.classifier = classifier
        self.processor = processor
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def get_params_gradients(self, 
                             dataloader: DataLoader, 
                             normalize: bool = True) -> torch.Tensor:
        """
        计算模型在一批数据上的平均参数梯度
        
        Args:
            dataloader: 数据加载器
            normalize: 是否对梯度进行归一化
            
        Returns:
            平均参数梯度的向量表示
        """
        # 确保处于评估模式但允许梯度计算
        self.classifier.eval()
        
        # 确保所有参数都启用了梯度计算
        for param in self.classifier.parameters():
            param.requires_grad = True
            
        grad_sum = None
        num_samples = 0
        
        for batch in dataloader:
            if len(batch) == 2:  # 假设dataloader返回(x, y)
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
            else:  
                # 处理其他格式的数据，需要根据实际情况调整
                continue
                
            batch_size = x.shape[0]
            num_samples += batch_size
            
            self.classifier.zero_grad()
            with torch.enable_grad():  # 显式启用梯度计算
                outputs = self.classifier(x)
                loss = self.criterion(outputs, y)
                loss.backward()
            
            # 收集所有参数的梯度
            grads = []
            for param in self.classifier.parameters():
                if param.grad is not None:
                    grads.append(param.grad.view(-1).detach().clone())  # 复制梯度以防后续操作影响计算
                else:
                    # 如果参数没有梯度，则使用零向量
                    grads.append(torch.zeros_like(param.view(-1)))
            
            if not grads:
                continue
                
            batch_grad = torch.cat(grads)
            
            if grad_sum is None:
                grad_sum = batch_grad
            else:
                grad_sum += batch_grad
        
        # 计算平均梯度
        if num_samples > 0 and grad_sum is not None:
            avg_grad = grad_sum / num_samples
            
            # 归一化梯度（可选）
            if normalize and torch.norm(avg_grad) > 0:
                avg_grad = avg_grad / torch.norm(avg_grad)
                
            return avg_grad
        else:
            raise ValueError("No valid samples processed")
    
    def compute_hessian_vector_product(self, 
                                      vector: torch.Tensor, 
                                      dataloader: DataLoader,
                                      damping: float = 1e-5) -> torch.Tensor:
        """
        计算黑森矩阵-向量乘积 (Hv)
        
        Args:
            vector: 与黑森矩阵相乘的向量
            dataloader: 用于计算黑森矩阵的数据
            damping: 阻尼系数，用于数值稳定性
            
        Returns:
            黑森矩阵-向量乘积
        """
        self.classifier.eval()
        
        # 首先收集模型参数
        params = [p for p in self.classifier.parameters() if p.requires_grad]
        
        for batch in dataloader:
            if len(batch) == 2:  # 假设dataloader返回(x, y)
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
            else:  
                # 处理其他格式的数据
                continue
                
            batch_size = x.shape[0]
            
            # 正向传播
            self.classifier.zero_grad()
            outputs = self.classifier(x)
            loss = self.criterion(outputs, y)
            
            # 计算一阶导数
            grads = torch.autograd.grad(loss, params, create_graph=True)
            
            # 计算一阶导数和向量的点积
            grad_vector_product = sum([torch.sum(g * v) for g, v in zip(grads, unflatten_vector(vector, params))])
            
            # 计算点积对参数的二阶导数
            hvp = torch.autograd.grad(grad_vector_product, params, retain_graph=True)
            
            break  # 只使用一个批次，这是一个近似
        
        # 将hvp展平并添加阻尼
        hvp_flat = flatten_tensor_list(hvp)
        hvp_flat += damping * vector
        
        return hvp_flat
            
    def compute_inverse_hvp(self, 
                           vector: torch.Tensor, 
                           dataloader: DataLoader,
                           num_iterations: int = 50,
                           damping: float = 1e-5,
                           recursion_depth: int = 10000) -> torch.Tensor:
        """
        使用共轭梯度法计算 (H^-1)v
        
        Args:
            vector: 要乘以逆黑森矩阵的向量
            dataloader: 用于计算黑森矩阵的数据
            num_iterations: 共轭梯度迭代次数
            damping: 阻尼系数
            recursion_depth: 最大递归深度（防止无限递归）
            
        Returns:
            逆黑森矩阵-向量乘积
        """
        # 使用共轭梯度法近似求解 Hx = v
        x = torch.zeros_like(vector)
        r = vector.clone()
        p = r.clone()
        
        for i in range(num_iterations):
            # 计算 Hp
            Hp = self.compute_hessian_vector_product(p, dataloader, damping)
            
            # 计算步长
            alpha = torch.dot(r, r) / torch.dot(p, Hp)
            
            # 更新x
            x += alpha * p
            
            # 更新残差
            r_new = r - alpha * Hp
            
            # 检查收敛
            if torch.norm(r_new) < 1e-10:
                break
                
            # 计算共轭方向的缩放系数
            beta = torch.dot(r_new, r_new) / torch.dot(r, r)
            
            # 更新方向
            p = r_new + beta * p
            r = r_new
        
        return x
        
    def calculate_influence_scores(self, 
                                imputed_df: pd.DataFrame,
                                validation_df: pd.DataFrame,
                                baseline_df: Optional[pd.DataFrame] = None) -> Dict[int, float]:
        """
        计算填补样本对验证集性能的影响力分数
        
        Args:
            imputed_df: 填补后的数据，包含索引
            validation_df: 验证集数据
            baseline_df: 基准版本数据（如简单填补的版本），如果为None则假设首次处理
            
        Returns:
            样本索引到影响力分数的映射
        """
        print("开始计算影响力分数...")
        start_time = time.time()
        
        # 1. 准备验证集数据
        x_num_val, x_cat_val = self.processor.transform_onehot(validation_df)
        target_col = self.processor.target_feature[0]
        target_values = validation_df[target_col].values
        if isinstance(target_values[0], (list, np.ndarray)):
            target_values = np.array([val[0] if isinstance(val, (list, np.ndarray)) else val for val in target_values])
        encoded_labels = np.array([self.processor.label_encoder.get(val, 0) for val in target_values])
        y_val = torch.LongTensor(encoded_labels)
        
        # 合并特征
        x_combined_val = torch.cat((x_num_val, x_cat_val), dim=1)
        val_dataset = TensorDataset(x_combined_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=16)
        
        # 2. 计算验证集梯度的平均值 v_val
        print("计算验证集梯度的平均值...")
        v_val = self.get_params_gradients(val_loader, normalize=True)
        
        # 3. 计算(H^-1)v_val
        print("计算H逆与v_val的乘积...")
        # 创建数据加载器用于Hessian计算
        if baseline_df is not None:
            # 使用baseline数据创建dataloader
            x_num_base, x_cat_onehot_base = self.processor.transform_onehot(baseline_df)
            x_combined_base = torch.cat((x_num_base, x_cat_onehot_base), dim=1)
            target_col = self.processor.target_feature[0]
            target_values = baseline_df[target_col].values
            if isinstance(target_values[0], (list, np.ndarray)):
                target_values = np.array([val[0] if isinstance(val, (list, np.ndarray)) else val for val in target_values])
            encoded_labels = np.array([self.processor.label_encoder.get(val, 0) for val in target_values])
            y_base = torch.LongTensor(encoded_labels)
            
            baseline_dataset = TensorDataset(x_combined_base, y_base)
            baseline_loader = DataLoader(baseline_dataset, batch_size=16)
            hvp_loader = baseline_loader
        else:
            # 如果没有baseline数据，则使用验证集作为备选
            hvp_loader = val_loader
        
        inverse_hvp = self.compute_inverse_hvp(v_val, hvp_loader)
        
        # 4. 计算每个填补样本的影响力分数
        influence_scores = {}
        
        # 转换填补数据和基准数据
        x_num_imp, x_cat_onehot_imp = self.processor.transform_onehot(imputed_df)
        x_combined_imp = torch.cat((x_num_imp, x_cat_onehot_imp), dim=1)
        target_col = self.processor.target_feature[0]
        target_values = imputed_df[target_col].values
        if isinstance(target_values[0], (list, np.ndarray)):
            target_values = np.array([val[0] if isinstance(val, (list, np.ndarray)) else val for val in target_values])
        encoded_labels = np.array([self.processor.label_encoder.get(val, 0) for val in target_values])
        y_imp = torch.LongTensor(encoded_labels)
        
        print("计算每个样本的影响力分数...")
        batch_size = 32
        for i in range(0, len(imputed_df), batch_size):
            end_idx = min(i + batch_size, len(imputed_df))
            
            # 处理填补样本批次
            x_batch_imp = x_combined_imp[i:end_idx].to(self.device)
            y_batch_imp = y_imp[i:end_idx].to(self.device)
            
            self.classifier.zero_grad()
            outputs_imp = self.classifier(x_batch_imp)
            loss_imp = self.criterion(outputs_imp, y_batch_imp)
            grads_imp = torch.autograd.grad(loss_imp, self.classifier.parameters())
            
            # 将梯度展平为向量
            grad_imp_vector = flatten_tensor_list(grads_imp)
            
            if baseline_df is not None:
                # 处理基准样本批次
                x_batch_base = x_combined_base[i:end_idx].to(self.device)
                y_batch_base = y_base[i:end_idx].to(self.device)
                
                self.classifier.zero_grad()
                outputs_base = self.classifier(x_batch_base)
                loss_base = self.criterion(outputs_base, y_batch_base)
                grads_base = torch.autograd.grad(loss_base, self.classifier.parameters())
                
                # 将梯度展平为向量
                grad_base_vector = flatten_tensor_list(grads_base)
                
                # 计算梯度差
                grad_diff = grad_imp_vector - grad_base_vector
            else:
                # 如果没有基准数据，直接使用填补样本的梯度
                grad_diff = grad_imp_vector
            
            # 计算影响力分数 -dot(inverse_hvp, grad_diff)
            inf_score = -torch.dot(inverse_hvp, grad_diff).item()
            
            # 存储每个样本的影响力分数
            for j in range(i, end_idx):
                influence_scores[imputed_df.index[j]] = inf_score / (end_idx - i)  # 平均到每个样本
        
        elapsed_time = time.time() - start_time
        print(f"影响力分数计算完成，耗时 {elapsed_time:.2f} 秒")
        
        return influence_scores

# 辅助函数
def flatten_tensor_list(tensor_list):
    """将张量列表展平为一个向量"""
    return torch.cat([t.view(-1) for t in tensor_list])

def unflatten_vector(vector, params_ref):
    """将展平的向量还原为原始参数的形状"""
    pointer = 0
    param_list = []
    
    for param in params_ref:
        num_params = param.numel()
        param_flat = vector[pointer:pointer + num_params]
        param_list.append(param_flat.view_as(param))
        pointer += num_params
        
    return param_list 