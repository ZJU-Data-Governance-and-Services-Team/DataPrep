import numpy as np
import torch
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(project_root)
from dataprep.tabular.imputation.VAEGAIN import VAEGAIN

def generate_fake_data(N=1000, D=10, missing_rate=0.2):
    """
    生成模拟数据
    Returns:
        data_true: 完整真实数据 (Ground Truth)
        data_missing: 带 NaN 的缺失数据
        mask: 掩码矩阵 (1=观测, 0=缺失)
    """
    # 1. 生成真实数据 (正态分布)
    data_true = np.random.randn(N, D)

    # 2. 生成掩码 (随机缺失 MCAR)
    # np.random.rand 返回 [0,1) 均匀分布，> 0.2 意味着保留 80% 的数据
    mask = (np.random.rand(N, D) > missing_rate).astype(float)

    # 3. 制造缺失
    data_missing = data_true.copy()
    data_missing[mask == 0] = np.nan

    return data_true, data_missing, mask


def test_vaegain():
    print("========================================")
    print("      Testing VAE-GAIN Algorithm        ")
    print("========================================")

    # 1. 准备数据
    data_true, data_missing, mask = generate_fake_data(N=1000, D=10, missing_rate=0.2)

    # 2. 初始化模型
    # VAE 通常 batch_size 不需要太大
    imputer = VAEGAIN(
        batch_size=32,
        epoch=10,
        latent_size=5,  # 隐变量维度
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # 3. 训练并预测
    print("\n[Step 1] Training & Predicting...")
    imputed_data = imputer.train_and_predict(data_missing, mask)
    metrics = imputer.estimate(data_true, imputed_data, mask)
    imputer = VAEGAIN.load_model("./temp/vaegain_train_544p_rgk/vaegain_imputer_complete.pkl")
    imputed_data = imputer.predict(data_missing)


    # 4. 评估结果
    print("\n[Step 2] Evaluating...")
    metrics = imputer.estimate(data_true, imputed_data, mask)


if __name__ == "__main__":
    test_vaegain()