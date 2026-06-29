import numpy as np
import torch
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(project_root)
from dataprep.tabular.imputation.GAIN import GAIN

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
    mask = (np.random.rand(N, D) > missing_rate).astype(float)

    # 3. 制造缺失
    data_missing = data_true.copy()
    data_missing[mask == 0] = np.nan

    return data_true, data_missing, mask


def test_gain():
    print("========================================")
    print("      Testing GAIN Algorithm            ")
    print("========================================")

    # 1. 准备数据
    N, D = 1000, 10
    data_true, data_missing, mask = generate_fake_data(N, D, missing_rate=0.2)
    print(f"Data Shape: {data_true.shape}, Missing Rate: 0.2")

    # 2. 初始化模型
    # 为了测试速度，epoch 设置较小
    imputer = GAIN(
        batch_size=64,
        hint_rate=0.9,
        alpha=100,
        epoch=10,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # 3. 训练并预测 (一行代码调用)
    print("\n[Step 1] Training & Predicting...")
    imputed_data = imputer.train_and_predict(data_missing, mask)
    metrics = imputer.estimate(data_true, imputed_data, mask)
    imputer = GAIN.load_model("./temp/gain_train_k_y25iky/gain_imputer_complete.pkl")
    imputed_data = imputer.predict(data_missing)

    # 4. 评估结果
    print("\n[Step 2] Evaluating...")
    metrics = imputer.estimate(data_true, imputed_data, mask)

    print("\n[Result] GAIN Test Passed!")


if __name__ == "__main__":
    test_gain()