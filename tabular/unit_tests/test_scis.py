import numpy as np
import torch
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(project_root)
from dataprep.tabular.imputation.SCIS import SCIS

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


def test_scis():
    print("========================================")
    print("      Testing SCIS Algorithm            ")
    print("========================================")

    # 1. 准备数据
    data_true, data_missing, mask = generate_fake_data(N=1000, D=10, missing_rate=0.2)

    # 2. 初始化模型
    imputer = SCIS(
        batch_size=64,
        epoch=100,
        initial_value=300,  # 初始训练集大小
        thre_value=0.2,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # 3. 训练并预测
    print("\n[Step 1] Training & Predicting (Includes 3 Phases)...")
    imputed_data = imputer.train_and_predict(data_missing, mask)
    metrics = imputer.estimate(data_true, imputed_data, mask)
    imputer = SCIS.load_model("./temp/scis_train_3s1si9c5/scis_imputer_complete.pkl")
    imputed_data = imputer.predict(data_missing)

    # 4. 评估结果
    print("\n[Step 2] Evaluating...")
    metrics = imputer.estimate(data_true, imputed_data, mask)

    print("\n[Result] SCIS Test Passed!")
    print(f"Final RMSE: {metrics['rmse']:.4f}")


if __name__ == "__main__":
    test_scis()