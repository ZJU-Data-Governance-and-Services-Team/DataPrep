import pandas as pd
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(project_root)
from dataprep.tabular.detection.ZeroED import ZeroED  # 导入类


# ==========================================
# 1. 准备数据
# ==========================================
def setup_data():
    # 构造一个包含 3 个明显错误的数据集
    data = pd.DataFrame({
        'name': ['Alice', 'Bob', 'ErrorVal', 'Charlie', 'David', 'Frank', 'Grace', 'Hebe', 'Ivy', 'Jack'],
        'age': ['25', '30', '999', '35', '40', '28', '22', '45', '33', '29'],
        'city': ['NY', 'LA', 'Unknown', 'SF', 'NY', 'LA', 'SF', 'NY', 'LA', 'SF']
    })
    return data


# ==========================================
# 2. 漂亮地打印结果 (解决“看不懂结果”的问题)
# ==========================================
def print_errors_nicely(df_data, df_mask):
    print("\n" + "=" * 40)
    print("      🔍 错误检测报告")
    print("=" * 40)

    error_count = 0
    # 遍历掩码矩阵，找到 True 的位置
    for col in df_mask.columns:
        for idx, is_error in df_mask[col].items():
            if is_error:  # 如果是 True，说明检测到了错误
                val = df_data.at[idx, col]
                print(f"🔴 [发现错误] 第 {idx} 行, 列 '{col}'")
                print(f"    -> 异常值: {val}")
                error_count += 1

    if error_count == 0:
        print("✅ 未发现任何错误。")
    else:
        print("-" * 40)
        print(f"总计发现 {error_count} 个错误。")
    print("=" * 40 + "\n")


# ==========================================
# 3. 主程序
# ==========================================
if __name__ == "__main__":  # 注意：这里必须是双下划线

    # 1. 准备数据
    data = setup_data()
    print(">>> 原始数据预览:")
    print(data.head(3))

    # 2. 初始化检测器
    print("\n>>> 初始化 ZeroED...")
    detector = ZeroED(
        api_key='EMPTY',
        model_name="qwen2.5-7b",
        base_url="http://localhost:8000/v1",
        local_model_use=True,
        n_method='kmeans',
        result_dir='../detection/result_final_run',
        verbose=True
    )

    try:
        # 3. 训练
        # 注意：这一步需要调用真实的 API，请确保 utility.py 里填了 API Key
        print("\n>>> 正在运行")
        detector.train(data)
        res_mask = detector.predict(data)
        print_errors_nicely(data, res_mask)

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        print("💡 提示: 如果是 API 报错，请检查API Key 是否正确，或网络是否通畅。")