import numpy as np
import torch
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(project_root)
from dataprep.tabular.imputation.EDIT import EDIT


def generate_fake_data(N=500, D=8, missing_rate=0.2, seed=42):
    np.random.seed(seed)
    data_true = np.random.randn(N, D)
    mask = (np.random.rand(N, D) > missing_rate).astype(float)

    data_missing = data_true.copy()
    data_missing[mask == 0] = np.nan

    return data_true, data_missing, mask


def test_edit():
    print("========================================")
    print("      Testing EDIT Algorithm            ")
    print("========================================")

    N, D = 500, 8
    data_true, data_missing, mask = generate_fake_data(N=N,D=D,missing_rate=0.2,seed=42)

    print(f"Data Shape: {data_true.shape}, Missing Rate: 0.2")

    imputer = EDIT(
        batch_size=8,
        hint_rate=0.9,
        alpha=10,
        epoch=1,
        initial_size=60,
        validation_size=40,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    print("\n[Step 1] Training & Predicting...")
    imputed_data = imputer.train_and_predict(data_missing, mask)

    print("\n[Step 2] Checking output...")
    assert imputed_data.shape == data_true.shape
    assert not np.isnan(imputed_data).any()

    observed = mask.astype(bool)
    np.testing.assert_array_almost_equal(imputed_data[observed],data_missing[observed],decimal=4)

    print("\n[Step 3] Evaluating...")
    metrics = imputer.estimate(data_true, imputed_data, mask)

    print("\n[Result] EDIT Test Passed!")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE : {metrics['mae']:.4f}")


if __name__ == "__main__":
    test_edit()