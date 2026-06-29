import sys
import os
import pandas as pd
import pytest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(project_root)

from dataprep.tabular.detection.MLNClean import MLNClean as MLNCleanDetector
from dataprep.tabular.correction.MLNClean import MLNClean as MLNCleanCorrector


def check_dependencies():
    try:
        import torch      # noqa: F401
        import pyro       # noqa: F401
        import Levenshtein  # noqa: F401
    except ImportError as e:
        print("\n[Skip] MLNClean full test skipped.")
        print("Reason:", e)
        print("Install with: pip install torch pyro-ppl python-Levenshtein")
        return False
    return True


def generate_fake_data():
    dirty_df = pd.DataFrame({
        "ID": [1, 2, 3, 4, 5, 6],
        "flight": ["CA1", "CA1", "CA1", "CA2", "CA2", "CA2"],
        "sched_dep_time": ["08:00", "08:00", "09:99", "10:00", "10:00", "11:99"],
    })

    evidence_df = pd.DataFrame({
        "flight": ["CA1", "CA1", "CA1", "CA2", "CA2", "CA2"],
        "sched_dep_time": ["08:00", "08:00", "08:00", "10:00", "10:00", "10:00"],
    })

    rules = [
        "!flight(x) v sched_dep_time(y)"
    ]

    return dirty_df, evidence_df, rules


def test_mlnclean():
    print("========================================")
    print("      Testing MLNClean Algorithm        ")
    print("========================================")

    if not check_dependencies():
        pytest.skip("MLNClean dependencies are not installed")

    dirty_df, evidence_df, rules = generate_fake_data()

    print("\n[Step 1] Detection...")
    detector = MLNCleanDetector(
        rules=rules,
        evidence_df=evidence_df,
        partition_number=1,
        agp_threshold=2,
        mcmc_samples=2,
        mcmc_warmup=2,
        verbose=True,
    )

    mask = detector.train_and_predict(dirty_df)

    assert isinstance(mask, pd.DataFrame)
    assert mask.shape == dirty_df.shape
    assert mask.dtypes.apply(lambda x: x == bool).all()

    print("\n[Step 2] Correction...")
    corrector = MLNCleanCorrector(
        rules=rules,
        evidence_df=evidence_df,
        cleaned_df=detector.cleaned_df_,
        verbose=True,
    )

    fixed_df = corrector.train_and_predict(
        dirty_df,
        detection_mask=mask,
    )

    assert isinstance(fixed_df, pd.DataFrame)
    assert fixed_df.shape == dirty_df.shape

    print("\n[Result] MLNClean Test Passed!")
    print(fixed_df)


if __name__ == "__main__":
    test_mlnclean()