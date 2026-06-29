# DataPrep

A comprehensive tool package designed for streamlined **data preparation**, cleaning, and preprocessing. This toolkit provides a robust environment for handling various data quality tasks, including integration with specialized error correction modules.

## 🛠 Installation & Environment Setup

Follow these steps to set up the environment and install the necessary dependencies. We recommend using **Conda** for environment management.

### 1. Create and Activate Environment
```bash
conda create -n dataprep python=3.10
conda activate dataprep
```

### 2. Clone the Repository
```bash
git clone [https://github.com/ZJU-Data-Governance-and-Services-Team/DataPrep.git](https://github.com/ZJU-Data-Governance-and-Services-Team/DataPrep.git)
cd DataPrep
```

### 3. Install Dependencies
Install PyTorch with CUDA 11.8 support, followed by the project-specific requirements:
```bash
# Install PyTorch ecosystem
conda install pytorch==2.2.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

## 🚀 Getting Started

### 1.Imputation
This module recovers missing values in tabular data using advanced generative models and strict paper-style imputation wrappers.

Currently integrated imputation algorithms:
- `GAIN`, `VAEGAIN`, `SCIS`
- `EDIT`
- `MISSTabular`, `MISSSaint`, `MISSTabFormer`, `MISSNPT`

### How to use:
You need a dataset with missing values and a corresponding boolean mask (where 1 indicates observed values and 0 indicates missing values).

```bash
import pandas as pd
from dataprep.tabular.imputation.GAIN import GAIN

# 1. Load Data and Mask (NumPy arrays expected)
data_missing = pd.read_csv('datasets/imputation/weather_raw.csv').values
missing_mask = pd.read_csv('datasets/imputation/weather_missing_mask.csv').values

# 2. Initialize the Model
# You can swap GAIN with VAEGAIN or SCIS
model = GAIN(
    batch_size=128, 
    hint_rate=0.9, 
    alpha=100, 
    epoch=1000,
    device='cuda' # or 'cpu'
)

# 3. Train and Predict
imputed_data = model.train_and_predict(data_missing, missing_mask)

# Save results
pd.DataFrame(imputed_data).to_csv('imputed_results.csv', index=False)
```

MISS wrappers preserve the original MISS training contract. In addition to the table with missing values and the missing mask, they require a clean/full table through `full_data` for expert sampling and IPS.

```python
import pandas as pd
from tabular.imputation.MISS_tabular import MISSTabular

data_missing = pd.read_csv("path/to/missing.csv")
data_full = pd.read_csv("path/to/full.csv")
missing_mask = data_missing.notna().astype(int).values

model = MISSTabular(epoch=10, device="cpu")
imputed = model.train_and_predict(
    data_missing,
    missing_mask=missing_mask,
    full_data=data_full,
)
```

### 2.Detection
This module identifies dirty or anomalous cells within a dataset. It includes LLM-assisted zero-shot detection (ZeroED), repair-derived detectors, and rule/prompt-based detectors.

Currently integrated detection algorithms:
- `ZeroED`
- `MLNClean`
- `IterClean`
- `Horizon`
- `SCAREd`

### How to use:
For LLM-based detection, you need to configure your API base and model name (e.g., qwen2.5-7b or gpt-3.5-turbo).

```bash
import pandas as pd
from dataprep.tabular.detection.ZeroED import ZeroED

# 1. Load Dirty Data
df_raw = pd.read_csv('datasets/detection/rayyan_dirty_100.csv')

# 2. Initialize Detector
detector = ZeroED(
        api_key='EMPTY',
        model_name="qwen2.5-7b",
        base_url="http://localhost:8000/v1",
        local_model_use=True,
        n_method='dbscan',  # 或者 'agglomerative'
        result_dir='./temp_zeroed_results',
        verbose=False
    )

# 3. Train and Predict 
error_mask = detector.train_and_predict(df_raw) # Returns a boolean matrix

# Save the generated mask
pd.DataFrame(error_mask).to_csv('generated_error_mask.csv', index=False)
```

### 3.Correction
Once errors are detected, this module repairs dirty cells. ZeroEC utilizes LLMs and local embeddings to smartly correct data based on context.

Currently integrated correction algorithms:
- `ZeroEC`
- `MLNClean`
- `IterClean`
- `TORepair`
- `Horizon`
- `SCAREd`

> ⚠️ **Prerequisite for ZeroEC:**
> The embedding-based correction requires the `all-MiniLM-L6-v2` model.
>
> 1. Visit the [ZeroEC Repository](https://github.com/YangChen32768/ZeroEC.git) or HuggingFace.
> 2. Download the `all-MiniLM-L6-v2` folder.
> 3. Place it within your local directory and point to it using the `embedding_model_path` parameter.

### How to use:
You need a dataset with missing values and a corresponding boolean mask (where 1 indicates observed values and 0 indicates missing values).

```bash
import pandas as pd
from dataprep.tabular.correction.ZeroEC import ZeroEC

# 1. Initialize Corrector
corrector = ZeroEC(
        model_name="qwen2.5-7b",
        openai_api_base="http://localhost:8000/v1",
        openai_api_key="EMPTY",
        embedding_model_path=f'tabular/correction/all-MiniLM-L6-v2',
        human_repair_num=10,
        output_dir=f'./runs_output',
        clean_data_path="datasets/detection/rayyan_clean_100.csv",
        dirty_data_path="datasets/detection/rayyan_dirty_100.csv",
        detection_path="datasets/detection/rayyan_dirty_error_detection_100.csv",
        prompt_dir=f'{BASE_DIR}/prompt_templates',
        max_workers=3
    )

# 2. Run Correction Pipeline
# The model will target only the cells flagged as True in the detection path
cleaned_df = corrector.train_and_predict()

# 3. Save Corrected Data
cleaned_df.to_csv('final_corrected_data.csv')
```

### 4.Cleaning
This module runs end-to-end table cleaning pipelines that combine detection, labeling, format repair, FD repair, semantic repair, and LLM-assisted reasoning.

Currently integrated cleaning algorithms:
- `ProClean`

### How to use:
ProClean requires local FastText and semantic model resources, plus an OpenAI-compatible LLM endpoint.

```python
from tabular.cleaning.ProClean import ProClean

cleaner = ProClean(
    dataset="example",
    dirty_path="path/to/dirty.csv",
    clean_path="path/to/clean.csv",
    result_root="results",
    llm_base_url="http://localhost:8000/v1",
    llm_api_key="EMPTY",
    llm_model="qwen2.5-7b",
    semantic_model_path="models/qwen-0.6B",
    fasttext_model_path="models/cc.en.300.bin",
)

result = cleaner.run()
```

## Current Known Issues

- **Backend dispatch is not yet wired for newly added algorithms.** The web console/backend route still needs explicit method registration for `EDIT`, `MISS-*`, `MLNClean`, `IterClean`, `TORepair`, `Horizon`, `SCAREd`, and cleaning algorithms such as `ProClean`.
- **Package layout is still transitional.** The repository mixes `dataprep...` package-style imports with direct source-tree `tabular...` imports. Fallback imports were added for the newly integrated files, but a formal package layout should still be normalized.
- **MISS algorithms require `full_data`.** This matches the original MISS-style supervised/expert-sampling flow, but it is not compatible with the old imputation API that only passes `data_missing` and `missing_mask`.
- **MISS example scripts are not normalized yet.** The original `MISS-*-test.py` scripts depend on external `miss数据集/MISS/...` paths and were not copied into the main test suite as runnable tests.
- **MLNClean requires rules and evidence data.** It expects MLN rules plus `evidence_df` or `evidence_path`; it also expects an `ID` column unless callers add one before invocation.
- **TORepair still needs complete config/data wiring.** Direct DataFrame mode requires `detection_mask` and feature metadata; default dataset-config mode expects `data/{dataset}/data_config.json`.

## TODO

- Register all newly integrated algorithms in backend dispatch and the web console method list.
- Standardize the Python package layout and remove the need for dual `dataprep...` / `tabular...` import fallbacks.
- Add minimal runnable examples for `MISS-*`, `MLNClean`, `TORepair`, and `ProClean` using local sample data.
- Add unit or smoke tests that do not require external datasets, API keys, or large model downloads.
- Document required inputs for each new algorithm, especially `full_data`, `detection_mask`, `rules_path`, `evidence_path`, and `data_config`.
- Decide whether repair-derived detection should be evaluated separately from independent detection methods.
- Pin or split heavy optional dependencies by algorithm group to avoid forcing every user to install all models/toolchains.
- Add CI checks for syntax, import smoke tests, and lightweight toy-data execution.

## 📊 Performance Comparison
As demonstrated in the benchmark results below, DataPrep's advanced algorithms consistently outperform traditional `scikit-learn` baselines across most data governance scenarios.
**Table 1: Imputation Performance Comparison (RMSE)**

| Algorithm | Weather | California | Kin8nm |
| :--- | :---: | :---: | :---: |
| **MICE** | 12.2531 | 546.3955 | 1.1907 |
| **MissF** | 10.9410 | 395.1089 | 0.9244 |
| **GAIN** | 10.5331 | 411.5608 | 0.9722 |
| **VAE-GAIN** | 10.2109 | 406.7808 | 0.8606 |
| **SCIS-GAIN** | 10.2650 | 389.6532 | 0.8904 |
> *Note: The values above represent the RMSE (Root Mean Square Error) performance of each algorithm on the corresponding datasets.*

<br>

**Table 2: Error Detection Performance Comparison**

| Algorithm | Metric | Flights | Tax | Rayyan |
| :--- | :--- | :---: | :---: | :---: |
| **ZeroED (LLM)** | F1-Score | 0.8392 | 0.6762 | 0.8559 |
| | Precision | 0.8535 | 0.5236 | 0.7725 |
| | Recall | 0.8277 | 0.9573 | 0.9608 |
| **LOF** | F1-Score | 0.6332 | 0.5015 | 0.6028 |
| | Precision | 0.8241 | 0.5133 | 0.768 |
| | Recall | 0.5142 | 0.4902 | 0.4961 |
| **Isolation Forest**| F1-Score | 0.5755 | 0.4793 | 0.6279 |
| | Precision | 0.7489 | 0.4907 | 0.8 |
| | Recall | 0.4673 | 0.4685 | 0.5168 |
> *Note: The values above represent the F1-Score, Precision, and Recall performance of each detection algorithm on the corresponding datasets.*

<br>

**Table 3: Data Correction Performance Comparison**

| Algorithm | Rayyan | Flights | Tax |
| :--- | :---: | :---: | :---: |
| **Sklearn (Mode)** | 0 | 0.0152 | 0 |
| **Sklearn (KNN)** | 0 | 0.0543 | 0 |
| **Sklearn (Iterative)** | 0 | 0.0467 | 0 |
| **ZeroEC (LLM)** | 0.5468 | 0.9779 | 0.2716 |
> *Note: Accuracy represents the proportion of correctly repaired cells strictly among those flagged as errors by the detection mask. A higher value indicates a stronger capability to restore dirty data to its ground truth.*

## 🖥️ Interactive Web Console

If you prefer to use DataPrep via an interactive graphical interface, please refer to the [DataPrep Console User Manual](./DataPrep%20Console%20User%20Manual.md) for detailed setup and usage instructions.
