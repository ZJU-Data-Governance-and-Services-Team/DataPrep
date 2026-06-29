# DataPrep Algorithms：Horizon 与 SCAREd

本仓库在 DataPrep 框架中扩展实现了两个表格数据质量算法：**Horizon** 与 **SCAREd**，并提供了可复现实验脚本，用于表格数据的 **correction（数据修复）** 与 **detection（错误检测）** 任务。

当前实现采用 **repair-first** 设计思路：

- `tabular/correction`：生成修复后的数据表；
- `tabular/detection`：生成单元格级错误位置矩阵；
- `examples`：提供可直接运行的实验入口，并自动保存指标汇总、修复/检测明细、mask 文件与 debug 信息。

该实现主要面向表格数据清洗实验，并在 `hospital_auto` 数据集上提供端到端复现流程。

---

## 功能特性

- 提供符合 DataPrep 风格的 Horizon 与 SCAREd 封装接口；
- 分离 correction 与 detection 两类任务接口；
- 支持 DataFrame 形式的输入与输出；
- 提供可复现的命令行实验脚本；
- 自动保存多类实验结果文件，包括：
  - 修复后的数据；
  - 错误位置 mask；
  - 被修改/被检测单元格明细；
  - 指标汇总表；
  - debug 诊断信息；
- SCAREd 支持通过 `repair_attrs` 和 `apply_only_detected` 控制修复范围；
- Horizon 支持 FD 规则自动发现与规则文件复用。

---

## 项目结构

```text
DataPrep-main/
├── datasets/
│   └── hospital_auto/
│       ├── dirty.csv
│       ├── clean.csv
│       └── constraints_auto.txt
├── examples/
│   ├── correction_horizon.py
│   ├── detection_horizon.py
│   ├── correction_scared.py
│   └── detection_scared.py
└── tabular/
    ├── correction/
    │   ├── Horizon.py
    │   ├── Horizon_modules.py
    │   ├── SCAREd.py
    │   └── SCAREd_modules.py
    └── detection/
        ├── Horizon.py
        ├── Horizon_modules.py
        ├── SCAREd.py
        └── SCAREd_modules.py
```

---

## 算法接口

| Algorithm | Correction module | Detection module | Correction output | Detection output |
|---|---|---|---|---|
| Horizon | `tabular/correction/Horizon.py` | `tabular/detection/Horizon.py` | `repaired_df` | `error_mask` |
| SCAREd | `tabular/correction/SCAREd.py` | `tabular/detection/SCAREd.py` | `repaired_df` | `error_mask` |

Horizon 与 SCAREd 均按 repair-first 方式实现：correction 模块先生成修复后的数据表，detection 模块再通过修复结果与原始脏数据之间的逐单元格差异得到错误位置矩阵：

```text
error_mask = repaired_df != dirty_df
```

这种设计能够保证 correction 与 detection 两端逻辑一致，同时保持 DataPrep 风格的模块化结构。

---

## 环境安装

推荐使用 Python 3.8 及以上版本。

```bash
conda create -n dataprep python=3.8
conda activate dataprep

cd DataPrep-main
pip install -r requirements.txt
pip install pandas numpy scikit-learn scipy openpyxl
```

后续命令默认在项目根目录下执行。

---

## 数据集说明

默认示例数据集为：

```text
datasets/hospital_auto/dirty.csv
datasets/hospital_auto/clean.csv
```

| File | Description |
|---|---|
| `dirty.csv` | 含有噪声或错误单元格的输入数据表 |
| `clean.csv` | 用于实验评价的干净数据表 |
| `constraints_auto.txt` | Horizon 使用的 FD 规则文件，可自动生成或复用 |

---

## Horizon 使用方法

### Horizon Correction

```powershell
python examples/correction_horizon.py `
  --clean_path datasets/hospital_auto/clean.csv `
  --dirty_path datasets/hospital_auto/dirty.csv `
  --output_dir temp/week2_results/horizon_correction
```

主要输出文件：

| Output file | Description |
|---|---|
| `repaired_horizon.csv` | Horizon 生成的修复后数据 |
| `horizon_result.csv` | correction 指标汇总 |
| `horizon_changed_detail.csv` | 单元格级修改明细 |
| `constraints_auto.txt` | 自动生成或复用的 FD 规则文件 |

### Horizon Detection

```powershell
python examples/detection_horizon.py `
  --dirty_path datasets/hospital_auto/dirty.csv `
  --clean_path datasets/hospital_auto/clean.csv `
  --rule_path temp/week2_results/horizon_correction/constraints_auto.txt `
  --output_dir temp/week2_results/horizon_detection
```

主要输出文件：

| Output file | Description |
|---|---|
| `error_mask_horizon.csv` | 单元格级布尔错误矩阵 |
| `horizon_detection_result.csv` | detection 指标汇总 |
| `horizon_detected_detail.csv` | 被检测为错误的单元格明细 |

---

## SCAREd 使用方法

### 关键参数

| Parameter | Description |
|---|---|
| `--reliable_attrs` | 指定可靠属性，用于分区和局部模型输入，例如 `provider_number` |
| `--repair_attrs` | 指定允许被修复的字段集合 |
| `--apply_only_detected` | 仅修复 detection mask 标记的位置 |
| `--use_perfect_detection_mask` | 基于 `clean.csv` 构造受控验证用的 perfect detection mask |
| `--use_clean_in_model` | 在受控验证流程中将 `clean_df` 传入内部工作流 |
| `--detection_mask_path` | 读取外部 detection mask，例如其他检测器输出的 mask |

### 为什么使用 `repair_attrs`

SCAREd 的原始思路中包含 reliable attributes 与 flexible attributes 的划分。重构实现中通过 `repair_attrs` 显式暴露修复范围，使用户能够将预测与修复限制在指定字段子集上，从而减少候选空间膨胀和过度修复风险。

在默认的 `hospital_auto` 实验中，使用以下字段作为可修复字段：

```text
city, county, state, zip, phone, address_1, emergency_service
```

这些字段主要属于地址、区域和联系方式相关属性，字段间关联较强，也与数据集中常见的 FD 风格依赖关系相匹配，例如地址到城市、地址到州、地址到邮编等关系。将修复范围限制在这些字段上，有助于在 flexible attributes 较多时降低候选预测复杂度，并减少不必要的误修。

---

## SCAREd Correction：`repair_attrs`

```powershell
python examples/correction_scared.py `
  --dirty_path datasets/hospital_auto/dirty.csv `
  --clean_path datasets/hospital_auto/clean.csv `
  --use_perfect_detection_mask `
  --use_clean_in_model `
  --reliable_attrs provider_number `
  --repair_attrs city,county,state,zip,phone,address_1,emergency_service `
  --output_dir temp/week2_results/scared_correction_repair_attrs
```

主要输出文件：

| Output file | Description |
|---|---|
| `repaired_scared.csv` | SCAREd 修复后的数据表 |
| `scared_correction_result.csv` | correction 指标汇总 |
| `scared_changed_detail.csv` | 被修改单元格明细 |
| `scared_repair_mask.csv` | 布尔修复位置矩阵 |
| `scared_used_detection_mask.csv` | 本次运行使用的 detection mask |
| `scared_debug_info.csv` | debug 信息，包括可靠属性、可修复字段、候选数量等 |

---

## SCAREd Detection：`repair_attrs`

```powershell
python examples/detection_scared.py `
  --dirty_path datasets/hospital_auto/dirty.csv `
  --clean_path datasets/hospital_auto/clean.csv `
  --use_perfect_detection_mask `
  --use_clean_in_model `
  --reliable_attrs provider_number `
  --repair_attrs city,county,state,zip,phone,address_1,emergency_service `
  --output_dir temp/week2_results/scared_detection_repair_attrs
```

主要输出文件：

| Output file | Description |
|---|---|
| `scared_detection_mask.csv` | 布尔错误检测矩阵 |
| `scared_detection_result.csv` | detection 指标汇总 |
| `scared_detected_detail.csv` | 被检测为错误的单元格明细 |
| `scared_detection_repaired.csv` | detection 过程中生成的修复数据 |
| `scared_used_detection_mask.csv` | 本次运行使用的 detection mask |
| `scared_detection_debug_info.csv` | detection 端 debug 信息 |

---

## SCAREd Correction：`apply_only_detected + repair_attrs`

```powershell
python examples/correction_scared.py `
  --dirty_path datasets/hospital_auto/dirty.csv `
  --clean_path datasets/hospital_auto/clean.csv `
  --use_perfect_detection_mask `
  --use_clean_in_model `
  --apply_only_detected `
  --reliable_attrs provider_number `
  --repair_attrs city,county,state,zip,phone,address_1,emergency_service `
  --output_dir temp/week2_results/scared_correction_apply_only_detected_repair_attrs
```

该设置将 SCAREd 的修复范围进一步限制为 detection mask 中标记的位置，适用于减少过度修复并避免修改未被判定为可疑的单元格。

---

## 一键复现实验

以下 PowerShell 脚本可从零开始运行主要 Horizon 与 SCAREd 实验：

```powershell
$ErrorActionPreference = "Stop"

cd D:\ZJU\DataPrep-main-0609\DataPrep-main

Remove-Item -Recurse -Force temp/week2_results -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force temp/week2_results | Out-Null

python examples/correction_horizon.py `
  --clean_path datasets/hospital_auto/clean.csv `
  --dirty_path datasets/hospital_auto/dirty.csv `
  --output_dir temp/week2_results/horizon_correction

python examples/detection_horizon.py `
  --dirty_path datasets/hospital_auto/dirty.csv `
  --clean_path datasets/hospital_auto/clean.csv `
  --rule_path temp/week2_results/horizon_correction/constraints_auto.txt `
  --output_dir temp/week2_results/horizon_detection

python examples/correction_scared.py `
  --dirty_path datasets/hospital_auto/dirty.csv `
  --clean_path datasets/hospital_auto/clean.csv `
  --use_perfect_detection_mask `
  --use_clean_in_model `
  --reliable_attrs provider_number `
  --repair_attrs city,county,state,zip,phone,address_1,emergency_service `
  --output_dir temp/week2_results/scared_correction_repair_attrs

python examples/detection_scared.py `
  --dirty_path datasets/hospital_auto/dirty.csv `
  --clean_path datasets/hospital_auto/clean.csv `
  --use_perfect_detection_mask `
  --use_clean_in_model `
  --reliable_attrs provider_number `
  --repair_attrs city,county,state,zip,phone,address_1,emergency_service `
  --output_dir temp/week2_results/scared_detection_repair_attrs

python examples/correction_scared.py `
  --dirty_path datasets/hospital_auto/dirty.csv `
  --clean_path datasets/hospital_auto/clean.csv `
  --use_perfect_detection_mask `
  --use_clean_in_model `
  --apply_only_detected `
  --reliable_attrs provider_number `
  --repair_attrs city,county,state,zip,phone,address_1,emergency_service `
  --output_dir temp/week2_results/scared_correction_apply_only_detected_repair_attrs
```

---

## `hospital_auto` 示例结果

以下结果来自 `hospital_auto` 数据集上的受控验证实验：

| Experiment | Precision | Recall | F1 | Fixed / Total | Changed Cells | New Errors |
|---|---:|---:|---:|---:|---:|---:|
| SCAREd Correction + `repair_attrs` | 79.91% | 36.74% | 50.34% | 187 / 509 | 234 | 45 |
| SCAREd Detection + `repair_attrs` | 80.77% | 37.13% | 50.87% | TP = 189 | Detected = 234 | FP = 45 |
| SCAREd Correction + `apply_only_detected + repair_attrs` | 98.94% | 36.74% | 53.58% | 187 / 509 | 189 | 0 |

结果解读：

- `repair_attrs` 通过限制可修复字段，提高了修复过程的可控性；
- `apply_only_detected` 进一步将修复范围限制在检测到的单元格上，有助于降低过度修复；
- 在上述受控设置中，`apply_only_detected + repair_attrs` 在修复 187 个真实错误单元格的同时，将新增错误数控制为 0。

---

## 受控验证与真实使用场景

部分示例使用以下参数：

```text
--use_perfect_detection_mask
--use_clean_in_model
```

这些参数用于受控验证场景：当 `clean.csv` 可用时，可以构造 perfect detection mask，或者用于分析修复行为。这类结果主要用于验证模块功能、修复范围控制和结果输出流程，不应直接等同于完全无监督的真实部署效果。

在真实使用场景中，如果不存在 ground-truth clean table，可以考虑：

1. 不使用 `--use_clean_in_model`；
2. 通过 `--detection_mask_path` 传入外部检测器生成的 mask；
3. 使用 Horizon 或其他检测方法生成 detection mask；
4. 在更多数据集上重新评估 `repair_attrs` 与 `reliable_attrs` 的选择。

---

## 输出文件说明

| File type | Example filename | Description |
|---|---|---|
| 指标汇总 | `*_result.csv` | precision、recall、F1、changed cells、runtime 等指标 |
| 修复数据 | `repaired_*.csv` | 最终修复后的数据表 |
| 错误/修复 mask | `*_mask.csv` | detected 或 repaired cells 的布尔矩阵 |
| 单元格明细 | `*_detail.csv` | 行列级别的修改或检测明细 |
| Debug 信息 | `*_debug_info.csv` | 属性选择、候选数量、运行诊断等信息 |
| FD 规则 | `constraints_auto.txt` | Horizon 使用的 FD 规则文件 |

---

## 快速检查

可以运行以下命令确认脚本接口可用：

```powershell
python examples/correction_scared.py -h
python examples/detection_scared.py -h
python examples/correction_horizon.py -h
python examples/detection_horizon.py -h
```

SCAREd 的帮助信息中应包含：

```text
--repair_attrs
--use_perfect_detection_mask
--use_clean_in_model
--detection_mask_path
```

---

## Notes

- Horizon 使用 FD 风格约束，并支持自动生成或复用 FD 规则文件；
- SCAREd 是 repair-first 方法，detection 结果由修复后的 changed cells 推导得到；
- SCAREd detection 侧复用 correction 侧候选生成逻辑，以避免两套实现产生不一致；
- `repair_attrs` 是用于显式限制修复范围的工程控制参数；
- `debug_info` 文件用于提高实验过程的可解释性和可复现性。
