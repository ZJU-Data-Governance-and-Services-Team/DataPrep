import os
import sys
import torch.nn as nn

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import random
import pandas as pd
import numpy as np
import itertools
import ast
import argparse
from utils.utility import load_config_json, split_dataframe, identify_complete_and_error_samples, split_dataframe_with_mask, identify_complete_and_error_samples_with_mask
from torch.utils.data import DataLoader
from models.diffae.core import TabularAE, TabularUNet, ResidualMLP, DiffusionUtils, train_embedding_direct
from models.bilevel_beta.training import BiLevelTrainer
from models.baselines.model import run_baseline_experiments
from datetime import datetime
from datasets.dataset import TabularDataset
from datasets.data_processor import DataProcessor, calculate_missing_rate
from models.imputation.standard_pipeline import combine_features, train_classifier_on_repaired_data, evaluate_classifier
from utils.error_detection import ErrorDetector, calculate_error_rate
from utils.feature_embedder import FeatureEmbedder
from utils.constants import TRAIN_DATA_MASK_STR, VAL_DATA_MASK_STR, TEST_DATA_MASK_STR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logging(args, results_dir="results"):
    """设置日志和结果保存目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{results_dir}/bilevel_{args.dataset}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    # 保存参数
    with open(f"{run_dir}/args.txt", "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    return run_dir


def preprocess_data(args, data_config):
    """
    数据预处理和划分
    
    Args:
        args: 命令行参数
        data_config: 数据配置
    
    Returns:
        处理器, 训练/验证/测试集, 训练集完整/缺失部分, 特征维度等信息, 以及相应的错误掩码
    """
    numerical_features = data_config["numerical_features"]
    categorical_features = data_config["categorical_features"]
    target_feature = data_config["target_feature"]
    
    df_path = data_config["data_path"]
    df = pd.read_csv(df_path)
    
    # 定义mask文件路径
    det_methods = "_".join(args.detection_methods) if not args.only_det_missing else "missing_values"
    mask_file_path = f"data/{args.dataset}/error_mask_{det_methods}.csv"

    full_error_mask = None
    error_detector = None
    if os.path.exists(mask_file_path):
        print(f"检测到预计算的错误掩码文件: {mask_file_path}，正在加载...")
        full_error_mask = pd.read_csv(mask_file_path, dtype=bool)
        print("错误掩码加载完成")
        print(f"原始错误率 (基于加载的掩码):")
        # for col, rate in full_error_result["column_rates"].items():
        #     print(f"  {col}: {rate:.4f}")
        # print(f"总体: {full_error_result['overall_rate']:.4f}")

    if full_error_mask is None: # 如果没有加载到mask，则进行计算并保存
        print("未检测到预计算的错误掩码文件，正在计算...")
        if args.only_det_missing:
            full_error_result = calculate_missing_rate(df)
            full_error_mask = full_error_result['error_mask']
            print(f"原始缺失率:")
            for col, rate in full_error_result["column_rates"].items():
                print(f"  {col}: {rate:.4f}")
            print(f"总体: {full_error_result['overall_rate']:.4f}")
        else:
            error_detector = ErrorDetector(
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                target_feature=target_feature,
                detection_methods=args.detection_methods,
            )
            error_detector.fit(df)
            full_error_result = calculate_error_rate(df, error_detector, dataset_name="original")
            full_error_mask = full_error_result['error_mask']
            print(f"原始错误率:")
            for col, rate in full_error_result["column_rates"].items():
                print(f"  {col}: {rate:.4f}")
            print(f"总体: {full_error_result['overall_rate']:.4f}")
        
        # 保存新计算的mask
        pd.DataFrame(full_error_mask).to_csv(mask_file_path, index=False)
        print(f"新计算的错误掩码已保存到: {mask_file_path}")

    # 使用新的分割方法同时分割数据和掩码
    (train_val_df, train_val_mask), (test_df, test_mask) = split_dataframe_with_mask(
        df, full_error_mask,
        test_size=args.test_split, 
        random_state=args.seed, 
        stratify_column=target_feature[0] if len(target_feature) > 0 else None
    )
    
    (train_df, train_mask), (val_df, val_mask) = split_dataframe_with_mask(
        train_val_df, train_val_mask,
        test_size=args.val_split / (1 - args.test_split), 
        random_state=args.seed, 
        stratify_column=target_feature[0] if len(target_feature) > 0 else None
    )
    
    # 初始化处理器
    processor = DataProcessor(
        numerical_features=numerical_features, 
        categorical_features=categorical_features,
        target_feature=target_feature,
        only_det_missing=args.only_det_missing,
        error_detector=error_detector,
        args=args
    )
    processor.fit(train_df)
    
    # 将切分好的错误掩码缓存到processor中
    print("缓存预计算的错误掩码...")
    processor.set_error_mask_cache('train', train_mask, train_df)
    processor.set_error_mask_cache('val', val_mask, val_df)
    processor.set_error_mask_cache('test', test_mask, test_df)
    print("错误掩码缓存完成")
    
    # 将error_detector设置到args中，以便兼容旧接口
    args.error_detector = error_detector
    
    # 使用新的方法区分完整样本和缺失样本（带掩码）
    (train_df_complete, train_complete_mask), (train_df_error, train_error_mask) = identify_complete_and_error_samples_with_mask(train_df, train_mask)
    
    print(f"数据集划分信息:")
    print(f"  训练集大小: {len(train_df)}")
    print(f"    完整样本: {len(train_df_complete)}")
    print(f"    缺失样本: {len(train_df_error)}")
    print(f"  验证集大小: {len(val_df)}")
    print(f"  测试集大小: {len(test_df)}")
    
    # 计算特征维度
    d_numerical = len(numerical_features)
    d_categorical = len(categorical_features)
    actual_cat_sizes = processor.categories  # Include "MASK"
    
    # 获取标签数量
    num_classes = len(processor.label_encoder)
    print(f"目标类别数量: {num_classes}")
    
    return (processor, train_df, val_df, test_df, 
            train_df_complete, train_df_error,
            d_numerical, d_categorical, actual_cat_sizes,
            num_classes, error_detector,
            train_mask, val_mask, test_mask,
            train_complete_mask, train_error_mask)

def set_seed(seed):
    # 设置随机种子确保实验可重复性
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保CUDA操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main(args):
    """
    Bi-Level优化主函数
    """
    # 解析detection_methods参数
    if isinstance(args.detection_methods, str):
        try:
            args.detection_methods = ast.literal_eval(args.detection_methods)
        except (ValueError, SyntaxError) as e:
            print(f"错误：{e}")
            print("【警告】使用默认值：['semantic_isolation_forest', 'missing_values']")
            args.detection_methods = ['semantic_isolation_forest', 'missing_values']

    print("\n所有参数:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    set_seed(args.seed)
    
    # 加载数据配置
    data_config_path = f"data/{args.dataset}/data_config.json"
    data_config = load_config_json(data_config_path)
    
    numerical_features = data_config["numerical_features"]
    categorical_features = data_config["categorical_features"]
    
    (processor, train_df, val_df, test_df, 
     train_df_complete, train_df_error,
     d_numerical, d_categorical, actual_cat_sizes,
     num_classes, error_detector,
     train_mask, val_mask, test_mask,
     train_complete_mask, train_error_mask) = preprocess_data(args, data_config)
    
    feature_embedder = FeatureEmbedder(
        d_numerical=d_numerical,
        d_categorical=d_categorical,
        actual_cat_sizes=actual_cat_sizes,
        d_embed=args.embed_dim,
        dropout=args.dropout,
        device=args.device
    ).to(args.device)
    
    # 运行基准实验
    # baseline_results = run_baseline_experiments(
    #     args=args,
    #     processor=processor,
    #     train_df=train_df,
    #     val_df=val_df,
    #     test_df=test_df,
    #     numerical_features=numerical_features,
    #     categorical_features=categorical_features,
    #     train_mask=train_mask,
    #     val_mask=val_mask,
    #     test_mask=test_mask
    # )
    
    print("\n" + "="*50)
    print("预训练TabularDiffAE填补模型")
    print("="*50)
    
    # 创建扩散工具类
    diffusion_utils = DiffusionUtils(
        num_timesteps=args.num_timesteps,
        schedule=args.schedule,
        device=args.device
    )
    
    # 创建训练数据加载器
    train_data = TabularDataset(train_df, processor, dataset_name=TRAIN_DATA_MASK_STR, error_mask_df=train_mask)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    
    # 创建TabularDiffAE模型
    single_channel_dim = (d_numerical + d_categorical) * args.embed_dim
    
    # 根据diffusion_embed参数决定输入维度
    total_dim = single_channel_dim * 2
    embedding_dim = single_channel_dim
    predict_noise = True if (args.diffusion_embed and args.use_diffusion_loss and (args.loss_data == 'bilevel_diff_val' or args.loss_data == 'diffusion_only')) else False
    
    print(f"创建TabularDiffAE模型: input_dim={total_dim}, hidden_dim={args.hidden_dim}")
    print(f"扩散模式: {'嵌入空间' if args.diffusion_embed else '原始数据空间'}")

    if args.repair_model == "ae":
        repair_model = TabularAE(
            input_dim=total_dim,
            time_emb_dim=args.time_emb_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            d_numerical=d_numerical,
            d_categorical=d_categorical,
            actual_cat_sizes=actual_cat_sizes,
            dropout=args.dropout,
            num_classes=0,  # 不使用标签条件
            predict_noise=predict_noise,
            embedding_dim=embedding_dim
        ).to(args.device)
    elif args.repair_model == "mlp":
        repair_model = ResidualMLP(
            input_dim=total_dim,
            num_layers=args.num_layers,
            time_emb_dim=args.time_emb_dim,
            hidden_dim=args.hidden_dim,
            d_numerical=d_numerical,
            d_categorical=d_categorical,
            actual_cat_sizes=actual_cat_sizes,
            dropout=args.dropout,
            predict_noise=predict_noise,
            embedding_dim=embedding_dim,
            num_classes=0,
        ).to(args.device)
    else:
        repair_model = TabularUNet(
            input_dim=total_dim,
            time_emb_dim=args.time_emb_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            d_numerical=d_numerical,
            d_categorical=d_categorical,
            actual_cat_sizes=actual_cat_sizes,
            dropout=args.dropout,
            predict_noise=predict_noise,
            embedding_dim=embedding_dim
        ).to(args.device)
    
    # # 训练TabularDiffAE模型
    # optimizer = torch.optim.Adam(
    #     itertools.chain(repair_model.parameters(), feature_embedder.parameters()), 
    #     lr=args.diff_pretrain_lr
    # )
    
    # if args.use_diffusion:
    #     from models.embedding_diffusion import train_embedding_diffae
    #     train_embedding_diffae(
    #         model=repair_model,
    #         feature_embedder=feature_embedder,
    #         diffusion=diffusion_utils,
    #         dataloader=train_dataloader,
    #         optimizer=optimizer,
    #         device=args.device,
    #         epochs=0,
    #         actual_cat_sizes=actual_cat_sizes,
    #         log_interval=10
    #     )
    # else:
    # train_embedding_direct(
    #     model=repair_model,
    #     feature_embedder=feature_embedder,
    #     dataloader=train_dataloader,
    #     optimizer=optimizer,
    #     device=args.device,
    #     epochs=args.diff_pretrain_epochs,
    #     label_mask_rate=1.0,
    #     log_interval=10
    # )
    
    # print("TabularDiffAE填补模型预训练完成")

    # # 使用预训练的TabularDiffAE模型直接填补所有数据
    # def rep_with_pretrained_model(df, df_name, dataset_name=None):
    #     """使用预训练模型填补数据的通用函数"""
    #     print(f"使用预训练模型填补{df_name}数据...")
    #     from models.imputation.repair_embed import unified_imputation
    #     if args.use_diffusion:
    #         return unified_imputation(
    #             df=df,
    #             model=repair_model,
    #             feature_embedder=feature_embedder,
    #             diffusion_utils=diffusion_utils,
    #             method="diffusion",
    #             processor=processor,
    #             num_steps=args.num_timesteps,
    #             batch_size=len(df),
    #             device=args.device,
    #             verbose=False,
    #             use_masked_labels=True,
    #             masked_label_value=num_classes,
    #             dataset_name=dataset_name
    #         )
    #     else:
    #         return unified_imputation(
    #             df=df,
    #             model=repair_model,
    #             feature_embedder=feature_embedder,
    #             method="direct",
    #             processor=processor,
    #             batch_size=len(df),
    #             device=args.device,
    #             verbose=False,
    #             use_masked_labels=True,
    #             masked_label_value=num_classes,
    #             dataset_name=dataset_name
    #         )
    
    # # 批量填补所有数据集
    # train_df_pretrained_imp = rep_with_pretrained_model(train_df, "训练", TRAIN_DATA_MASK_STR)
    # val_df_pretrained_imp = rep_with_pretrained_model(val_df, "验证", VAL_DATA_MASK_STR)
    # test_df_pretrained_imp = rep_with_pretrained_model(test_df, "测试", TEST_DATA_MASK_STR)
    
    # def prepare_data_for_classifier(df_imp, dataset_name=None):
    #     """准备数据用于分类器训练/评估的通用函数"""
    #     x_num, x_cat_onehot = processor.transform_onehot(df_imp, dataset_name=dataset_name)
    #     x_combined = combine_features(x_num, x_cat_onehot).to(args.device)
        
    #     target_col = processor.target_feature[0]
    #     target_values = df_imp[target_col].values
    #     if isinstance(target_values[0], (list, np.ndarray)):
    #         target_values = np.array([val[0] if isinstance(val, (list, np.ndarray)) else val for val in target_values])
    #     encoded_labels = np.array([processor.label_encoder.get(val, 0) for val in target_values])
    #     y = torch.LongTensor(encoded_labels).to(args.device)
        
    #     return x_combined, y
    
    # # 准备所有数据集
    # print("\n" + "="*50)
    # print("对比实验：直接使用预训练的TabularDiffAE填补模型")
    # print("="*50)
    # print("基于填补后的数据训练分类器...")
    
    # x_combined_train, y_train = prepare_data_for_classifier(train_df_pretrained_imp, TRAIN_DATA_MASK_STR)
    # x_combined_val, y_val = prepare_data_for_classifier(val_df_pretrained_imp, VAL_DATA_MASK_STR)
    # x_combined_test, y_test = prepare_data_for_classifier(test_df_pretrained_imp, TEST_DATA_MASK_STR)
    
    # # 训练分类器
    # input_dim = x_combined_train.shape[1]
    # output_dim = len(processor.label_encoder)
    
    # print(f"训练分类器: input_dim={input_dim}, hidden_dim={args.classi_hidden_dim}, output_dim={output_dim}")
    # print(f"训练数据集大小: {len(train_df_pretrained_imp)}")
    
    # pretrained_imp_classifier, _ = train_classifier_on_repaired_data(
    #     x_combined=x_combined_train,
    #     y=y_train,
    #     input_dim=input_dim,
    #     output_dim=output_dim,
    #     hidden_dim=args.classi_hidden_dim,
    #     epochs=args.classi_epochs,
    #     batch_size=args.batch_size,
    #     lr=args.classi_lr,
    #     device=args.device,
    #     l2_lambda=args.l2_lambda_classifier,
    #     validation_split=0.0,  # 不从训练集中划分验证集
    #     patience=args.patience,
    #     x_val=x_combined_val,  # 使用外部验证集
    #     y_val=y_val
    # )
    
    # # 在测试集上评估预训练修复+分类器的方法
    # print("评估预训练修复+分类器方法...")
    # pretrained_imp_metrics = evaluate_classifier(
    #     classifier=pretrained_imp_classifier,
    #     x_combined=x_combined_test,
    #     y=y_test,
    #     classifier_criterion=nn.CrossEntropyLoss(),
    #     prefix="预训练修复+分类器 (测试集) "
    # )
    
    # print("预训练修复+分类器基线方法评估完成")
    # print(f"  测试准确率: {pretrained_imp_metrics['accuracy']:.4f}")
    # print(f"  测试F1分数: {pretrained_imp_metrics['f1']:.4f}")

    # 创建BiLevelTrainer
    print("\n" + "="*50)
    print("初始化Bi-Level训练器")
    print("="*50)

    print(f"创建{args.repair_model.upper()}修复模型: input_dim={total_dim}, hidden_dim={args.hidden_dim}")
    print(f"扩散模式: {'嵌入空间' if args.diffusion_embed else '原始数据空间'}")

    if args.repair_model == "ae":
        repair_model = TabularAE(
            input_dim=total_dim,
            time_emb_dim=args.time_emb_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            d_numerical=d_numerical,
            d_categorical=d_categorical,
            actual_cat_sizes=actual_cat_sizes,
            dropout=args.dropout,
            num_classes=0,  # 不使用标签条件
            predict_noise=predict_noise,
            embedding_dim=embedding_dim
        ).to(args.device)
    elif args.repair_model == "mlp":
        repair_model = ResidualMLP(
            input_dim=total_dim,
            num_layers=args.num_layers,
            time_emb_dim=args.time_emb_dim,
            hidden_dim=args.hidden_dim,
            d_numerical=d_numerical,
            d_categorical=d_categorical,
            actual_cat_sizes=actual_cat_sizes,
            dropout=args.dropout,
            predict_noise=predict_noise,
            embedding_dim=embedding_dim,
            num_classes=0,
        ).to(args.device)
    else:
        repair_model = TabularUNet(
            input_dim=total_dim,
            time_emb_dim=args.time_emb_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            d_numerical=d_numerical,
            d_categorical=d_categorical,
            actual_cat_sizes=actual_cat_sizes,
            dropout=args.dropout,
            predict_noise=predict_noise,
            embedding_dim=embedding_dim
        ).to(args.device)
    feature_embedder = FeatureEmbedder(
        d_numerical=d_numerical,
        d_categorical=d_categorical,
        actual_cat_sizes=actual_cat_sizes,
        d_embed=args.embed_dim,
        dropout=args.dropout,
        device=args.device
    ).to(args.device)
    
    bilevel_trainer = BiLevelTrainer(
        args=args,
        processor=processor,
        d_numerical=d_numerical,
        d_categorical=d_categorical,
        actual_cat_sizes=actual_cat_sizes,
        num_classes=num_classes,
        device=args.device,
        repair_model=repair_model,
        diffusion_utils=diffusion_utils,
        feature_embedder=feature_embedder
    )
    
    # 运行bi-level优化
    print("\n" + "="*50)
    print("运行Bi-Level优化")
    print("="*50)
    
    bilevel_results = bilevel_trainer.run_bilevel_optimization(
        train_df_clean=train_df_complete,
        train_df_error=train_df_error,
        val_df=val_df,
        test_df=test_df,
        train_clean_mask=train_complete_mask,
        train_error_mask=train_error_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    # 打印四个结果的对比表格
    print("\n" + "="*80)
    print("完整实验结果对比")
    print("="*80)
    
    # 整理所有方法的结果
    all_methods = {}
    
    # # 1. 基准实验结果
    # for method_name, metrics in baseline_results.items():
    #     display_name = "占位符修复" if "Placeholder" in method_name else "均值/众数修复"
    #     all_methods[display_name] = metrics
    
    # # 2. 预训练修复结果
    # all_methods["预训练修复+分类器"] = pretrained_imp_metrics
    
    # 3. Bi-Level优化结果
    if 'final_test_metrics' in bilevel_results:
        all_methods["Bi-Level优化"] = bilevel_results['final_test_metrics']
    else:
        # 如果没有final_test_metrics，使用最后一次迭代的结果
        if bilevel_results['iterations']:
            last_iteration = bilevel_results['iterations'][-1]
            all_methods["Bi-Level优化"] = {
                'loss': last_iteration['test_loss'],
                'accuracy': last_iteration['test_accuracy'],
                'f1': last_iteration['test_f1']
            }
        else:
            all_methods["Bi-Level优化"] = {'loss': float('nan'), 'accuracy': float('nan'), 'f1': float('nan')}
    
    # 显示Bi-Level优化的迭代过程摘要
    if bilevel_results['iterations']:
        print(f"\nBi-Level优化迭代过程摘要:")
        print(f"  总迭代次数: {len(bilevel_results['iterations'])}")
        print(f"  最佳迭代: {bilevel_results.get('best_iteration', 'N/A')}")
        print(f"  最佳验证准确率: {bilevel_results.get('best_val_accuracy', 'N/A'):.4f}")
        
        print(f"\n  各迭代表现:")
        print(f"  {'迭代':<6} | {'测试损失':<10} | {'测试准确率':<12} | {'测试F1':<10} | {'验证损失':<10} | {'验证准确率':<12} | {'验证F1':<10}")
        print("  " + "-" * 80)  # 调整分隔线长度
        
        for iteration in bilevel_results['iterations']:  # 只显示前10次迭代
            print(f"  {iteration['iteration']:<6} | {iteration['test_loss']:<10.4f} | "
                  f"{iteration['test_accuracy']:<12.4f} | {iteration['test_f1']:<10.4f} | "
                  f"{iteration.get('val_loss', float('nan')):<10.4f} | "
                  f"{iteration.get('val_accuracy', float('nan')):<12.4f} | "
                  f"{iteration.get('val_f1', float('nan')):<10.4f}")


    # 打印对比表格
    print("-" * 110)
    metric_names = {
        'loss': '损失',
        'accuracy': '准确率', 
        'f1': 'F1分数'
    }
    
    # Calculate column widths
    metric_col_width = max(len('指标'), max(len(name) for name in metric_names.values()))
    
    method_col_widths = {}
    for method_name in all_methods.keys():
        max_val_len = 0
        for metric in ['loss', 'accuracy', 'f1']:
            value = all_methods[method_name].get(metric, float('nan'))
            if not np.isnan(value):
                max_val_len = max(max_val_len, len(f"{value:.4f}"))
            else:
                max_val_len = max(max_val_len, len("N/A"))
        method_col_widths[method_name] = max(len(method_name), max_val_len)

    # Total width for separator
    total_width = metric_col_width
    for width in method_col_widths.values():
        total_width += width + 3 # +3 for " | " (space, pipe, space)

    print("-" * total_width)
    
    # 表头
    header = f"{'指标':<{metric_col_width}}"
    for method_name in all_methods.keys():
        header += f" | {method_name:<{method_col_widths[method_name]}}"
    print(header)
    print("-" * total_width)
    
    # 内容行
    for metric in ['loss', 'accuracy', 'f1']:
        row = f"{metric_names[metric]:<{metric_col_width}}"
        for method_name, metrics in all_methods.items():
            value = metrics.get(metric, float('nan'))
            if not np.isnan(value):
                row += f" | {value:>{method_col_widths[method_name]}.4f}" # Right align numbers
            else:
                row += f" | {'N/A':>{method_col_widths[method_name]}}" # Right align N/A
        print(row)
    
    print("-" * total_width)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bi-Level Active Data Repair")
    
    # 数据集参数
    parser.add_argument("--dataset", type=str, default="Marketing", help="数据集名称")
    parser.add_argument("--seed", type=int, default=43, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="使用的设备")
    parser.add_argument("--test_split", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--val_split", type=float, default=0.2, help="验证集比例（占总数据）")
    parser.add_argument("--detection_methods", type=str, default="['raha']", help="检测方法（JSON格式字符串）")
    parser.add_argument("--only_det_missing", type=bool, default=False, help="是否只检测缺失值")
    parser.add_argument("--loss_data", type=str, default="bilevel_diff_val", help="损失函数数据集")
    
    # 训练参数
    parser.add_argument("--use_gradnorm", type=bool, default=False, help="是否使用GradNorm")
    parser.add_argument("--classi_epochs", type=int, default=500, help="迭代中分类器每次训练的轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")
    parser.add_argument("--classi_lr", type=float, default=1e-3, help="分类器学习率")

    # 修复模型参数
    parser.add_argument("--time_emb_dim", type=int, default=128, help="时间嵌入维度")
    parser.add_argument("--hidden_dim", type=int, default=64, help="隐藏层维度")
    parser.add_argument("--latent_dim", type=int, default=16, help="潜在维度")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout率")
    parser.add_argument("--embed_dim", type=int, default=4, help="特征嵌入维度")
    parser.add_argument("--repair_model", type=str, default="unet", choices=["ae", "unet", "mlp"], help="修复模型架构，可选值: ae 或 unet 或 mlp")
    parser.add_argument("--num_layers", type=int, default=2, help="残差MLP的层数")

    # 扩散机制参数
    parser.add_argument("--enable_pretraining", type=str, default=False, help="是否使用diffusion机制进行pretraining")
    parser.add_argument("--use_diffusion", type=ast.literal_eval, default=False, help="是否使用diffusion机制进行填补（用于对照实验）")
    parser.add_argument("--diffusion_embed", type=ast.literal_eval, default=True, help="是否在嵌入空间中进行扩散（True:嵌入空间，False:原始数据空间）")
    parser.add_argument("--use_diffusion_loss", type=ast.literal_eval, default=True, help="在外层优化中使用Diffusion损失")
    parser.add_argument("--lambda_diff", type=float, default=1.0, help="外层优化中Diffusion损失的权重")
    parser.add_argument("--lambda_explicit_bilevel", type=float, default=1, help="外层优化中显式双层梯度损失的权重")
    parser.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "linear"], help="噪声调度")
    parser.add_argument("--num_timesteps", type=int, default=10, help="扩散时间步数")
    
    # 分类器参数
    parser.add_argument("--classi_hidden_dim", type=list, default=[256], help="分类器隐藏层维度")
    parser.add_argument("--l2_lambda_classifier", type=float, default=0.01, help="分类器L2正则化强度")
    parser.add_argument("--patience", type=int, default=300, help="早停耐心值")
    
    # bilevel参数
    parser.add_argument("--align_step_size", type=float, default=0.1, help="梯度对齐步长")
    parser.add_argument("--damping", type=float, default=0.1, help="阻尼参数")
    parser.add_argument("--hvp_batch_size", type=int, default=64, help="HVP批次大小")
    parser.add_argument("--num_iterations", type=int, default=1, help="内部迭代次数")
    parser.add_argument("--inner_steps", type=int, default=5, help="内层分类器训练步数（可自定义K值）")
    parser.add_argument("--outer_steps", type=int, default=1, help="外层分类器训练步数（可自定义K值）")
    parser.add_argument("--mu_align", type=float, default=0.1, help="梯度对齐损失中的μ参数，防止梯度塌缩")
    parser.add_argument("--outer_lr", type=float, default=1e-7, help="外层优化学习率")
    parser.add_argument("--outer_epochs", type=int, default=1, help="外层优化的epochs数量")
    parser.add_argument("--hvp_eps", type=float, default=1e-4, help="HVP计算中的扰动大小epsilon")
    parser.add_argument("--loss_strategy", type=str, default="explicit_bilevel", help="损失函数策略[alignment, explicit_bilevel, separability, reconstruction]组合")
    
    args = parser.parse_args()
    if args.loss_data == 'bilevel_only_val':
        args.use_diffusion = False
        args.use_diffusion_loss = False
    elif args.loss_data == 'diffusion_only':
        args.use_diffusion = True
        args.use_diffusion_loss = True
    else:
        args.use_diffusion = True
        args.use_diffusion_loss = True

    # if args.dataset in ['Hospital', 'Flights', 'Beers', 'Tax']:
    #     args.detection_methods = "['raha']"
    # else:
    #     args.detection_methods = "['semantic_isolation_forest', 'missing_values']"
    
    main(args) 