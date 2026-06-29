import torch
import pandas as pd
import numpy as np
import argparse
from utils.utility import load_config_json, split_dataframe, calculate_missing_rate, perform_simple_repair
from utils.main_modules import MLPClassifier
from utils.feature_embedder import FeatureEmbedder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from models.imputation.embedding import rep_embeds_diffusion
from models.embedding_diffusion import train_embedding_diffae
from models.diffae.core import TabularAE, TabularUNet, DiffusionUtils
from datasets.dataset import TabularDataset
from datasets.data_processor import DataProcessor
from utils.error_detection import ErrorDetector, calculate_error_rate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_onehot_encoding(x_cat, actual_cat_sizes, device):
    """
    Create one-hot encodings for categorical features.
    
    Args:
        x_cat: Categorical features tensor
        actual_cat_sizes: List of actual category sizes
        device: Device for tensor operations
        
    Returns:
        One-hot encoded categorical features
    """
    onehot_list = []
    if x_cat.shape[1] > 0:
        for i, n_cats in enumerate(actual_cat_sizes):
            if n_cats > 0:
                valid_indices = x_cat[:, i].clamp(0, n_cats - 1)
                onehot = torch.nn.functional.one_hot(valid_indices, num_classes=n_cats).float()
                onehot_list.append(onehot)
        if onehot_list:
            x_cat_onehot = torch.cat(onehot_list, dim=1)
        else:
            x_cat_onehot = torch.empty((x_cat.shape[0], 0), device=device)
    else:
        x_cat_onehot = torch.empty((x_cat.shape[0] if x_cat.shape[0] > 0 else 0, 0), device=device)
    
    return x_cat_onehot

def combine_features(x_num, x_cat_onehot):
    """
    Combine numerical and one-hot categorical features.
    
    Args:
        x_num: Numerical features tensor
        x_cat_onehot: One-hot encoded categorical features tensor
        
    Returns:
        Combined features tensor
    """
    if x_num is None or x_num.shape[1] == 0:
        return x_cat_onehot
    elif x_cat_onehot is None or x_cat_onehot.shape[1] == 0:
        return x_num
    else:
        return torch.cat((x_num, x_cat_onehot), dim=1)

def prepare_tensor_data(x_num, x_cat, y, actual_cat_sizes, device):
    """
    Prepare tensor data for model input by handling None values and transferring to device.
    
    Args:
        x_num: Numerical features
        x_cat: Categorical features
        y: Target labels
        actual_cat_sizes: Actual category sizes
        device: Device to transfer data to
        
    Returns:
        Tuple of (x_num, x_cat, y, x_combined) tensors ready for model input
    """
    if x_num is not None:
        x_num = x_num.clone().detach().float().to(device)
    else:
        x_num = torch.empty((len(y), 0), device=device)
    
    if x_cat is not None:
        x_cat = x_cat.clone().detach().long().to(device)
    else:
        x_cat = torch.empty((len(y), 0), dtype=torch.long, device=device)
    
    x_cat_onehot = create_onehot_encoding(x_cat, actual_cat_sizes, device)
    x_combined = combine_features(x_num, x_cat_onehot)

    y = y.clone().detach().long().to(device)
    
    return x_num, x_cat, y, x_combined

def evaluate_classifier(classifier, x_combined, y, classifier_criterion, prefix=""):
    """
    Evaluate classifier and return metrics.
    
    Args:
        classifier: Classifier model
        x_combined: Combined features tensor
        y: Target labels tensor
        classifier_criterion: Loss criterion
        prefix: Optional prefix for print statements
        
    Returns:
        Dictionary containing evaluation metrics
    """
    classifier.eval()
    with torch.no_grad():
        outputs = classifier(x_combined)
        loss = classifier_criterion(outputs, y.squeeze())
        
        _, predicted = torch.max(outputs.data, 1)
        
        y_true = y.cpu().numpy()
        y_pred = predicted.cpu().numpy()
        
        test_f1 = f1_score(y_true, y_pred, average='weighted')
        test_accuracy = accuracy_score(y_true, y_pred)
        
        print(f"{prefix} Loss: {loss.item():.4f}")
        print(f"{prefix} Accuracy: {test_accuracy:.4f}")
        print(f"{prefix} F1 Score: {test_f1:.4f}")
        
        return {
            'loss': loss.item(),
            'accuracy': test_accuracy,
            'f1': test_f1
        }


def train_classifier_on_repaired_data(x_combined, y, input_dim, output_dim, hidden_dim=[256], 
                                    epochs=100, batch_size=32, lr=0.001, device=DEVICE, l2_lambda=0.0,
                                    validation_split=0.1, patience=10, 
                                    x_val=None, y_val=None, pretrained_classifier=None, log_interval=100):
    """
    训练或微调分类器。
    
    Args:
        x_combined: 组合特征张量
        y: 目标标签张量
        input_dim: 分类器的输入维度
        output_dim: 分类器的输出维度
        hidden_dim: 分类器的隐藏维度
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        device: 训练设备
        l2_lambda: L2正则化强度（默认：0.0）
        validation_split: 用于验证的数据比例（默认：0.1）- 仅在未提供验证集时使用
        patience: 早停等待轮数（默认：10）
        x_val: 可选的验证特征张量
        y_val: 可选的验证标签张量
        pretrained_classifier: 可选的预训练分类器模型，如果提供则进行微调而非重新训练
        
    Returns:
        训练好的分类器和损失函数
    """
    # 确保批大小不超过数据集大小
    batch_size = min(batch_size, len(x_combined))
    
    # 检查是否提供了外部验证集
    use_validation = False
    train_x, train_y = x_combined, y
    
    if x_val is not None and y_val is not None and len(x_val) > 0 and len(y_val) > 0:
        use_validation = True
        val_x, val_y = x_val, y_val
        print(f"使用外部提供的验证集: 训练集={len(train_x)}，验证集={len(val_x)}")
    elif validation_split > 0 and len(x_combined) >= 10:
        # 如果未提供外部验证集，则从训练集中划分
        use_validation = True
        
        # Ensure validation split is reasonable
        if validation_split >= 0.5:
            validation_split = 0.2
            
        split_idx = int(len(x_combined) * (1 - validation_split))
        indices = torch.randperm(len(x_combined))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        if len(train_indices) == 0 or len(val_indices) == 0:
            print("警告: 划分的数据集为空，不使用验证集")
            use_validation = False
            train_x, train_y = x_combined, y
        else:
            train_x = x_combined[train_indices]
            train_y = y[train_indices]
            val_x = x_combined[val_indices]
            val_y = y[val_indices]
            print(f"从训练集中划分验证集: 训练集={len(train_indices)}，验证集={len(val_indices)}")
    else:
        print("数据集太小或验证划分比例为0，不使用验证集")
    
    train_dataset = TensorDataset(train_x, train_y)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    # 使用预训练分类器或创建新的分类器
    if pretrained_classifier is not None:
        print("使用预训练分类器进行微调")
        classifier = pretrained_classifier
    else:
        print("创建新的分类器")
        classifier = MLPClassifier(input_dim, hidden_dim, output_dim).to(device)
    
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0
    
    for epoch in range(epochs):
        classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for x_batch, y_batch in train_dataloader:
            if torch.cuda.is_available():
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

            if len(y_batch) == 0:
                continue
                
            optimizer.zero_grad()

            outputs = classifier(x_batch)

            y_batch_target = y_batch.view(-1) if y_batch.dim() > 1 else y_batch
            
            try:
                loss = criterion(outputs, y_batch_target)
            except Exception as e:
                print(f"计算损失时出错: {e}")
                print(f"输出形状: {outputs.shape}, 目标形状: {y_batch_target.shape}")
                continue
            
            # # Add L2 regularization
            # if l2_lambda > 0:
            #     l2_reg = torch.tensor(0.).to(device)
            #     for param in classifier.parameters():
            #         l2_reg += torch.norm(param, p=2)**2
            #     loss += l2_lambda * l2_reg
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch_target.size(0)
            correct += (predicted == y_batch_target).sum().item()
        
        train_loss = running_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        train_acc = 100 * correct / total if total > 0 else 0
        
        if use_validation:
            classifier.eval()
            with torch.no_grad():
                if torch.cuda.is_available():
                    val_x = val_x.to(device)
                    val_y = val_y.to(device)
                outputs = classifier(val_x)
                
                # Ensure val_y has correct dimensions
                val_y_target = val_y.view(-1) if val_y.dim() > 1 else val_y
                
                v_loss = criterion(outputs, val_y_target)
                
                # # Add L2 regularization
                # if l2_lambda > 0:
                #     l2_reg = torch.tensor(0.).to(device)
                #     for param in classifier.parameters():
                #         l2_reg += torch.norm(param, p=2)**2
                #     v_loss += l2_lambda * l2_reg
                
                val_loss = v_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct = (predicted == val_y_target).sum().item()
                val_acc = 100 * val_correct / val_y_target.size(0)
                
                # Count at each epoch
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = classifier.state_dict().copy()
                    counter = 0
                else:
                    counter += 1
                
                if counter > patience:
                    print(f'早停在第{epoch+1}轮。最佳验证损失: {best_val_loss:.4f}')
                    break
                    
        else:
            best_model_state = classifier.state_dict().copy()
            val_loss = 0
            val_acc = 0
        
        if (epoch + 1) % log_interval == 0:
            if use_validation:
                print(f'轮次 {epoch+1}/{epochs}, 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%, 验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%')
            else:
                print(f'轮次 {epoch+1}/{epochs}, 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
    
    if best_model_state is not None:
        classifier.load_state_dict(best_model_state)
    
    print('分类器训练完成')
    return classifier, criterion


def main(args):
    """
    Main function to run the standard imputation pipeline.
    
    Args:
        args: Command line arguments
    """
    print(f"Using device: {args.device}")
    
    data_config_path = f"data/{args.dataset}/data_config.json"
        
    data_config = load_config_json(data_config_path)
    
    numerical_features = data_config["numerical_features"]
    categorical_features = data_config["categorical_features"]
    target_feature = data_config["target_feature"]
    
    df_path = data_config["data_path"]
    df = pd.read_csv(df_path)
    
    if args.only_det_missing:
        ori_error_rates = calculate_missing_rate(df)
    else:
        error_detector = ErrorDetector(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            target_feature=target_feature,
            detection_methods=['semantic_isolation_forest', 'missing_values'],
        )
        error_detector.fit(df)
        ori_error_rates = calculate_error_rate(df, error_detector, dataset_name="standard_pipeline")
    print(f"Original Missing Rates:")
    for col, rate in ori_error_rates["column_rates"].items():
        print(f"  {col}: {rate:.4f}")
    print(f"Overall: {ori_error_rates['overall_rate']:.4f}")
    
    train_df, test_df = split_dataframe(df, test_size=0.2, random_state=args.seed, stratify_column=target_feature[0] if len(target_feature) > 0 else None)
    
    train_df_missing = train_df.copy()
    test_df_missing = test_df.copy()
    
    processor = DataProcessor(
        numerical_features=numerical_features, 
        categorical_features=categorical_features,
        target_feature=target_feature
    )
    processor.fit(train_df_missing)
    
    d_numerical = len(numerical_features)
    d_categorical = len(categorical_features)
    actual_cat_sizes = processor.categories
    input_dim = d_numerical + d_categorical
        
    # 获取标签数量
    num_classes = len(processor.label_encoder)
    print(f"Number of target classes: {num_classes}")
    
    # 使用统一的掩码标签值
    masked_label_value = num_classes  # 使用num_classes作为掩码值（超出正常类别范围）
    
    # 模型列表：将训练的不同模型及其配置
    models_to_train = []
    
    # 添加标签条件化模型
    if args.model_type == "diffae":
        label_cond_model = TabularAE(
            input_dim=input_dim*args.embed_dim,
            time_emb_dim=args.time_emb_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            d_numerical=d_numerical,
            d_categorical=d_categorical,
            actual_cat_sizes=actual_cat_sizes,
            dropout=args.dropout,
            num_classes=num_classes+1,  # +1 是为了包含掩码标签值
            d_label_embed=args.d_label_embed,
            masked_label_value=masked_label_value
        ).to(args.device)
        models_to_train.append(("label_cond", label_cond_model))
    else:  # "unet"
        print("注意：UNet模型尚未实现标签条件化")
    
    diffusion_utils = DiffusionUtils(
        num_timesteps=args.num_timesteps,
        schedule=args.schedule,
        device=args.device
    )
    
    train_data = TabularDataset(train_df_missing, processor)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    
    # 创建特征嵌入器（用于所有模型）
    feature_embedder = FeatureEmbedder(
        d_numerical=d_numerical,
        d_categorical=d_categorical,
        actual_cat_sizes=actual_cat_sizes,
        d_embed=args.embed_dim,
        dropout=args.dropout,
        device=args.device
    ).to(args.device)
    
    # 训练并评估每个模型
    model_results = {}
    
    # 存储原始数据和简单填补的结果，稍后比较用
    original_metrics = None
    simple_imp_metrics = None
    
    for model_name, model in models_to_train:
        print("\n" + "="*50)
        print(f"开始训练和评估模型：{model_name}")
        print("="*50)
        
        # 训练模型
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        if model_name == "label_cond":
            print("使用标签条件化训练")
            train_embedding_diffae(
                model=model,
                feature_embedder=feature_embedder,
                diffusion=diffusion_utils,
                dataloader=train_dataloader,
                optimizer=optimizer,
                device=args.device,
                epochs=args.diff_epochs,
                actual_cat_sizes=actual_cat_sizes,
                log_interval=10,
                label_mask_rate=args.label_mask_rate,
                num_classes=num_classes+1,
                masked_label_value=masked_label_value
            )
        else:
            print("不使用标签条件化训练")
            train_embedding_diffae(
                model=model,
                feature_embedder=feature_embedder,
                diffusion=diffusion_utils,
                dataloader=train_dataloader,
                optimizer=optimizer,
                device=args.device,
                epochs=args.diff_epochs,
                actual_cat_sizes=actual_cat_sizes,
                log_interval=10
            )
        
        # 填补训练数据
        print("\n" + "-"*50)
        print(f"使用{model_name}模型填补训练数据")
        print("-"*50)
        
        if model_name == "label_cond":
            train_df_rep = rep_embeds_diffusion(
                df=train_df_missing,
                model=model,
                feature_embedder=feature_embedder,
                diffusion_utils=diffusion_utils,
                processor=processor,
                num_steps=args.num_timesteps,
                batch_size=len(train_df_missing),
                device=args.device,
                verbose=True,
                use_masked_labels=False,  # 对训练数据使用真实标签
                masked_label_value=masked_label_value
            )
        else:
            # 无标签条件化模型无需传递标签相关参数
            train_df_rep = rep_embeds_diffusion(
                df=train_df_missing,
                model=model,
                feature_embedder=feature_embedder,
                diffusion_utils=diffusion_utils,
                processor=processor,
                num_steps=args.num_timesteps,
                batch_size=len(train_df_missing),
                device=args.device,
                verbose=True,
                use_masked_labels=False  # 不使用掩码标签
            )
        
        # 训练分类器
        print("\n" + "-"*50)
        print(f"在{model_name}模型填补的数据上训练分类器")
        print("-"*50)
        
        # 处理用于分类器的数据
        if args.use_embedding:
            if args.use_embeddings_for_classifier:
                x_num_imp, x_cat_imp, y_imp, _, _ = processor.transform(train_df_rep)
                x_num_imp = x_num_imp.to(args.device)
                x_cat_imp = x_cat_imp.to(args.device)
                y_imp = y_imp.to(args.device)
                
                with torch.no_grad():
                    x_emb, _ = feature_embedder(x_num_imp, x_cat_imp)
                
                classifier, _ = train_classifier_on_repaired_data(
                    x_combined=x_emb,
                    y=y_imp,
                    input_dim=x_emb.shape[1],
                    output_dim=len(np.unique(train_df[target_feature].values)),
                    hidden_dim=args.classi_hidden_dim,
                    epochs=args.classi_epochs,
                    batch_size=args.batch_size,
                    lr=args.classi_lr,
                    device=args.device,
                    l2_lambda=args.l2_lambda_classifier,
                    validation_split=args.val_split,
                    patience=args.patience,
                    x_val=None,
                    y_val=None
                )
            else:
                x_num_imp, x_cat_onehot_imp = processor.transform_onehot(train_df_rep, dataset_name="train")
                
                target_col = target_feature[0]
                target_values = train_df_rep[target_col].values
                if isinstance(target_values[0], (list, np.ndarray)):
                    target_values = np.array([val[0] if isinstance(val, (list, np.ndarray)) else val for val in target_values])
                
                encoded_labels = np.array([processor.label_encoder.get(val, 0) for val in target_values])
                y_imp = torch.LongTensor(encoded_labels)
                
                x_combined_imp = combine_features(x_num_imp, x_cat_onehot_imp)
                
                x_combined_imp = x_combined_imp.to(args.device)
                y_imp = y_imp.to(args.device)
                
                classifier, _ = train_classifier_on_repaired_data(
                    x_combined=x_combined_imp,
                    y=y_imp,
                    input_dim=x_combined_imp.shape[1],
                    output_dim=len(np.unique(train_df[target_feature].values)),
                    hidden_dim=args.classi_hidden_dim,
                    epochs=args.classi_epochs,
                    batch_size=args.batch_size,
                    lr=args.classi_lr,
                    device=args.device,
                    l2_lambda=args.l2_lambda_classifier,
                    validation_split=args.val_split,
                    patience=args.patience,
                    x_val=None,
                    y_val=None
                )
        
        # 填补测试数据
        print("\n" + "-"*50)
        print(f"使用{model_name}模型填补测试数据")
        print("-"*50)
        
        if model_name == "label_cond":
            test_df_imputed = rep_embeds_diffusion(
                df=test_df_missing,
                model=model,
                feature_embedder=feature_embedder,
                diffusion_utils=diffusion_utils,
                processor=processor,
                num_steps=args.num_timesteps,
                batch_size=len(test_df_missing),
                device=args.device,
                verbose=True,
                use_masked_labels=True,  # 对测试数据使用掩码标签值
                masked_label_value=masked_label_value
            )
        else:
            # 无标签条件化模型无需传递标签相关参数
            test_df_imputed = rep_embeds_diffusion(
                df=test_df_missing,
                model=model,
                feature_embedder=feature_embedder,
                diffusion_utils=diffusion_utils,
                processor=processor,
                num_steps=args.num_timesteps,
                batch_size=len(test_df_missing),
                device=args.device,
                verbose=True,
                use_masked_labels=False  # 不使用掩码标签
            )
        
        # 评估分类器
        print("\n" + "-"*50)
        print(f"在{model_name}模型填补的测试数据上评估分类器")
        print("-"*50)
        
        x_num_test, x_cat_test, y_test, _, _ = processor.transform(test_df_imputed)
        
        x_num_test = x_num_test.to(args.device)
        x_cat_test = x_cat_test.to(args.device)
        y_test = y_test.to(args.device)
        
        with torch.no_grad():
            x_emb_test, _ = feature_embedder(x_num_test, x_cat_test)
            
        diffusion_metrics = evaluate_classifier(
            classifier=classifier,
            x_combined=x_emb_test,
            y=y_test,
            classifier_criterion=torch.nn.CrossEntropyLoss(),
            prefix=f"{model_name} "
        )
        
        # 保存该模型的结果
        model_results[model_name] = diffusion_metrics
    
    # 如果需要比较，评估原始数据（不填补）
    print("\n" + "-"*50)
    print("评估原始数据（不填补）")
    print("-"*50)
    
    # 使用原始训练和测试数据
    train_df_ori = train_df_missing
    test_df_ori = test_df_missing

    if not train_df_ori.empty and not test_df_ori.empty and len(train_df_ori) > 1 and len(test_df_ori) > 1:
        x_num_ori_train, x_cat_onehot_ori_train = processor.transform_onehot(train_df_ori, dataset_name="train")
        target_col = target_feature[0]
        target_values = train_df_ori[target_col].values
        if isinstance(target_values[0], (list, np.ndarray)):
            target_values = np.array([val[0] if isinstance(val, (list, np.ndarray)) else val for val in target_values])
        encoded_labels = np.array([processor.label_encoder.get(val, 0) for val in target_values])
        y_ori_train = torch.LongTensor(encoded_labels)
        
        x_combined_ori_train = combine_features(x_num_ori_train, x_cat_onehot_ori_train)
        
        x_combined_ori_train = x_combined_ori_train.to(args.device)
        y_ori_train = y_ori_train.to(args.device)

        print("训练原始数据分类器（无填补）...")
        output_dim_classifier = len(np.unique(train_df[target_feature].values))

        complete_case_classifier, complete_case_criterion = train_classifier_on_repaired_data(
            x_combined=x_combined_ori_train,
            y=y_ori_train,
            input_dim=x_combined_ori_train.shape[1],
            output_dim=output_dim_classifier,
            hidden_dim=args.classi_hidden_dim,
            epochs=args.classi_epochs,
            batch_size=args.batch_size,
            lr=args.classi_lr,
            device=args.device,
            l2_lambda=args.l2_lambda_classifier,
            validation_split=args.val_split,
            patience=args.patience,
            x_val=None,
            y_val=None
        )

        x_num_ori_test, x_cat_onehot_ori_test = processor.transform_onehot(test_df_ori, dataset_name="test")
        target_col = target_feature[0]
        target_values = test_df_ori[target_col].values
        if isinstance(target_values[0], (list, np.ndarray)):
            target_values = np.array([val[0] if isinstance(val, (list, np.ndarray)) else val for val in target_values])
        encoded_labels = np.array([processor.label_encoder.get(val, 0) for val in target_values])
        y_ori_test = torch.LongTensor(encoded_labels)

        x_combined_ori_test = combine_features(x_num_ori_test, x_cat_onehot_ori_test)
        
        x_combined_ori_test = x_combined_ori_test.to(args.device)
        y_ori_test = y_ori_test.to(args.device)

        print("评估原始数据分类器（无填补）...")
        original_metrics = evaluate_classifier(
            classifier=complete_case_classifier,
            x_combined=x_combined_ori_test,
            y=y_ori_test,
            classifier_criterion=complete_case_criterion,
            prefix="原始数据 "
        )
    else:
        print("跳过原始数据评估，因为训练或测试数据为空或太小。")
        original_metrics = {
            'loss': float('nan'),
            'accuracy': float('nan'),
            'f1': float('nan')
        }

    # 如果需要比较，评估简单填补
    print("\n" + "-"*50)
    print("评估简单填补（均值/众数填补）")
    print("-"*50)
    
    train_df_simple_imp = perform_simple_repair(
        df=train_df_missing,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        placeholder=True
    )
    
    test_df_simple_imp = perform_simple_repair(
        df=test_df_missing,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        placeholder=True
    )
    
    num_mask = ~test_df_missing[numerical_features].isna().values
    num_mask = torch.FloatTensor(num_mask.astype(np.float32))
    
    cat_mask = ~test_df_missing[categorical_features].isna().values
    cat_mask = torch.FloatTensor(cat_mask.astype(np.float32))
    
    x_num_simple_train, x_cat_onehot_simple_train = processor.transform_onehot(train_df_simple_imp, dataset_name="train")
    target_col = target_feature[0]
    target_values = train_df_simple_imp[target_col].values
    if isinstance(target_values[0], (list, np.ndarray)):
        target_values = np.array([val[0] if isinstance(val, (list, np.ndarray)) else val for val in target_values])
    encoded_labels = np.array([processor.label_encoder.get(val, 0) for val in target_values])
    y_simple_train = torch.LongTensor(encoded_labels)
    
    x_combined_simple_train = combine_features(x_num_simple_train, x_cat_onehot_simple_train)
    
    x_combined_simple_train = x_combined_simple_train.to(args.device)
    y_simple_train = y_simple_train.to(args.device)
    
    simple_imp_classifier, simple_imp_criterion = train_classifier_on_repaired_data(
        x_combined=x_combined_simple_train,
        y=y_simple_train,
        input_dim=x_combined_simple_train.shape[1],
        output_dim=output_dim_classifier,
        hidden_dim=args.classi_hidden_dim,
        epochs=args.classi_epochs,
        batch_size=args.batch_size,
        lr=args.classi_lr,
        device=args.device,
        l2_lambda=args.l2_lambda_classifier,
        validation_split=args.val_split,
        patience=args.patience,
        x_val=None,
        y_val=None
    )
    
    x_num_simple_test, x_cat_onehot_simple_test = processor.transform_onehot(test_df_simple_imp, dataset_name="test")
    target_col = target_feature[0]
    target_values = test_df_simple_imp[target_col].values
    if isinstance(target_values[0], (list, np.ndarray)):
        target_values = np.array([val[0] if isinstance(val, (list, np.ndarray)) else val for val in target_values])
    
    encoded_labels = np.array([processor.label_encoder.get(val, 0) for val in target_values])
    y_simple_test = torch.LongTensor(encoded_labels)
    
    x_combined_simple_test = combine_features(x_num_simple_test, x_cat_onehot_simple_test)
    
    x_combined_simple_test = x_combined_simple_test.to(args.device)
    y_simple_test = y_simple_test.to(args.device)
    
    simple_imp_metrics = evaluate_classifier(
        classifier=simple_imp_classifier,
        x_combined=x_combined_simple_test,
        y=y_simple_test,
        classifier_criterion=simple_imp_criterion,
        prefix="简单填补 "
    )
    
    # 比较所有方法的结果
    print("\n" + "="*50)
    print("各种方法的性能比较")
    print("="*50)
    
    # 创建一个包含所有方法的结果字典
    all_methods = {}
    if original_metrics:
        all_methods["原始数据"] = original_metrics
    if simple_imp_metrics:
        all_methods["简单填补"] = simple_imp_metrics
    for model_name, metrics in model_results.items():
        display_name = "标签条件化" if model_name == "label_cond" else "无标签条件化"
        all_methods[display_name] = metrics
    
    # 打印比较表格
    print("-" * 100)
    metric_names = {
        'loss': '损失',
        'accuracy': '准确率',
        'f1': 'F1分数'
    }
    
    # 表头
    header = f"{'指标':<10}"
    for method_name in all_methods.keys():
        header += f" | {method_name:<15}"
    print(header)
    print("-" * 100)
    
    # 内容行
    for metric in ['loss', 'accuracy', 'f1']:
        row = f"{metric_names[metric]:<10}"
        baseline_value = None
        for method_name, metrics in all_methods.items():
            value = metrics.get(metric, float('nan'))
            row += f" | {value:<15.4f}"
            if method_name == "简单填补":
                baseline_value = value
        print(row)
    
    print("-" * 100)
    
    # 如果有标签条件化和无标签条件化的结果，计算改进百分比
    if "标签条件化" in all_methods and "无标签条件化" in all_methods and "简单填补" in all_methods:
        print("\n改进百分比：")
        print("-" * 60)
        print(f"{'指标':<10} | {'标签 vs 无标签':<15} | {'标签 vs 简单填补':<15}")
        print("-" * 60)
        
        for metric in ['loss', 'accuracy', 'f1']:
            label_cond_value = all_methods["标签条件化"].get(metric, float('nan'))
            no_label_cond_value = all_methods["无标签条件化"].get(metric, float('nan'))
            simple_imp_value = all_methods["简单填补"].get(metric, float('nan'))
            
            improvement_vs_no_label = float('nan')
            improvement_vs_simple = float('nan')
            
            if not np.isnan(label_cond_value) and not np.isnan(no_label_cond_value):
                if metric == 'loss':
                    improvement_vs_no_label = ((no_label_cond_value - label_cond_value) / no_label_cond_value) * 100
                else:
                    improvement_vs_no_label = ((label_cond_value - no_label_cond_value) / no_label_cond_value) * 100
            
            if not np.isnan(label_cond_value) and not np.isnan(simple_imp_value):
                if metric == 'loss':
                    improvement_vs_simple = ((simple_imp_value - label_cond_value) / simple_imp_value) * 100
                else:
                    improvement_vs_simple = ((label_cond_value - simple_imp_value) / simple_imp_value) * 100
            
            print(f"{metric_names[metric]:<10} | {improvement_vs_no_label:<15.2f}% | {improvement_vs_simple:<15.2f}%")
        
        print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standard Imputation Pipeline")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="Titanic", help="Dataset name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    
    # Diffusion model parameters
    parser.add_argument("--model_type", type=str, default="diffae", choices=["diffae", "unet"], help="Diffusion model type")
    parser.add_argument("--time_emb_dim", type=int, default=128, help="Time embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--num_timesteps", type=int, default=100, help="Number of diffusion timesteps")
    parser.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "linear"], help="Noise schedule")
    
    # Diffusion model training parameters
    parser.add_argument("--diff_epochs", type=int, default=150, help="Number of diffusion model training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    
    # Embedding approach parameters
    parser.add_argument("--embed_dim", type=int, default=8, help="Dimension of feature embeddings")
    
    # 标签条件化参数
    parser.add_argument("--d_label_embed", type=int, default=16, help="Dimension of label embeddings")
    parser.add_argument("--label_mask_rate", type=float, default=0.3, help="Rate of labels to mask during training")
    parser.add_argument("--use_label_conditioning", action="store_true", help="Whether to use label conditioning")
    parser.add_argument("--compare_conditioning", action="store_true", help="Whether to compare models with and without label conditioning")
    
    # Classifier parameters
    parser.add_argument("--classi_epochs", type=int, default=150, help="Number of classifier training epochs")
    parser.add_argument("--classi_lr", type=float, default=0.0001, help="Learning rate for classifier")
    parser.add_argument("--classi_hidden_dim", type=list, default=[256], help="Hidden dimension for classifier")
    parser.add_argument("--l2_lambda_classifier", type=float, default=0.01, help="L2 regularization strength for classifier")
    parser.add_argument("--patience", type=int, default=10, help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of data to use for validation")
    
    args = parser.parse_args()
    
    main(args)
