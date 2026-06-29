import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from utils.utility import load_config_json, split_dataframe, perform_simple_repair, perform_nan_only_repair
from utils.metrics import classi_metrics
from utils.main_modules import MLPClassifier
from utils.constants import TRAIN_DATA_MASK_STR, VAL_DATA_MASK_STR, TEST_DATA_MASK_STR
from torch.utils.data import DataLoader, Dataset
from models.imputation.standard_pipeline import train_classifier_on_repaired_data, combine_features, evaluate_classifier
from datasets.data_processor import DataProcessor


def _prepare_classifier_data(df: pd.DataFrame, processor: DataProcessor, device: str, dataset_name: str = None):
    """Helper function to prepare data for classifier training/evaluation."""
    x_num, x_cat_onehot = processor.transform_onehot(df, dataset_name=dataset_name)
    target_col = processor.target_feature[0]
    target_values = df[target_col].values
    if isinstance(target_values[0], (list, np.ndarray)):
        target_values = np.array([val[0] if isinstance(val, (list, np.ndarray)) else val for val in target_values])
    encoded_labels = np.array([processor.label_encoder.get(val, 0) for val in target_values])
    y = torch.LongTensor(encoded_labels).to(device)
    x_combined = combine_features(x_num, x_cat_onehot).to(device)
    return x_combined, y


def run_baseline_experiments(args, processor, train_df, val_df, test_df, 
                           numerical_features, categorical_features,
                           train_mask=None, val_mask=None, test_mask=None):
    """运行基准实验"""
    
    baseline_results = {}
    exps_str_list = [
        "基准实验1：使用Placeholder值填补",
        "基准实验2：使用均值/众数填补",
        "基准实验3：仅修复NaN值（Placeholder）",
        "基准实验4：仅修复NaN值（均值/众数）",
    ]
    
    for exp_id, exp_str in enumerate(exps_str_list):
        print("\n" + "="*50)
        print(exp_str)
        print("="*50)

        train_dataset_name = TRAIN_DATA_MASK_STR
        val_dataset_name = VAL_DATA_MASK_STR
        test_dataset_name = TEST_DATA_MASK_STR
        
        # 根据实验类型选择修复方法
        if exp_id in [0, 1]:  # 原有的实验：使用error mask
            place_holder = exp_id == 0
            train_df_simple_rep = perform_simple_repair(
                args=args,
                df=train_df,
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                dataset_name=train_dataset_name,
                placeholder=place_holder,
                external_error_mask=train_mask
            )
            
            val_df_simple_rep = perform_simple_repair(
                args=args,
                df=val_df,
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                dataset_name=val_dataset_name,
                placeholder=place_holder,
                external_error_mask=val_mask
            )
            
            test_df_simple_rep = perform_simple_repair(
                args=args,
                df=test_df,
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                dataset_name=test_dataset_name,
                placeholder=place_holder,
                external_error_mask=test_mask
            )
        else:  # 新实验：仅修复NaN值
            place_holder = exp_id == 2  # 实验3使用placeholder，实验4使用均值/众数
            train_df_simple_rep = perform_nan_only_repair(
                df=train_df,
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                placeholder=place_holder
            )
            
            val_df_simple_rep = perform_nan_only_repair(
                df=val_df,
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                placeholder=place_holder
            )
            
            test_df_simple_rep = perform_nan_only_repair(
                df=test_df,
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                placeholder=place_holder
            )
        
        # 准备分类器训练数据
        x_combined_train_si, y_train_si = _prepare_classifier_data(train_df_simple_rep, processor, args.device, dataset_name=train_dataset_name)
        
        # 准备验证集数据
        x_combined_val_si, y_val_si = _prepare_classifier_data(val_df_simple_rep, processor, args.device, dataset_name=val_dataset_name)
        
        # 训练分类器
        input_dim = x_combined_train_si.shape[1]
        output_dim = len(processor.label_encoder)
        
        simple_imp_classifier, _ = train_classifier_on_repaired_data(
            x_combined=x_combined_train_si,
            y=y_train_si,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=args.classi_hidden_dim,
            epochs=args.classi_epochs,
            batch_size=args.batch_size,
            lr=args.classi_lr,
            device=args.device,
            l2_lambda=args.l2_lambda_classifier,
            validation_split=0.0,
            patience=args.patience,
            x_val=x_combined_val_si,
            y_val=y_val_si
        )
        
        # 在测试集上评估
        x_combined_test_si, y_test_si = _prepare_classifier_data(test_df_simple_rep, processor, args.device, dataset_name=test_dataset_name)
        
        simple_imp_metrics = evaluate_classifier(
            classifier=simple_imp_classifier,
            x_combined=x_combined_test_si,
            y=y_test_si,
            classifier_criterion=torch.nn.CrossEntropyLoss()
        )
        exp_str = exp_str.split("：")[1]
        baseline_results[exp_str] = simple_imp_metrics
    
    return baseline_results

class BaselineTabularDataset(Dataset):
    """
    A simpler dataset class for the baseline, applying transformations.
    It expects pre-processed numerical and categorical data.
    """
    def __init__(self, X_num, X_cat, y, device):
        self.X_num = X_num.clone().detach().to(device) if X_num is not None and X_num.shape[1] > 0 else None
        # X_cat is expected to be one-hot encoded numpy array here
        self.X_cat = X_cat.clone().detach().to(device) if X_cat is not None and X_cat.shape[1] > 0 else None
        self.y = y.clone().detach().to(device)

        # Combine features for easier access in __getitem__
        features = []
        if self.X_num is not None:
            features.append(self.X_num)
        if self.X_cat is not None:
            features.append(self.X_cat)

        if not features:
             raise ValueError("Dataset must have at least numerical or categorical features.")
        elif len(features) == 1:
            self.X = features[0]
        else:
            self.X = torch.cat(features, dim=1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Main Execution ---
if __name__ == "__main__":
    # Load data and configuration
    df = pd.read_csv("data/Titanic/raw/dirty_train.csv")
    if 'PassengerId' in df.columns: df = df.drop('PassengerId', axis=1)
    if 'Name' in df.columns: df = df.drop('Name', axis=1)
    data_config = load_config_json("data/Titanic/data_type.json")

    # NaN filling
    for col in df.columns:
        if col in data_config.get("numerical_features", []):
            fill_value = df[col].median() if not df[col].isnull().all() else 0
            df[col] = df[col].fillna(fill_value)
        elif col in data_config.get("categorical_features", []):
            fill_value = 'Unknown'
            df[col] = df[col].fillna(fill_value)
        elif col in data_config.get("target_feature", []):
             df = df.dropna(subset=[col])

    # Split data into train and test sets
    train_df, test_df = split_dataframe(
        df, test_size=0.2,
        stratify_column=data_config["target_feature"][0] if data_config.get("target_feature") else None
    )

    # Initialize DataProcessor
    processor = DataProcessor(
        numerical_features=data_config.get("numerical_features"),
        categorical_features=data_config.get("categorical_features"),
        target_feature=data_config.get("target_feature")
    )
    processor.fit(df)

    # Process training data
    X_train_num, X_train_cat, y_train = processor.transform(train_df)
    cat_sizes = processor.categories
    actual_cat_sizes = [c - 1 for c in cat_sizes]

    # Convert categorical data to one-hot encoding
    X_train_cat_onehot_list = []
    if X_train_cat is not None and processor.d_categorical > 0:
        for i, n_cats_actual in enumerate(actual_cat_sizes):
            if n_cats_actual > 0:
                onehot = F.one_hot(X_train_cat[:, i].clone().detach().clamp(0, n_cats_actual - 1), num_classes=n_cats_actual).float()
                X_train_cat_onehot_list.append(onehot)
        X_train_cat_onehot = torch.cat(X_train_cat_onehot_list, dim=1)
    else:
        b_size = X_train_num.shape[0] if X_train_num is not None else len(y_train)
        X_train_cat_onehot = torch.empty(b_size, 0)

    # Handle numerical data
    if X_train_num is not None:
        X_train_num = X_train_num.clone().detach().float()
    else:
        b_size = X_train_cat_onehot.shape[0] if X_train_cat_onehot.shape[1] > 0 else len(y_train)
        X_train_num = torch.empty(b_size, 0)

    # Process test data
    X_test_num, X_test_cat, y_test = processor.transform(test_df)

    # Convert test categorical data to one-hot encoding
    X_test_cat_onehot_list = []
    if X_test_cat is not None and processor.d_categorical > 0:
        for i, n_cats_actual in enumerate(actual_cat_sizes):
            if n_cats_actual > 0:
                onehot = F.one_hot(X_test_cat[:, i].clone().detach().clamp(0, n_cats_actual - 1), num_classes=n_cats_actual).float()
                X_test_cat_onehot_list.append(onehot)
        X_test_cat_onehot = torch.cat(X_test_cat_onehot_list, dim=1)
    else:
        b_size = X_test_num.shape[0] if X_test_num is not None else len(y_test)
        X_test_cat_onehot = torch.empty(b_size, 0)

    # Handle numerical test data
    if X_test_num is not None:
        X_test_num = X_test_num.clone().detach().float()
    else:
        b_size = X_test_cat_onehot.shape[0] if X_test_cat_onehot.shape[1] > 0 else len(y_test)
        X_test_num = torch.empty(b_size, 0)

    # --- Setup for PyTorch ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create Datasets and DataLoaders
    train_dataset = BaselineTabularDataset(X_train_num, X_train_cat_onehot, y_train, device=device)
    test_dataset = BaselineTabularDataset(X_test_num, X_test_cat_onehot, y_test, device=device)

    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Model Definition ---
    # Calculate input dimension for the MLP based on processed features
    d_input = processor.d_numerical + sum(actual_cat_sizes)
    output_classes, _ = torch.unique(y_train, return_counts=True)
    num_output_classes = len(output_classes)
    # Use the same MLPClassifier structure
    classifier_model = MLPClassifier(
        input_size=d_input,
        hidden_sizes=[256],
        num_classes=num_output_classes
    ).to(device)

    # --- Training Loop ---
    criterion = nn.CrossEntropyLoss() # Standard loss for classification
    optimizer = optim.Adam(classifier_model.parameters(), lr=1e-4) # Common optimizer and LR
    epochs = 50

    print("Starting Baseline Training...")
    classifier_model.train() # Set model to training mode
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_dataloader:
            optimizer.zero_grad()
            outputs = classifier_model(batch_X)
            loss = criterion(outputs, batch_y.squeeze())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_dataloader):.4f}")

    # --- Testing ---
    print("\nStarting Testing...")
    classifier_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in test_dataloader:
            outputs = classifier_model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    metrics = classi_metrics(all_preds, all_labels)
    print("\n--- Test Results ---")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

    # Print the structure of MLPClassifier
    def print_mlp_classifier_structure(model):
        print("\nMLPClassifier Structure:")
        print("-------------------------")
        print(f"Input Size: {model.model[0].in_features}")
        
        hidden_layers = []
        current_layer = 0
        while current_layer < len(model.model):
            if isinstance(model.model[current_layer], nn.Linear):
                if current_layer < len(model.model) - 1:  # Not the output layer
                    hidden_layers.append(model.model[current_layer].out_features)
                else:  # Output layer
                    output_size = model.model[current_layer].out_features
                current_layer += 1
            else:
                current_layer += 1
        
        print(f"Hidden Layers: {hidden_layers}")
        print(f"Output Size: {output_size}")
        print(f"Activation: ReLU")
        print(f"Dropout Rate: 0.2")
        print("-------------------------")
    
    print_mlp_classifier_structure(classifier_model)