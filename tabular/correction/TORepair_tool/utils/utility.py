import torch
import pandas as pd
import numpy as np
import json
from typing import Tuple
from sklearn.model_selection import train_test_split
from utils.constants import ADDITIONAL_MISSING_VALUES
from utils.error_detection import calculate_error_rate


def load_config_json(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config


def perform_simple_repair(args, df, numerical_features, categorical_features, dataset_name, placeholder=False, external_error_mask=None):
    """
    Perform simple imputation on the dataframe:
    - For numerical features: fill with mean
    - For categorical features: fill with 'MASK' or mode
    
    Args:
        df: Input dataframe
        numerical_features: List of numerical column names
        categorical_features: List of categorical column names
        dataset_name: Dataset name for error detection (unused if external_error_mask is provided)
        placeholder: Whether to use placeholder values
        external_error_mask: Pre-computed error mask DataFrame (True=error/missing, False=correct)
        
    Returns:
        Dataframe with imputed values
    """
    df_imputed = df.copy()
    
    # 使用外部传入的错误掩码，或者计算新的掩码
    if external_error_mask is not None:
        combined_err_mask = external_error_mask
    else:
        if args.only_det_missing or args.error_detector is None:
            combined_err_mask = calculate_missing_rate(df_imputed)['error_mask']
        else:
            combined_err_mask = calculate_error_rate(df_imputed, args.error_detector, dataset_name=dataset_name)['error_mask']
    
    # Fill numerical features with mean
    for col in numerical_features:
        # 检查此列是否有error
        if combined_err_mask[col].any():
            # 将Series转换为布尔数组以避免索引错误
            error_mask_col = combined_err_mask[col].values
            if not placeholder:
                # 计算非error的均值
                valid_values = df_imputed.loc[~error_mask_col, col]
                if len(valid_values) > 0:
                    mean_value = valid_values.mean()
                    
                    # 将所有error值（包括NaN和其他error表示）替换为均值，并确保数据类型匹配
                    if np.issubdtype(df_imputed[col].dtype, np.integer):
                        # 对于整数类型列，将均值转换为整数
                        df_imputed.loc[error_mask_col, col] = int(mean_value)
                    else:
                        # 对于浮点类型列，保持浮点数
                        df_imputed.loc[error_mask_col, col] = mean_value
                    
                    # 将列转换为数值类型，以防有些值是字符串类型的缺失表示
                    df_imputed[col] = pd.to_numeric(df_imputed[col], errors='coerce')
            else:
                if np.issubdtype(df_imputed[col].dtype, np.integer):
                    df_imputed.loc[error_mask_col, col] = -1
                else:
                    df_imputed.loc[error_mask_col, col] = float(-1)
    
    # Fill categorical features with 'MASK' or mode
    for col in categorical_features:
        # 检查此列是否有error
        if combined_err_mask[col].any():
            # 将Series转换为布尔数组以避免索引错误
            error_mask_col = combined_err_mask[col].values
            if placeholder:
                # 对于初始填补，使用'MASK'
                df_imputed.loc[error_mask_col, col] = "MASK"
            else:
                # 计算非error的众数
                valid_values = df_imputed.loc[~error_mask_col, col]
                if len(valid_values) > 0:
                    mode_value = valid_values.mode()[0]
                    df_imputed.loc[error_mask_col, col] = mode_value
    
    return df_imputed


def perform_nan_only_repair(df, numerical_features, categorical_features, placeholder=False):
    """
    仅对NaN值进行修复，忽略error mask
    - 对于数值特征：如果placeholder=True用-1填充，否则用均值填充
    - 对于分类特征：如果placeholder=True用'MASK'填充，否则用众数填充
    
    Args:
        df: 输入数据框
        numerical_features: 数值特征列表
        categorical_features: 分类特征列表  
        placeholder: 是否使用placeholder值
        
    Returns:
        修复NaN值后的数据框
    """
    df_imputed = df.copy()
    
    # 处理数值特征
    for col in numerical_features:
        if df_imputed[col].isna().any():
            nan_mask = df_imputed[col].isna()
            if placeholder:
                # 使用placeholder值
                if np.issubdtype(df_imputed[col].dtype, np.integer):
                    df_imputed.loc[nan_mask, col] = -1
                else:
                    df_imputed.loc[nan_mask, col] = float(-1)
            else:
                # 使用均值填充
                valid_values = df_imputed[col].dropna()
                if len(valid_values) > 0:
                    mean_value = valid_values.mean()
                    if np.issubdtype(df_imputed[col].dtype, np.integer):
                        df_imputed.loc[nan_mask, col] = int(mean_value)
                    else:
                        df_imputed.loc[nan_mask, col] = mean_value
    
    # 处理分类特征
    for col in categorical_features:
        if df_imputed[col].isna().any():
            nan_mask = df_imputed[col].isna()
            if placeholder:
                # 使用'MASK'填充
                df_imputed.loc[nan_mask, col] = "MASK"
            else:
                # 使用众数填充
                valid_values = df_imputed[col].dropna()
                if len(valid_values) > 0:
                    mode_value = valid_values.mode()[0]
                    df_imputed.loc[nan_mask, col] = mode_value
    
    return df_imputed


def calculate_missing_rate(df: pd.DataFrame) -> dict:
    """
    Calculate the missing rate in a dataframe, considering NaN, None, 
    and optionally specified placeholder values (including empty/whitespace strings).

    Args:
        df (pd.DataFrame): Input dataframe.
        additional_missing_values (list, optional): A list of additional values 
            to be treated as missing. Defaults to a common set:
            ['', ' ', 'NA', 'N/A', 'na', 'n/a', 'null', 'Null', 'NULL', 
             'None', 'none', '?', '-']. Note that Python's None is handled
             by isna() regardless of this list. Include '' and/or ' ' 
             if empty or whitespace strings should be counted.

    Returns:
        dict: A dictionary containing:
            - "column_rates": A dictionary of missing rates per column.
            - "overall_rate": The overall missing rate as a float between 0 and 1.

    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If total cells is zero (empty DataFrame shape).
    """
        
    # Use a default list if none provided
    additional_missing_values = ADDITIONAL_MISSING_VALUES
    
    missing_nan_mask = df.isna()
    missing_additional_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    
    if additional_missing_values:
        missing_additional_mask = df.isin(additional_missing_values)

    check_whitespace = '' in additional_missing_values or ' ' in additional_missing_values
    
    if check_whitespace:
        for col in df.select_dtypes(include=['object', 'string']).columns:
            is_whitespace_empty = df[col].astype(str).str.strip().isin(['']) 
            missing_additional_mask[col] = missing_additional_mask[col] | is_whitespace_empty

    combined_missing_mask = missing_nan_mask | missing_additional_mask
    missing_cells_per_column = combined_missing_mask.sum()
    total_cells = df.size  # df.size is equivalent to df.shape[0] * df.shape[1]
    
    if total_cells == 0:
        raise ValueError("Cannot calculate missing rate for an empty DataFrame (zero cells).")
    
    # Calculate column-wise missing rates
    column_rates = (missing_cells_per_column / df.shape[0]).to_dict()
    
    # Calculate overall missing rate
    overall_rate = missing_cells_per_column.sum() / total_cells
    
    return {
        "error_mask": combined_missing_mask,
        "column_rates": column_rates,
        "overall_rate": overall_rate
    }


def split_dataframe_with_mask(df, mask_df, test_size=0.2, random_state=42, stratify_column=None):
    """
    Split a dataframe and its corresponding mask into training and testing sets.
    
    Args:
        df: pandas DataFrame to split
        mask_df: pandas DataFrame mask corresponding to df
        test_size: proportion of data to use for testing (default: 0.2)
        random_state: random seed for reproducibility (default: 42)
        stratify_column: column name to use for stratified splitting (default: None)
            If provided, ensures that the distribution of values in this column
            is preserved in both train and test sets.
    
    Returns:
        (train_df, train_mask), (test_df, test_mask): tuples containing data and mask splits
    """
    # 处理stratify列，确保不包含NaN，否则train_test_split会报错
    if stratify_column and stratify_column in df.columns:
        stratify = df[stratify_column]
        if stratify.isna().any():
            # 若存在NaN，填充占位符以避免报错，保持类别信息
            if pd.api.types.is_numeric_dtype(stratify):
                stratify = stratify.fillna(-1)  # 使用-1作为占位符
            else:
                stratify = stratify.astype(object).fillna("MASK")
    else:
        stratify = None
        
    # 使用相同的索引来分割数据和掩码
    train_indices, test_indices = train_test_split(
        df.index, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify
    )
    
    train_df = df.loc[train_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)
    train_mask = mask_df.loc[train_indices].reset_index(drop=True)
    test_mask = mask_df.loc[test_indices].reset_index(drop=True)
    
    return (train_df, train_mask), (test_df, test_mask)


def identify_complete_and_error_samples_with_mask(df: pd.DataFrame, error_mask_df: pd.DataFrame) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    识别完全无error的样本和至少有一个error的样本，使用预计算的错误掩码
    
    Args:
        df: 输入数据框
        error_mask_df: 预计算的错误掩码DataFrame
        
    Returns:
        ((完全无error的样本子集, 对应的掩码), (至少有一个error的样本子集, 对应的掩码))
    """
    # 识别完全无error的行
    non_err_mask = ~error_mask_df.any(axis=1)
    df_complete = df[non_err_mask].copy().reset_index(drop=True)
    mask_complete = error_mask_df[non_err_mask].copy().reset_index(drop=True)
    
    # 识别有至少一个error的行
    err_mask = error_mask_df.any(axis=1)
    df_missing = df[err_mask].copy()
    mask_missing = error_mask_df[err_mask].copy()
    
    # 计算每行error的数量并按error数量升序排序
    missing_counts = error_mask_df.sum(axis=1)
    df_missing['missing_count'] = missing_counts
    mask_missing['missing_count'] = missing_counts
    
    # 排序
    sort_indices = df_missing['missing_count'].argsort()
    df_missing = df_missing.iloc[sort_indices].reset_index(drop=True)
    mask_missing = mask_missing.iloc[sort_indices].reset_index(drop=True)
    
    # 删除辅助列
    df_missing = df_missing.drop(columns=['missing_count'])
    mask_missing = mask_missing.drop(columns=['missing_count'])
    
    return (df_complete, mask_complete), (df_missing, mask_missing)


def identify_complete_and_error_samples(args, df: pd.DataFrame, dataset_name: str, error_mask_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    识别完全无error的样本和至少有一个error的样本
    
    Args:
        args: 参数对象
        df: 输入数据框
        dataset_name: 数据集名称
        error_mask_df: 可选的预计算错误掩码，如果提供则直接使用
        
    Returns:
        完全无error的样本子集, 至少有一个error的样本子集（按error数量排序）
    """
    if error_mask_df is not None:
        # 使用新的带掩码的函数
        (df_complete, _), (df_missing, _) = identify_complete_and_error_samples_with_mask(df, error_mask_df)
        return df_complete, df_missing
    else:
        # 保持原有逻辑作为兼容性
        # 合并所有error掩码
        if args.only_det_missing:
            combined_error_mask = calculate_missing_rate(df)['error_mask']
        else:
            combined_error_mask = calculate_error_rate(df, args.error_detector, dataset_name=dataset_name)['error_mask']
        
        # 识别完全无error的行
        complete_mask = ~combined_error_mask.any(axis=1)
        df_complete = df[complete_mask].copy().reset_index(drop=True)
        
        # 识别有至少一个error的行
        missing_mask = combined_error_mask.any(axis=1)
        df_missing = df[missing_mask].copy()
        
        # 计算每行error的数量并按error数量升序排序
        missing_counts = combined_error_mask.sum(axis=1)
        df_missing['missing_count'] = missing_counts
        df_missing = df_missing.sort_values('missing_count').reset_index()
        
        # 保留原始索引信息以便后续参考，同时删除辅助列
        df_missing = df_missing.rename(columns={'index': 'original_index'})
        df_missing = df_missing.drop(columns=['missing_count'])
        
        return df_complete, df_missing


def split_dataframe(df, test_size=0.2, random_state=42, stratify_column=None):
    """
    Split a dataframe into training and testing sets.
    
    Args:
        df: pandas DataFrame to split
        test_size: proportion of data to use for testing (default: 0.2)
        random_state: random seed for reproducibility (default: 42)
        stratify_column: column name to use for stratified splitting (default: None)
            If provided, ensures that the distribution of values in this column
            is preserved in both train and test sets.
    
    Returns:
        train_df: pandas DataFrame containing training data
        test_df: pandas DataFrame containing testing data
    """
    # 处理stratify列，确保不包含NaN
    if stratify_column and stratify_column in df.columns:
        stratify = df[stratify_column]
        if stratify.isna().any():
            if pd.api.types.is_numeric_dtype(stratify):
                stratify = stratify.fillna(-1)
            else:
                stratify = stratify.astype(object).fillna("MISSING")
    else:
        stratify = None
        
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify
    )
    return train_df, test_df
