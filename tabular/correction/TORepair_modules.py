import ast
import contextlib
import importlib
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def resolve_project_root(denoising_project_root=None):
    if denoising_project_root:
        project_root = Path(denoising_project_root).resolve()
    else:
        project_root = Path(__file__).resolve().parent / "TORepair_tool"

    if not project_root.exists():
        raise FileNotFoundError(f"Denoising project root not found: {project_root}")
    return str(project_root)


def ensure_project_imports(project_root):
    project_root = str(Path(project_root).resolve())
    if project_root in sys.path:
        sys.path.remove(project_root)
    sys.path.insert(0, project_root)

    # Force TORepair imports to resolve from TORepair_tool instead of any
    # previously imported third-party/local packages with the same names.
    for package_name in ("models", "datasets", "utils"):
        module = sys.modules.get(package_name)
        module_file = getattr(module, "__file__", "") if module is not None else ""
        if module is not None and module_file and not str(Path(module_file).resolve()).startswith(project_root):
            for loaded_name in list(sys.modules):
                if loaded_name == package_name or loaded_name.startswith(package_name + "."):
                    del sys.modules[loaded_name]


def import_torepair_module(project_root, module_name):
    ensure_project_imports(project_root)
    return importlib.import_module(module_name)


@contextlib.contextmanager
def working_directory(path):
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def _parse_detection_methods(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return ["semantic_isolation_forest", "missing_values"]
    return value


def _ensure_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def _as_dataframe(value, name):
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, (str, os.PathLike)):
        return pd.read_csv(value)
    raise TypeError(f"{name} must be a pandas DataFrame or a CSV path.")


def _normalize_data_config(data_config):
    normalized = dict(data_config)
    normalized["numerical_features"] = _ensure_list(normalized.get("numerical_features")) or []
    normalized["categorical_features"] = _ensure_list(normalized.get("categorical_features")) or []
    normalized["target_feature"] = _ensure_list(normalized.get("target_feature")) or []
    return normalized


def build_args(params, project_root):
    ensure_project_imports(project_root)
    import torch

    device = params.get("device")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    args = SimpleNamespace(
        dataset=params.get("dataset", "Beers"),
        seed=params.get("seed", 43),
        device=device,
        test_split=params.get("test_split", 0.2),
        val_split=params.get("val_split", 0.2),
        detection_methods=_parse_detection_methods(params.get("detection_methods", "['raha']")),
        only_det_missing=params.get("only_det_missing", False),
        loss_data=params.get("loss_data", "bilevel_diff_val"),
        use_gradnorm=params.get("use_gradnorm", False),
        classi_epochs=params.get("classi_epochs", 500),
        batch_size=params.get("batch_size", 64),
        classi_lr=params.get("classi_lr", 1e-3),
        time_emb_dim=params.get("time_emb_dim", 128),
        hidden_dim=params.get("hidden_dim", 64),
        latent_dim=params.get("latent_dim", 16),
        dropout=params.get("dropout", 0),
        embed_dim=params.get("embed_dim", 4),
        repair_model=params.get("repair_model", "unet"),
        num_layers=params.get("num_layers", 2),
        enable_pretraining=params.get("enable_pretraining", False),
        use_diffusion=params.get("use_diffusion", False),
        diffusion_embed=params.get("diffusion_embed", True),
        use_diffusion_loss=params.get("use_diffusion_loss", True),
        lambda_diff=params.get("lambda_diff", 1.0),
        # training.py reads these names, while the original CLI exposes
        # lambda_diff and lambda_explicit_bilevel.
        lambda_diffusion=params.get("lambda_diffusion", params.get("lambda_diff", 1.0)),
        lambda_bilevel=params.get("lambda_bilevel", params.get("lambda_explicit_bilevel", 1)),
        lambda_explicit_bilevel=params.get("lambda_explicit_bilevel", 1),
        schedule=params.get("schedule", "cosine"),
        num_timesteps=params.get("num_timesteps", 10),
        classi_hidden_dim=params.get("classi_hidden_dim", [256]),
        l2_lambda_classifier=params.get("l2_lambda_classifier", 0.01),
        patience=params.get("patience", 300),
        align_step_size=params.get("align_step_size", 0.1),
        damping=params.get("damping", 0.1),
        hvp_batch_size=params.get("hvp_batch_size", 64),
        num_iterations=params.get("num_iterations", 1),
        inner_steps=params.get("inner_steps", 5),
        outer_steps=params.get("outer_steps", 1),
        mu_align=params.get("mu_align", 0.1),
        outer_lr=params.get("outer_lr", 1e-7),
        outer_epochs=params.get("outer_epochs", 1),
        hvp_eps=params.get("hvp_eps", 1e-4),
        loss_strategy=params.get("loss_strategy", "explicit_bilevel"),
        classifier_type=params.get("classifier_type", "mlp"),
        error_detector=None,
    )

    # Keep the original bilevel_beta/pipeline.py post-parse behavior.
    if args.loss_data == "bilevel_only_val":
        args.use_diffusion = False
        args.use_diffusion_loss = False
    elif args.loss_data == "diffusion_only":
        args.use_diffusion = True
        args.use_diffusion_loss = True
    else:
        args.use_diffusion = True
        args.use_diffusion_loss = True

    return args


def _build_data_config_from_frames(params, dirty_csv, detection_mask):
    numerical_features = _ensure_list(params.get("numerical_features"))
    categorical_features = _ensure_list(params.get("categorical_features"))
    target_feature = _ensure_list(params.get("target_feature"))

    if numerical_features is None or categorical_features is None or target_feature is None:
        raise ValueError(
            "When dirty_csv is provided, numerical_features, categorical_features, "
            "and target_feature must also be provided."
        )
    if detection_mask is None:
        raise ValueError("TORepair correction requires detection_mask when dirty_csv is provided.")

    dirty_df = _as_dataframe(dirty_csv, "dirty_csv")
    mask_df = _as_dataframe(detection_mask, "detection_mask").astype(bool)
    if len(dirty_df) != len(mask_df):
        raise ValueError("dirty_csv and detection_mask must have the same number of rows.")
    missing_mask_cols = [col for col in dirty_df.columns if col not in mask_df.columns]
    if missing_mask_cols:
        raise ValueError(f"detection_mask is missing columns: {missing_mask_cols}")

    temp_dir = tempfile.mkdtemp(prefix="torepair_dataframe_")
    data_path = os.path.join(temp_dir, "dirty.csv")
    mask_path = os.path.join(temp_dir, "error_mask_external.csv")
    dirty_df.to_csv(data_path, index=False)
    mask_df[dirty_df.columns].to_csv(mask_path, index=False)

    data_config = {
        "data_path": data_path,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "target_feature": target_feature,
        "_external_mask_path": mask_path,
        "_temp_dir": temp_dir,
    }
    return data_config


def _load_data_config(args, data_config, project_root):
    load_config_json = import_torepair_module(project_root, "utils.utility").load_config_json

    if data_config is not None:
        if isinstance(data_config, (str, os.PathLike)):
            data_config = load_config_json(str(data_config))
        return _normalize_data_config(data_config)

    data_config_path = f"data/{args.dataset}/data_config.json"
    if not os.path.exists(os.path.join(project_root, data_config_path)):
        raise FileNotFoundError(
            f"Missing original data config: {os.path.join(project_root, data_config_path)}. "
            "Pass data_config or pass dirty_csv with feature column lists."
        )
    return _normalize_data_config(load_config_json(data_config_path))


def _preprocess_dataframe_mode(args, data_config):
    project_root = os.getcwd()
    DataProcessor = import_torepair_module(project_root, "datasets.data_processor").DataProcessor
    utility_module = import_torepair_module(project_root, "utils.utility")
    split_dataframe_with_mask = utility_module.split_dataframe_with_mask
    identify_complete_and_error_samples_with_mask = utility_module.identify_complete_and_error_samples_with_mask

    df = pd.read_csv(data_config["data_path"])
    full_error_mask = pd.read_csv(data_config["_external_mask_path"]).astype(bool)

    target_feature = data_config["target_feature"]
    (train_val_df, train_val_mask), (test_df, test_mask) = split_dataframe_with_mask(
        df,
        full_error_mask,
        test_size=args.test_split,
        random_state=args.seed,
        stratify_column=target_feature[0] if len(target_feature) > 0 else None,
    )
    (train_df, train_mask), (val_df, val_mask) = split_dataframe_with_mask(
        train_val_df,
        train_val_mask,
        test_size=args.val_split / (1 - args.test_split),
        random_state=args.seed,
        stratify_column=target_feature[0] if len(target_feature) > 0 else None,
    )

    processor = DataProcessor(
        numerical_features=data_config["numerical_features"],
        categorical_features=data_config["categorical_features"],
        target_feature=target_feature,
        only_det_missing=args.only_det_missing,
        error_detector=None,
        args=args,
    )
    processor.fit(train_df)
    processor.set_error_mask_cache("train", train_mask, train_df)
    processor.set_error_mask_cache("val", val_mask, val_df)
    processor.set_error_mask_cache("test", test_mask, test_df)
    args.error_detector = None

    (train_df_complete, train_complete_mask), (train_df_error, train_error_mask) = (
        identify_complete_and_error_samples_with_mask(train_df, train_mask)
    )

    d_numerical = len(data_config["numerical_features"])
    d_categorical = len(data_config["categorical_features"])
    actual_cat_sizes = processor.categories
    num_classes = len(processor.label_encoder)

    return (
        processor,
        train_df,
        val_df,
        test_df,
        train_df_complete,
        train_df_error,
        d_numerical,
        d_categorical,
        actual_cat_sizes,
        num_classes,
        None,
        train_mask,
        val_mask,
        test_mask,
        train_complete_mask,
        train_error_mask,
    )


def _create_repair_model(project_root, args, d_numerical, d_categorical, actual_cat_sizes, total_dim, embedding_dim, predict_noise):
    diffae_core = import_torepair_module(project_root, "models.diffae.core")
    TabularAE = diffae_core.TabularAE
    TabularUNet = diffae_core.TabularUNet
    ResidualMLP = diffae_core.ResidualMLP

    if args.repair_model == "ae":
        return TabularAE(
            input_dim=total_dim,
            time_emb_dim=args.time_emb_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            d_numerical=d_numerical,
            d_categorical=d_categorical,
            actual_cat_sizes=actual_cat_sizes,
            dropout=args.dropout,
            num_classes=0,
            predict_noise=predict_noise,
            embedding_dim=embedding_dim,
        ).to(args.device)
    if args.repair_model == "mlp":
        return ResidualMLP(
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
    return TabularUNet(
        input_dim=total_dim,
        time_emb_dim=args.time_emb_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        d_numerical=d_numerical,
        d_categorical=d_categorical,
        actual_cat_sizes=actual_cat_sizes,
        dropout=args.dropout,
        predict_noise=predict_noise,
        embedding_dim=embedding_dim,
    ).to(args.device)


def train_torepair_pipeline(params, dirty_csv=None, detection_mask=None, data_config=None):
    project_root = resolve_project_root(params.get("denoising_project_root"))
    ensure_project_imports(project_root)

    if dirty_csv is not None:
        data_config = _build_data_config_from_frames(params, dirty_csv, detection_mask)

    args = build_args(params, project_root)

    with working_directory(project_root):
        from torch.utils.data import DataLoader

        original_pipeline = import_torepair_module(project_root, "models.bilevel_beta.pipeline")
        BiLevelTrainer = import_torepair_module(project_root, "models.bilevel_beta.training").BiLevelTrainer
        TabularDataset = import_torepair_module(project_root, "datasets.dataset").TabularDataset
        DiffusionUtils = import_torepair_module(project_root, "models.diffae.core").DiffusionUtils
        FeatureEmbedder = import_torepair_module(project_root, "utils.feature_embedder").FeatureEmbedder
        TRAIN_DATA_MASK_STR = import_torepair_module(project_root, "utils.constants").TRAIN_DATA_MASK_STR

        original_pipeline.set_seed(args.seed)
        loaded_config = _load_data_config(args, data_config, project_root)

        if dirty_csv is not None:
            preprocess_result = _preprocess_dataframe_mode(args, loaded_config)
        else:
            preprocess_result = original_pipeline.preprocess_data(args, loaded_config)

        (
            processor,
            train_df,
            val_df,
            test_df,
            train_df_complete,
            train_df_error,
            d_numerical,
            d_categorical,
            actual_cat_sizes,
            num_classes,
            error_detector,
            train_mask,
            val_mask,
            test_mask,
            train_complete_mask,
            train_error_mask,
        ) = preprocess_result

        # Match the original pipeline's construction order as closely as possible.
        feature_embedder = FeatureEmbedder(
            d_numerical=d_numerical,
            d_categorical=d_categorical,
            actual_cat_sizes=actual_cat_sizes,
            d_embed=args.embed_dim,
            dropout=args.dropout,
            device=args.device,
        ).to(args.device)

        diffusion_utils = DiffusionUtils(
            num_timesteps=args.num_timesteps,
            schedule=args.schedule,
            device=args.device,
        )

        train_data = TabularDataset(
            train_df,
            processor,
            dataset_name=TRAIN_DATA_MASK_STR,
            error_mask_df=train_mask,
        )
        DataLoader(train_data, batch_size=args.batch_size, shuffle=False)

        single_channel_dim = (d_numerical + d_categorical) * args.embed_dim
        total_dim = single_channel_dim * 2
        embedding_dim = single_channel_dim
        predict_noise = (
            True
            if (
                args.diffusion_embed
                and args.use_diffusion_loss
                and (args.loss_data == "bilevel_diff_val" or args.loss_data == "diffusion_only")
            )
            else False
        )

        _create_repair_model(
            project_root,
            args,
            d_numerical,
            d_categorical,
            actual_cat_sizes,
            total_dim,
            embedding_dim,
            predict_noise,
        )

        repair_model = _create_repair_model(
            project_root,
            args,
            d_numerical,
            d_categorical,
            actual_cat_sizes,
            total_dim,
            embedding_dim,
            predict_noise,
        )
        feature_embedder = FeatureEmbedder(
            d_numerical=d_numerical,
            d_categorical=d_categorical,
            actual_cat_sizes=actual_cat_sizes,
            d_embed=args.embed_dim,
            dropout=args.dropout,
            device=args.device,
        ).to(args.device)

        trainer = BiLevelTrainer(
            args=args,
            processor=processor,
            d_numerical=d_numerical,
            d_categorical=d_categorical,
            actual_cat_sizes=actual_cat_sizes,
            num_classes=num_classes,
            device=args.device,
            repair_model=repair_model,
            diffusion_utils=diffusion_utils,
            feature_embedder=feature_embedder,
        )

        train_results = trainer.run_bilevel_optimization(
            train_df_clean=train_df_complete,
            train_df_error=train_df_error,
            val_df=val_df,
            test_df=test_df,
            train_clean_mask=train_complete_mask,
            train_error_mask=train_error_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

    context = {
        "project_root": project_root,
        "trainer": trainer,
        "processor": processor,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "data_config": loaded_config,
        "args": args,
    }
    return args, context, train_results


def repair_dataframe(context, dirty_csv=None, detection_mask=None, dataset_name=None):
    project_root = context["project_root"]
    TEST_DATA_MASK_STR = import_torepair_module(project_root, "utils.constants").TEST_DATA_MASK_STR

    trainer = context["trainer"]
    if dirty_csv is None:
        dirty_csv = context["test_df"]
        detection_mask = context["test_mask"]
        dataset_name = dataset_name or TEST_DATA_MASK_STR
    else:
        dirty_csv = _as_dataframe(dirty_csv, "dirty_csv").reset_index(drop=True)
        if detection_mask is not None:
            detection_mask = _as_dataframe(detection_mask, "detection_mask").reset_index(drop=True).astype(bool)
            if len(dirty_csv) != len(detection_mask):
                raise ValueError("dirty_csv and detection_mask must have the same number of rows.")
            missing_mask_cols = [col for col in dirty_csv.columns if col not in detection_mask.columns]
            if missing_mask_cols:
                raise ValueError(f"detection_mask is missing columns: {missing_mask_cols}")
            detection_mask = detection_mask[dirty_csv.columns]

    with working_directory(project_root):
        return trainer.impute_with_model(
            dirty_csv,
            dataset_name=dataset_name,
            error_mask_df=detection_mask,
        )
