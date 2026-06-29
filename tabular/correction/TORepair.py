import os

try:
    from dataprep.base import BaseEstimator
    import dataprep.tabular.correction.TORepair_modules as modules
except ModuleNotFoundError:
    from base import BaseEstimator
    import tabular.correction.TORepair_modules as modules


class TORepair(BaseEstimator):
    def __init__(self,
                 denoising_project_root=None,
                 dataset="Beers",
                 device=None,
                 detection_methods="['raha']",
                 only_det_missing=False,
                 loss_data="bilevel_diff_val",
                 seed=43,
                 test_split=0.2,
                 val_split=0.2,
                 batch_size=64,
                 repair_model="unet",
                 classifier_type="mlp",
                 use_gradnorm=False,
                 classi_epochs=500,
                 classi_lr=1e-3,
                 time_emb_dim=128,
                 hidden_dim=64,
                 latent_dim=16,
                 dropout=0,
                 embed_dim=4,
                 num_layers=2,
                 enable_pretraining=False,
                 use_diffusion=False,
                 diffusion_embed=True,
                 use_diffusion_loss=True,
                 lambda_diff=1.0,
                 lambda_explicit_bilevel=1,
                 schedule="cosine",
                 num_timesteps=10,
                 classi_hidden_dim=None,
                 l2_lambda_classifier=0.01,
                 patience=300,
                 align_step_size=0.1,
                 damping=0.1,
                 hvp_batch_size=64,
                 num_iterations=1,
                 inner_steps=5,
                 outer_steps=1,
                 mu_align=0.1,
                 outer_lr=1e-7,
                 outer_epochs=1,
                 hvp_eps=1e-4,
                 loss_strategy="explicit_bilevel",
                 output_dir="./result/torepair",
                 numerical_features=None,
                 categorical_features=None,
                 target_feature=None,
                 **kwargs):
        # 1. Attribute assignment
        self.denoising_project_root = denoising_project_root
        self.dataset = dataset
        self.device = device
        self.detection_methods = detection_methods
        self.only_det_missing = only_det_missing
        self.loss_data = loss_data
        self.seed = seed
        self.test_split = test_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.repair_model = repair_model
        self.classifier_type = classifier_type
        self.use_gradnorm = use_gradnorm
        self.classi_epochs = classi_epochs
        self.classi_lr = classi_lr
        self.time_emb_dim = time_emb_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.enable_pretraining = enable_pretraining
        self.use_diffusion = use_diffusion
        self.diffusion_embed = diffusion_embed
        self.use_diffusion_loss = use_diffusion_loss
        self.lambda_diff = lambda_diff
        self.lambda_explicit_bilevel = lambda_explicit_bilevel
        self.schedule = schedule
        self.num_timesteps = num_timesteps
        self.classi_hidden_dim = classi_hidden_dim or [256]
        self.l2_lambda_classifier = l2_lambda_classifier
        self.patience = patience
        self.align_step_size = align_step_size
        self.damping = damping
        self.hvp_batch_size = hvp_batch_size
        self.num_iterations = num_iterations
        self.inner_steps = inner_steps
        self.outer_steps = outer_steps
        self.mu_align = mu_align
        self.outer_lr = outer_lr
        self.outer_epochs = outer_epochs
        self.hvp_eps = hvp_eps
        self.loss_strategy = loss_strategy
        self.output_dir = output_dir
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.target_feature = target_feature
        self.kwargs = kwargs

        # 2. State containers
        self.is_trained_ = False
        self.args_ = None
        self.context_ = None
        self.train_results_ = None

        # 3. Output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def _build_params(self):
        params = {
            "denoising_project_root": self.denoising_project_root,
            "dataset": self.dataset,
            "device": self.device,
            "detection_methods": self.detection_methods,
            "only_det_missing": self.only_det_missing,
            "loss_data": self.loss_data,
            "seed": self.seed,
            "test_split": self.test_split,
            "val_split": self.val_split,
            "batch_size": self.batch_size,
            "repair_model": self.repair_model,
            "classifier_type": self.classifier_type,
            "use_gradnorm": self.use_gradnorm,
            "classi_epochs": self.classi_epochs,
            "classi_lr": self.classi_lr,
            "time_emb_dim": self.time_emb_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "dropout": self.dropout,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "enable_pretraining": self.enable_pretraining,
            "use_diffusion": self.use_diffusion,
            "diffusion_embed": self.diffusion_embed,
            "use_diffusion_loss": self.use_diffusion_loss,
            "lambda_diff": self.lambda_diff,
            "lambda_explicit_bilevel": self.lambda_explicit_bilevel,
            "schedule": self.schedule,
            "num_timesteps": self.num_timesteps,
            "classi_hidden_dim": self.classi_hidden_dim,
            "l2_lambda_classifier": self.l2_lambda_classifier,
            "patience": self.patience,
            "align_step_size": self.align_step_size,
            "damping": self.damping,
            "hvp_batch_size": self.hvp_batch_size,
            "num_iterations": self.num_iterations,
            "inner_steps": self.inner_steps,
            "outer_steps": self.outer_steps,
            "mu_align": self.mu_align,
            "outer_lr": self.outer_lr,
            "outer_epochs": self.outer_epochs,
            "hvp_eps": self.hvp_eps,
            "loss_strategy": self.loss_strategy,
            "output_dir": self.output_dir,
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "target_feature": self.target_feature,
        }
        params.update(self.kwargs)
        return params

    def train(self, dirty_csv=None, detection_mask=None, data_config=None, **kwargs):
        params = self._build_params()
        params.update(kwargs)

        self.args_, self.context_, self.train_results_ = modules.train_torepair_pipeline(
            params=params,
            dirty_csv=dirty_csv,
            detection_mask=detection_mask,
            data_config=data_config,
        )
        self.is_trained_ = True
        return self

    def predict(self, dirty_csv=None, detection_mask=None, dataset_name=None):
        if not self.is_trained_:
            raise RuntimeError("Model is not trained. Run .train() first.")

        repaired_data = modules.repair_dataframe(
            context=self.context_,
            dirty_csv=dirty_csv,
            detection_mask=detection_mask,
            dataset_name=dataset_name,
        )
        return repaired_data

    def train_and_predict(self, dirty_csv=None, detection_mask=None, data_config=None, **kwargs):
        self.train(dirty_csv=dirty_csv, detection_mask=detection_mask, data_config=data_config, **kwargs)
        return self.predict(dirty_csv=dirty_csv, detection_mask=detection_mask)
