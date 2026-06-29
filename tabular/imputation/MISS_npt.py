import pandas as pd
import torch

try:
    from dataprep.tabular.imputation.base import BaseImputer
    import dataprep.tabular.imputation.MISS_npt_modules as mnm
except ImportError:
    from tabular.imputation.base import BaseImputer
    import tabular.imputation.MISS_npt_modules as mnm


class MISSNPT(BaseImputer):
    """DataPrep wrapper for MISS-NPT."""

    def __init__(
            self,
            batch_size=128,
            epoch=200,
            pretrain_epoch=20,
            enable_pretrain=False,
            lr=0.001,
            weight_decay=0.0,
            dim_hidden=64,
            stacking_depth=4,
            num_heads=8,
            hidden_dropout=0.1,
            attention_dropout=0.1,
            task=None,
            ips_num=40,
            ips_method="xgb",
            sampling_method="feature",
            observed_num=1,
            complete_sample="no-Random",
            datasplit=None,
            cat_idxs=None,
            cat_threshold=100,
            reconstruction_weight=1.0,
            contrastive_weight=1.0,
            supervised_reconstruction_weight=0.0,
            early_stopping_rounds=5,
            eval_every_n=5,
            total_steps=100000,
            flat_lr_proportion=0.7,
            gradient_clip=1.0,
            lookahead_alpha=0.5,
            lookahead_update_cadence=6,
            device=None,
            random_state=42,
            dset_seed=None,
            label_column=None,
            return_dataframe=False,
            verbose=False):
        self.batch_size = batch_size
        self.epoch = epoch
        self.pretrain_epoch = pretrain_epoch
        self.enable_pretrain = enable_pretrain
        self.lr = lr
        self.weight_decay = weight_decay
        self.dim_hidden = dim_hidden
        self.stacking_depth = stacking_depth
        self.num_heads = num_heads
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.task = task
        self.ips_num = ips_num
        self.ips_method = ips_method
        self.sampling_method = sampling_method
        self.observed_num = observed_num
        self.complete_sample = complete_sample
        self.datasplit = datasplit if datasplit is not None else [.8, .1, .1]
        self.cat_idxs = cat_idxs
        self.cat_threshold = cat_threshold
        self.reconstruction_weight = reconstruction_weight
        self.contrastive_weight = contrastive_weight
        self.supervised_reconstruction_weight = supervised_reconstruction_weight
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_every_n = eval_every_n
        self.total_steps = total_steps
        self.flat_lr_proportion = flat_lr_proportion
        self.gradient_clip = gradient_clip
        self.lookahead_alpha = lookahead_alpha
        self.lookahead_update_cadence = lookahead_update_cadence
        self.random_state = random_state
        self.dset_seed = dset_seed if dset_seed is not None else random_state
        self.label_column = label_column
        self.return_dataframe = return_dataframe
        self.verbose = verbose
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.train_state = None
        self.columns = None
        self.index = None

    def _remember_structure(self, data):
        if isinstance(data, pd.DataFrame):
            self.columns = data.columns
            self.index = data.index

    def _params(self, overrides):
        params = {
            "batch_size": self.batch_size,
            "epoch": self.epoch,
            "pretrain_epoch": self.pretrain_epoch,
            "enable_pretrain": self.enable_pretrain,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "dim_hidden": self.dim_hidden,
            "stacking_depth": self.stacking_depth,
            "num_heads": self.num_heads,
            "hidden_dropout": self.hidden_dropout,
            "attention_dropout": self.attention_dropout,
            "task": self.task,
            "ips_num": self.ips_num,
            "ips_method": self.ips_method,
            "sampling_method": self.sampling_method,
            "observed_num": self.observed_num,
            "complete_sample": self.complete_sample,
            "datasplit": self.datasplit,
            "cat_idxs": self.cat_idxs,
            "cat_threshold": self.cat_threshold,
            "reconstruction_weight": self.reconstruction_weight,
            "contrastive_weight": self.contrastive_weight,
            "supervised_reconstruction_weight": self.supervised_reconstruction_weight,
            "early_stopping_rounds": self.early_stopping_rounds,
            "eval_every_n": self.eval_every_n,
            "total_steps": self.total_steps,
            "flat_lr_proportion": self.flat_lr_proportion,
            "gradient_clip": self.gradient_clip,
            "lookahead_alpha": self.lookahead_alpha,
            "lookahead_update_cadence": self.lookahead_update_cadence,
            "device": self.device,
            "random_state": self.random_state,
            "dset_seed": self.dset_seed,
            "verbose": self.verbose,
        }
        params.update(overrides)
        return params

    def train(self, data, missing_mask=None, full_data=None, labels=None,
              label_column=None, **kwargs):
        if full_data is None:
            raise ValueError(
                "Strict MISS-NPT requires full_data for expert sampling and IPS."
            )

        self._remember_structure(data)
        result = mnm.train_miss_npt_algorithm(
            data=data,
            missing_mask=missing_mask,
            params=self._params(kwargs),
            device=self.device,
            full_data=full_data,
            labels=labels,
            label_column=label_column or self.label_column,
        )
        self.model = result.get("model")
        self.train_state = result
        if self.model is None:
            raise RuntimeError("MISS_npt_modules did not return a trained model.")
        return self

    def predict(self, data, missing_mask=None):
        if self.model is None:
            raise RuntimeError("Model needs to be trained first. Call train() or train_and_predict().")

        was_dataframe = isinstance(data, pd.DataFrame)
        result = mnm.predict_miss_npt_algorithm(
            model=self.model,
            data=data,
            missing_mask=missing_mask,
            train_state=self.train_state,
            device=self.device,
        )
        if self.return_dataframe or was_dataframe:
            return result
        return result.to_numpy()

    def train_and_predict(self, data, missing_mask=None, full_data=None, labels=None,
                          label_column=None, **kwargs):
        self.train(
            data,
            missing_mask=missing_mask,
            full_data=full_data,
            labels=labels,
            label_column=label_column,
            **kwargs,
        )
        return self.predict(data, missing_mask=missing_mask)


MISSNpt = MISSNPT
