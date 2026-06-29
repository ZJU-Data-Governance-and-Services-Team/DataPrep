import pandas as pd
import torch

try:
    from dataprep.tabular.imputation.base import BaseImputer
    import dataprep.tabular.imputation.MISS_tabular_modules as mtm
except ImportError:
    from tabular.imputation.base import BaseImputer
    import tabular.imputation.MISS_tabular_modules as mtm


class MISSTabular(BaseImputer):
    """DataPrep wrapper for MISS-tabular.

    MISS-tabular follows the supervised FT-Transformer training flow in the
    original `code/tabular/.../train5.py`: `data` is the table with missing
    values, `full_data` is the clean table used by the expert-sampling/IPS step,
    and the label column defaults to the last column.
    """

    def __init__(
            self,
            batch_size=256,
            epoch=200,
            lr=0.0001,
            weight_decay=1e-5,
            d_token=32,
            n_blocks=3,
            attention_heads=8,
            attention_dropout=0.2,
            ffn_dropout=0.1,
            residual_dropout=0.0,
            task=None,
            ips_num=40,
            ips_method="xgb",
            sampling_method="feature",
            observed_num=1,
            complete_sample="no-Random",
            mask_rate=0.1,
            weights_cat=1.0,
            weights_num=1.0,
            datasplit=None,
            cat_idxs=None,
            cat_threshold=100,
            optimizer="AdamW",
            patience=None,
            device=None,
            random_state=0,
            dset_seed=None,
            label_column=None,
            return_dataframe=False,
            verbose=False):
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.d_token = d_token
        self.n_blocks = n_blocks
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.task = task
        self.ips_num = ips_num
        self.ips_method = ips_method
        self.sampling_method = sampling_method
        self.observed_num = observed_num
        self.complete_sample = complete_sample
        self.mask_rate = mask_rate
        self.weights_cat = weights_cat
        self.weights_num = weights_num
        self.datasplit = datasplit if datasplit is not None else [.8, .1, .1]
        self.cat_idxs = cat_idxs
        self.cat_threshold = cat_threshold
        self.optimizer = optimizer
        self.patience = patience
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
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "d_token": self.d_token,
            "n_blocks": self.n_blocks,
            "attention_heads": self.attention_heads,
            "attention_dropout": self.attention_dropout,
            "ffn_dropout": self.ffn_dropout,
            "residual_dropout": self.residual_dropout,
            "task": self.task,
            "ips_num": self.ips_num,
            "ips_method": self.ips_method,
            "sampling_method": self.sampling_method,
            "observed_num": self.observed_num,
            "complete_sample": self.complete_sample,
            "mask_rate": self.mask_rate,
            "weights_cat": self.weights_cat,
            "weights_num": self.weights_num,
            "datasplit": self.datasplit,
            "cat_idxs": self.cat_idxs,
            "cat_threshold": self.cat_threshold,
            "optimizer": self.optimizer,
            "patience": self.patience,
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
                "Strict MISS-tabular requires full_data. Pass the clean/full "
                "table matching the original train5.py `datafull` input."
            )

        self._remember_structure(data)
        result = mtm.train_miss_tabular_algorithm(
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
            raise RuntimeError("MISS_tabular_modules did not return a trained model.")
        return self

    def predict(self, data, missing_mask=None):
        if self.model is None:
            raise RuntimeError("Model needs to be trained first. Call train() or train_and_predict().")

        was_dataframe = isinstance(data, pd.DataFrame)
        result = mtm.predict_miss_tabular_algorithm(
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
