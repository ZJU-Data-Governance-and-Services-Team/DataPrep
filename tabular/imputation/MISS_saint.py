import numpy as np
import pandas as pd
import torch

try:
    from dataprep.tabular.imputation.base import BaseImputer
    import dataprep.tabular.imputation.MISS_saint_modules as msm
except ImportError:
    from tabular.imputation.base import BaseImputer
    import tabular.imputation.MISS_saint_modules as msm


class MISSSaint(BaseImputer):
    """DataPrep wrapper for the original MISS-SAINT training flow.

    Strict MISS-SAINT is supervised: the missing table and full table must both
    include the label column, or labels must be passed separately. The default
    label column is the last column, matching code/saint/saint-main/train.py.
    """

    def __init__(
            self,
            batch_size=256,
            epoch=200,
            lr=0.0001,
            embedding_size=32,
            transformer_depth=6,
            attention_heads=8,
            attention_dropout=0.1,
            ff_dropout=0.1,
            cont_embeddings="MLP",
            attentiontype="col",
            final_mlp_style="sep",
            optimizer="AdamW",
            scheduler="cosine",
            task=None,
            ips_num=40,
            ips_method="xgb",
            sampling_method="feature",
            observed_num=1,
            complete_sample="no-Random",
            datasplit=None,
            cat_idxs=None,
            cat_threshold=100,
            include_reconstruction_loss=False,
            lam2=1,
            lam3=10,
            device=None,
            random_state=0,
            dset_seed=None,
            label_column=None,
            return_dataframe=False,
            verbose=False):
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr
        self.embedding_size = embedding_size
        self.transformer_depth = transformer_depth
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout
        self.ff_dropout = ff_dropout
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.task = task
        self.ips_num = ips_num
        self.ips_method = ips_method
        self.sampling_method = sampling_method
        self.observed_num = observed_num
        self.complete_sample = complete_sample
        self.datasplit = datasplit if datasplit is not None else [.8, .1, .1]
        self.cat_idxs = cat_idxs
        self.cat_threshold = cat_threshold
        self.include_reconstruction_loss = include_reconstruction_loss
        self.lam2 = lam2
        self.lam3 = lam3
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
            "embedding_size": self.embedding_size,
            "transformer_depth": self.transformer_depth,
            "attention_heads": self.attention_heads,
            "attention_dropout": self.attention_dropout,
            "ff_dropout": self.ff_dropout,
            "cont_embeddings": self.cont_embeddings,
            "attentiontype": self.attentiontype,
            "final_mlp_style": self.final_mlp_style,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "task": self.task,
            "ips_num": self.ips_num,
            "ips_method": self.ips_method,
            "sampling_method": self.sampling_method,
            "observed_num": self.observed_num,
            "complete_sample": self.complete_sample,
            "datasplit": self.datasplit,
            "cat_idxs": self.cat_idxs,
            "cat_threshold": self.cat_threshold,
            "include_reconstruction_loss": self.include_reconstruction_loss,
            "lam2": self.lam2,
            "lam3": self.lam3,
            "device": self.device,
            "random_state": self.random_state,
            "dset_seed": self.dset_seed,
            "verbose": self.verbose,
        }
        params.update(overrides)
        return params

    def train(self, data, missing_mask=None, full_data=None, labels=None,
              label_column=None, **kwargs):
        """Train MISS-SAINT.

        Args:
            data: missing data table (`datamiss` in original train.py).
            missing_mask: accepted for BaseImputer compatibility; strict mode
                recomputes masks from NaNs after `sampling()`.
            full_data: clean/full table (`datafull` in original train.py).
            labels: optional labels if data/full_data do not include a label
                column.
            label_column: label column name. Defaults to the last column.
        """
        if full_data is None:
            raise ValueError(
                "Strict MISS-SAINT requires full_data. Pass the clean/full table "
                "matching the original train.py `datafull` input."
            )

        self._remember_structure(data)
        params = self._params(kwargs)
        result = msm.train_miss_saint_algorithm(
            data=data,
            missing_mask=missing_mask,
            params=params,
            device=self.device,
            full_data=full_data,
            labels=labels,
            label_column=label_column or self.label_column,
        )

        self.model = result.get("model")
        self.train_state = result
        if self.model is None:
            raise RuntimeError("MISS_saint_modules did not return a trained model.")
        return self

    def predict(self, data, missing_mask=None):
        if self.model is None:
            raise RuntimeError("Model needs to be trained first. Call train() or train_and_predict().")

        was_dataframe = isinstance(data, pd.DataFrame)
        result = msm.predict_miss_saint_algorithm(
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
