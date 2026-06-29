import pandas as pd
import torch

try:
    from dataprep.tabular.imputation.base import BaseImputer
    import dataprep.tabular.imputation.MISS_tabformer_modules as mtfm
except ImportError:
    from tabular.imputation.base import BaseImputer
    import tabular.imputation.MISS_tabformer_modules as mtfm


class MISSTabFormer(BaseImputer):
    """DataPrep wrapper for MISS-tabformer.

    This follows the original TabSurvey `TabTransformer_our` flow: expert
    sampling + IPS, mean-filled model inputs, categorical transformer pretrain,
    frozen transformer, then supervised downstream training.
    """

    def __init__(
            self,
            batch_size=128,
            val_batch_size=128,
            epoch=1000,
            pretrain_epoch=20,
            lr=1e-5,
            weight_decay=1e-5,
            dim=128,
            depth=1,
            heads=4,
            dropout=0.4,
            task=None,
            ips_num=20,
            ips_method="xgb",
            sampling_method="feature",
            observed_num=1,
            complete_sample="no-Random",
            datasplit=None,
            cat_idxs=None,
            cat_threshold=100,
            early_stopping_rounds=20,
            device=None,
            random_state=0,
            dset_seed=None,
            label_column=None,
            return_dataframe=False,
            verbose=False):
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.epoch = epoch
        self.pretrain_epoch = pretrain_epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dropout = dropout
        self.task = task
        self.ips_num = ips_num
        self.ips_method = ips_method
        self.sampling_method = sampling_method
        self.observed_num = observed_num
        self.complete_sample = complete_sample
        self.datasplit = datasplit if datasplit is not None else [.8, .1, .1]
        self.cat_idxs = cat_idxs
        self.cat_threshold = cat_threshold
        self.early_stopping_rounds = early_stopping_rounds
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
            "val_batch_size": self.val_batch_size,
            "epoch": self.epoch,
            "pretrain_epoch": self.pretrain_epoch,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "dim": self.dim,
            "depth": self.depth,
            "heads": self.heads,
            "dropout": self.dropout,
            "task": self.task,
            "ips_num": self.ips_num,
            "ips_method": self.ips_method,
            "sampling_method": self.sampling_method,
            "observed_num": self.observed_num,
            "complete_sample": self.complete_sample,
            "datasplit": self.datasplit,
            "cat_idxs": self.cat_idxs,
            "cat_threshold": self.cat_threshold,
            "early_stopping_rounds": self.early_stopping_rounds,
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
                "Strict MISS-tabformer requires full_data for the original "
                "expert-sampling and IPS computation."
            )

        self._remember_structure(data)
        result = mtfm.train_miss_tabformer_algorithm(
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
            raise RuntimeError("MISS_tabformer_modules did not return a trained model.")
        return self

    def predict(self, data, missing_mask=None):
        if self.model is None:
            raise RuntimeError("Model needs to be trained first. Call train() or train_and_predict().")

        was_dataframe = isinstance(data, pd.DataFrame)
        result = mtfm.predict_miss_tabformer_algorithm(
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


MISSTabformer = MISSTabFormer
