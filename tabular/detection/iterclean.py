import os

import pandas as pd

try:
    from dataprep.base import BaseEstimator
    import dataprep.tabular.detection.iterclean_modules as modules
except ModuleNotFoundError:
    from base import BaseEstimator
    import tabular.detection.iterclean_modules as modules


class IterClean(BaseEstimator):
    """DataPrep wrapper for the detection part of IterClean.

    IterClean is prompt-driven, so the training step only stores runtime
    configuration. The detection result returned by ``predict`` is a boolean
    DataFrame with the same shape as the input data.
    """

    def __init__(self,
                 model_name="gpt-3.5-turbo",
                 api_use=True,
                 api_key=None,
                 base_url=None,
                 llm_backend="openai",
                 prompt_detect=None,
                 prompt_detect_path=None,
                 prompt_verify=None,
                 prompt_verify_path=None,
                 use_verify=True,
                 batch_size=5,
                 max_workers=4,
                 batch_mode="ourbat",
                 dataset_name=None,
                 ref_column=None,
                 result_dir="./result",
                 save_responses=True,
                 verbose=True,
                 **kwargs):
        # 1. Attribute assignment
        self.model_name = model_name
        self.api_use = api_use
        self.api_key = api_key
        self.base_url = base_url
        self.llm_backend = llm_backend
        self.prompt_detect = prompt_detect
        self.prompt_detect_path = prompt_detect_path
        self.prompt_verify = prompt_verify
        self.prompt_verify_path = prompt_verify_path
        self.use_verify = use_verify
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.batch_mode = batch_mode
        self.dataset_name = dataset_name
        self.ref_column = ref_column
        self.result_dir = result_dir
        self.save_responses = save_responses
        self.verbose = verbose
        self.kwargs = kwargs

        # 2. State containers
        self.is_trained_ = False
        self.detection_records_ = []
        self.detection_reasons_ = []
        self.raw_detection_responses_ = []
        self.raw_verify_responses_ = []

        # 3. Logger
        os.makedirs(self.result_dir, exist_ok=True)
        self.logger = modules.Logger(self.result_dir, verbose=verbose)

    def _build_params(self):
        return {
            "model_name": self.model_name,
            "api_use": self.api_use,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "llm_backend": self.llm_backend,
            "prompt_detect": self.prompt_detect,
            "prompt_detect_path": self.prompt_detect_path,
            "prompt_verify": self.prompt_verify,
            "prompt_verify_path": self.prompt_verify_path,
            "use_verify": self.use_verify,
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
            "batch_mode": self.batch_mode,
            "dataset_name": self.dataset_name,
            "ref_column": self.ref_column,
            "result_dir": self.result_dir,
            "save_responses": self.save_responses,
            **self.kwargs,
        }

    def train(self, dirty_csv=None, **kwargs):
        """Prepare the prompt-driven detector.

        The original IterClean detection stage does not fit a local model. This
        method keeps the DataPrep estimator interface and validates the runtime
        configuration.
        """
        params = self._build_params()
        params.update(kwargs)
        modules.validate_params(params)
        self.params_ = params
        self.is_trained_ = True
        return self

    def predict(self, dirty_csv):
        if not self.is_trained_:
            self.train(dirty_csv)

        self.logger.info(f"Starting IterClean detection on shape {dirty_csv.shape}.")
        result = modules.detect_dataframe(dirty_csv, self.params_, self.logger)

        self.detection_records_ = result.records
        self.detection_reasons_ = result.reasons
        self.raw_detection_responses_ = result.raw_detection_responses
        self.raw_verify_responses_ = result.raw_verify_responses

        self.logger.info(f"IterClean detection finished with {len(result.records)} cells flagged.")
        return result.mask

    def train_and_predict(self, dirty_csv, **kwargs):
        self.train(dirty_csv, **kwargs)
        return self.predict(dirty_csv)
