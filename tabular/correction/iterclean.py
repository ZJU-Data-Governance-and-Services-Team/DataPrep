import os

try:
    from dataprep.base import BaseEstimator
    import dataprep.tabular.correction.iterclean_modules as modules
except ModuleNotFoundError:
    from base import BaseEstimator
    import tabular.correction.iterclean_modules as modules


class IterClean(BaseEstimator):
    """DataPrep wrapper for the correction part of IterClean.

    If detection results are not provided to ``predict``, this wrapper first
    runs the IterClean detection/verification pipeline and then runs repair.
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
                 prompt_repair=None,
                 prompt_repair_path=None,
                 use_verify=True,
                 batch_size=5,
                 max_round=1,
                 batch_size_step=2,
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
        self.prompt_repair = prompt_repair
        self.prompt_repair_path = prompt_repair_path
        self.use_verify = use_verify
        self.batch_size = batch_size
        self.max_round = max_round
        self.batch_size_step = batch_size_step
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
        self.repair_records_ = []
        self.raw_detection_responses_ = []
        self.raw_verify_responses_ = []
        self.raw_repair_responses_ = []

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
            "prompt_repair": self.prompt_repair,
            "prompt_repair_path": self.prompt_repair_path,
            "use_verify": self.use_verify,
            "batch_size": self.batch_size,
            "max_round": self.max_round,
            "batch_size_step": self.batch_size_step,
            "max_workers": self.max_workers,
            "batch_mode": self.batch_mode,
            "dataset_name": self.dataset_name,
            "ref_column": self.ref_column,
            "result_dir": self.result_dir,
            "save_responses": self.save_responses,
            **self.kwargs,
        }

    def train(self, dirty_csv=None, **kwargs):
        params = self._build_params()
        params.update(kwargs)
        modules.validate_params(params)
        self.params_ = params
        self.is_trained_ = True
        return self

    def predict(self,
                dirty_csv,
                detection_mask=None,
                detection_records=None,
                detection_reasons=None):
        if not self.is_trained_:
            self.train(dirty_csv)

        self.logger.info(f"Starting IterClean correction on shape {dirty_csv.shape}.")
        result = modules.repair_dataframe(
            dirty_csv,
            self.params_,
            self.logger,
            detection_mask=detection_mask,
            detection_records=detection_records,
            detection_reasons=detection_reasons,
        )

        self.detection_records_ = result.detection_records
        self.detection_reasons_ = result.detection_reasons
        self.repair_records_ = result.repair_records
        self.raw_detection_responses_ = result.raw_detection_responses
        self.raw_verify_responses_ = result.raw_verify_responses
        self.raw_repair_responses_ = result.raw_repair_responses

        self.logger.info(f"IterClean correction finished with {len(result.repair_records)} cells repaired.")
        return result.corrected_df

    def train_and_predict(self,
                          dirty_csv,
                          detection_mask=None,
                          detection_records=None,
                          detection_reasons=None,
                          **kwargs):
        self.train(dirty_csv, **kwargs)
        return self.predict(
            dirty_csv,
            detection_mask=detection_mask,
            detection_records=detection_records,
            detection_reasons=detection_reasons,
        )
