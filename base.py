from abc import ABC, abstractmethod
try:
    import joblib  # 或者使⽤ pickle/cloudpickle
except ModuleNotFoundError:
    import pickle

    class _PickleJoblibFallback:
        @staticmethod
        def dump(obj, path):
            with open(path, "wb") as f:
                pickle.dump(obj, f)

        @staticmethod
        def load(path):
            with open(path, "rb") as f:
                return pickle.load(f)

    joblib = _PickleJoblibFallback()
import tempfile
import shutil
import os
from pathlib import Path

class BaseEstimator(ABC):

    @abstractmethod
    def train(self, *args, **kwargs):
        """
        训练模型。具体参数由各任务的抽象基类定义。
        对于⽆需训练的模型，可以为空实现，直接 return self。
        """
        raise NotImplementedError


    @abstractmethod
    def predict(self, *args, **kwargs):
        """
        使⽤已训练的模型进⾏核⼼转换操作。具体参数由各任务的抽象基类定义。
        """
        raise NotImplementedError

    @abstractmethod
    def train_and_predict(self, *args, **kwargs):
        """先训练，后转换。"""
        # 注意：这⾥的参数传递需要更精细的设计，或者假设train和predict的输⼊参数有重叠
        # ⼀个简单的实现如下，但具体实现可能需要调整
        self.train(*args, **kwargs)
        return self.predict(*args, **kwargs)


    def save_model(self, path: str):
        """将训练好的模型实例保存到⽂件。"""
        # 使⽤ joblib 可以更好地处理包含numpy数组的对象
        joblib.dump(self, path)
        print(f"Model saved to {path}")


    @staticmethod
    def load_model(path: str):
        """从⽂件加载模型实例。"""
        model = joblib.load(path)
        print(f"Model loaded from {path}")
        return model

    def _create_temp_dir(self, prefix="model_run_"):
        """
        创建一个唯一的临时目录。
        prefix: 目录的前缀，方便识别 (例如 'gain_', 'scis_')
        """
        parent_dir = Path("./temp")
        parent_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = tempfile.mkdtemp(prefix=prefix, dir='./temp')
        print(f"[System] Temporary checkpoint directory created at: {self.temp_dir}")
        return self.temp_dir

    def _save_checkpoint(self, filename="checkpoint.pth"):
        """
        将当前对象(模型)保存到临时目录中。
        """
        if self.temp_dir is None:
            raise RuntimeError("Temporary directory not created. Call _create_temp_dir first.")

        save_path = os.path.join(self.temp_dir, filename)
        self.save_model(save_path)
        return save_path

    def cleanup(self):
        """
        训练结束后清理临时目录
        """
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"[System] Cleaned up temporary directory: {self.temp_dir}")
            self.temp_dir = None
