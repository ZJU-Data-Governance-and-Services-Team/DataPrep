import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(project_root)
from dataprep.tabular.correction.zeroec import ZeroEC

zeroec = ZeroEC(model_name="qwen2.5-7b",
                 openai_api_base="http://localhost:8000/v1",
                 openai_api_key="EMPTY",
                 embedding_model_path='../correction/all-MiniLM-L6-v2',
                 human_repair_num=10,
                 output_dir='../correction/runs_rayyan',
                 clean_data_path='../correction/datasets/rayyan/rayyan_clean.csv',
                 dirty_data_path='../correction/datasets/rayyan/rayyan_dirty.csv',
                 detection_path='../correction/datasets/rayyan/rayyan_dirty_error_detection.csv',
                 prompt_dir='/../correction/prompt_templates',
                 max_workers=3,)
zeroec.train_and_predict()