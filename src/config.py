import random
from datetime import datetime
from pathlib import Path
from typing import Optional
import torch
from pydantic import BaseModel


class RegressorConfig(BaseModel):
    csv_file: Path = Path("/home/student/work/filtered_train.csv")
    img_dir: Path = Path("/home/student/work/train/")

    heatmap_size: int = 64


class ClassifierConfig(BaseModel):
    dataset: Path = "./data/raw/classifier"
    tflite: Path = Path()

    num_classes: int = 4


class Config(BaseModel):
    # === Common ===
    img_size: int = 224
    batch: int = 64

    checkpoint: Optional[Path] = None

    def init_checkpoint(self, name: str):
        self.checkpoint = Path(
            "./data/checkpoints/" + f"{name}_{datetime.now().strftime('%m%d_%H:%M:%S')}"
        )
        self.checkpoint.mkdir(parents=True, exist_ok=False)

    classifier: ClassifierConfig = ClassifierConfig()
    regressor: RegressorConfig = RegressorConfig()

    # === Eval settings ===
    eval_dataset: Path = Path(
        "/home/syrenny/Desktop/clones/keypoint_regression/data/raw/test"
    )


torch.manual_seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()
