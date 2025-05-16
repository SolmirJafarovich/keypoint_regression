import datetime
import random
from pathlib import Path

import torch
from pydantic import BaseModel


class RegressorConfig(BaseModel):
    csv_file: Path = Path("./data/raw/regressor")
    img_dir: Path = Path("/home/student/work/train/")

    heatmap_size: int = 64


class ClassifierConfig(BaseModel):
    dataset: Path = "./data/raw/classifier"
    tflite: Path = Path()

    num_classes: int = 4


class Config(BaseModel):
    # === Common ===
    img_size: int = 224

    checkpoint: Path | None = None

    def init_checkpoint(self, name: str) -> Path:
        self.checkpoint = (
            "./data/checkpoints/" + f"{name}_{datetime.now().strftime('%m%d_%H:%M:%S')}"
        )

    classifier: ClassifierConfig = ClassifierConfig()
    regressor: RegressorConfig = RegressorConfig()

    # === Eval settings ===
    eval_dataset: Path = Path("./data/raw/test")


torch.manual_seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()
