import random
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class RegressorConfig(BaseModel):
    csv_file: Path = Path("./data/raw/subset/subset.csv")
    img_dir: Path = Path("./data/raw/subset/images")

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
    eval_dataset: Path = Path("./data/raw/test")


random.seed(42)

config = Config()
