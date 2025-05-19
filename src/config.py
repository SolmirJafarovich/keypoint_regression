import random
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class RegressorConfig(BaseModel):
    csv_file: Path = Path("../filtered_train.csv")
    img_dir: Path = Path("../train")

    heatmap_size: int = 64


class ClassifierConfig(BaseModel):
    image_root: Path = Path("../depth")  # Путь к папке с depth/Belly, depth/Back и т.д.
    keypoints_json: Path = Path("./data/raw/cached_keypoints1_flattened.json")  # Путь к JSON с предсказанными ключевыми точками
    tflite: Path = Path("./checkpoints/classifier.tflite")  # Путь, куда сохраняется tflite-модель (если нужно)

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
