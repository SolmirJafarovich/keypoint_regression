from pathlib import Path
from typing import Annotated

import ai_edge_torch
import torch
import typer
from ai_edge_torch.quantize.pt2e_quantizer import (
    PT2EQuantizer,
    get_symmetric_quantization_config,
)
from ai_edge_torch.quantize.quant_config import QuantConfig
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.config import config, device
from src.dataset import DepthKeypointDataset
from src.models import BlazePoseLite, CombinedClassifier

app = typer.Typer()


def calibrate(model):
    limb_connections = [
        (0, 1),
        (1, 2),
        (2, 3),  # левая часть лица
        (0, 4),
        (4, 5),
        (5, 6),  # правая часть лица
        (2, 7),
        (3, 7),  # левое ухо
        (5, 8),
        (6, 8),  # правое ухо
        (7, 9),
        (8, 10),  # рот
        (9, 10),  # соединение рта
        (11, 12),  # плечи
        (12, 14),
        (14, 16),  # правая рука
        (16, 18),
        (18, 20),
        (16, 20),  # правая кисть
        (11, 13),
        (13, 15),  # левая рука
        (15, 17),
        (17, 19),
        (15, 19),  # левая кисть
        (11, 23),
        (12, 24),  # туловище к бедрам
        (23, 25),
        (25, 27),
        (27, 29),
        (29, 31),  # левая нога
        (23, 24),  # соединение бедер
        (24, 26),
        (26, 28),
        (28, 30),
        (30, 32),  # правая нога
    ]
    transform = transforms.Compose(
        [
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    dataset = DepthKeypointDataset(
        transform=transform, limb_connections=limb_connections
    )
    data_loader = DataLoader(
        dataset[:500],
        batch_size=config.batch,
        shuffle=True,
        pin_memory=True,
    )
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Calibration"):
            images = batch["image"].to(device)
            model(images)


@app.command()
def convert(
    checkpoint: Path = typer.Option("--checkpoint"),
    is_classifier: Annotated[bool, typer.Option("--is-classifier")] = False,
):
    # === Load and initialize model ===

    if is_classifier:
        model_fp32 = CombinedClassifier()
        sample_input = (
            torch.randn(1, 1, config.img_size, config.img_size),
            torch.randn(1, 66),
        )
    else:
        model_fp32 = BlazePoseLite()
        sample_input = (torch.randn(1, 1, config.img_size, config.img_size),)

    state_dict = torch.load(
        checkpoint / "weights.pth", map_location=device, weights_only=True
    )
    model_fp32.load_state_dict(state_dict)
    model_fp32.eval()

    # === Configure quantization ===

    quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=False, is_dynamic=True)
    )

    # === Quantize ===

    model_fp32 = capture_pre_autograd_graph(model_fp32, sample_input)
    model_fp32 = prepare_pt2e(model_fp32, quantizer)

    # TODO calibration

    calibrate(model_fp32)

    model_i8 = convert_pt2e(model_fp32, fold_quantize=False)

    # === Convert to an ai_edge_torch model ===

    tflite_model = ai_edge_torch.convert(
        module=model_i8,
        sample_args=sample_input,
        quant_config=QuantConfig(pt2e_quantizer=quantizer),
    )

    # === Test the model ===

    # Run the prepared model with sample input data to ensure that internal observers are populated with correct values
    pytorch_output = model_fp32(*sample_input)
    tflite_output = tflite_model(*sample_input)

    print("PyTorch output:", pytorch_output)
    print("TFLite output:", tflite_output)

    # === Save the model ===

    tflite_path = checkpoint / "weights.tflite"
    tflite_model.export(tflite_path)

    typer.echo(f"Готово! Модель сохранена как: {tflite_path}")


if __name__ == "__main__":
    app()
