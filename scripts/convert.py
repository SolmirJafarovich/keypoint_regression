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
from torch.utils.data import Subset

from src.dataset import DepthKeypointDataset
from src.config import config, device
from src.models import BlazePoseLite, CombinedClassifier
app = typer.Typer(pretty_exceptions_enable=False)


def calibrate(model):
    transform = transforms.Compose(
        [
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    dataset = DepthKeypointDataset(
        transform=transform
    )
    data_loader = DataLoader(
        dataset=Subset(dataset, list(range(500))),
        batch_size=config.batch,
        shuffle=True,
    )
    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"]
            print("Model input dtype:", images.dtype)
            print(images.shape, images.dtype, images.device)
            print(next(model.parameters()).device)
            _ = model(images.to(torch.float32))


@app.command()
def convert(
    checkpoint: Path = typer.Option("--checkpoint"),
    is_classifier: Annotated[bool, typer.Option("--is-classifier")] = False,
):
    # === Load and initialize model ===

    if is_classifier:
        model_fp32 = CombinedClassifier()
        sample_input = (
            torch.randn(1, 1, config.img_size, config.img_size, dtype=torch.float),
            torch.randn(1, 66, dtype=torch.float),
        )
    else:
        model_fp32 = BlazePoseLite()
        sample_input = (torch.randn(1, 1, config.img_size, config.img_size, dtype=torch.float),)

    state_dict = torch.load(
        checkpoint / "weights.pth", weights_only=True
    )
    model_fp32.load_state_dict(state_dict)
    model_fp32.eval()

    # === Configure quantization ===

    quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False)
    )

    # === Quantize ===

    # model_fp32 = capture_pre_autograd_graph(model_fp32, sample_input)
    model_fp32 = torch.export.export_for_training(model_fp32, sample_input).module()
    model_fp32 = prepare_pt2e(model_fp32, quantizer)

    def check_input_dtype(module, input, output):
        if isinstance(input, tuple):
            input = input[0]
        if hasattr(input, 'dtype') and input.dtype != torch.float32:
            print(f"⚠️ {module.__class__.__name__} получает {input.dtype}")
    
    model_fp32.apply(lambda m: m.register_forward_hook(check_input_dtype))

    # TODO calibration

    calibrate(model_fp32)

    model_i8 = convert_pt2e(model_fp32, fold_quantize=True)

    # === Convert to an ai_edge_torch model ===

    tflite_model = ai_edge_torch.convert(
        module=model_i8,
        sample_args=sample_input,
        quant_config=QuantConfig(pt2e_quantizer=quantizer),
    )

    # tfl_converter_flags = {'optimizations': [tf.lite.Optimize.DEFAULT]}

    # tflite_model = ai_edge_torch.convert(
    #     model_fp32, sample_input, _ai_edge_converter_flags=tfl_converter_flags
    # )

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
