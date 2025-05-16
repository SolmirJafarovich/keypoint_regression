from pathlib import Path

import ai_edge_torch
import torch
import typer
from ai_edge_torch.quantize.pt2e_quantizer import (
    PT2EQuantizer,
    get_symmetric_quantization_config,
)
from ai_edge_torch.quantize.quant_config import QuantConfig
from model import BlazePoseLite
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

app = typer.Typer()


@app.command()
def convert(checkpoint: Path = typer.Option("--checkpoint")):
    # === Load model ===
    model = BlazePoseLite()
    model.load_state_dict(
        torch.load(
            checkpoint / "weights.pth",
            map_location="cuda",
        )
    )
    model.eval()

    sample_input = (torch.randn(64, 1, 224, 224),)

    # === Configure quantization ===

    pt2e_quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
    )

    pt2e_torch_model = capture_pre_autograd_graph(model, sample_input)
    pt2e_torch_model = prepare_pt2e(pt2e_torch_model, pt2e_quantizer)

    # Run the prepared model with sample input data to ensure that internal observers are populated with correct values
    pt2e_torch_model(*sample_input)

    # Convert the prepared model to a quantized model
    pt2e_torch_model = convert_pt2e(pt2e_torch_model, fold_quantize=False)

    # === Convert to an ai_edge_torch model ===
    pt2e_drq_model = ai_edge_torch.convert(
        pt2e_torch_model,
        sample_input,
        quant_config=QuantConfig(pt2e_quantizer=pt2e_quantizer),
    )

    tflite_path = checkpoint / "weights.tflite"
    pt2e_drq_model.export(tflite_path)

    typer.echo(f"Готово! Модель сохранена как: {tflite_path}")
