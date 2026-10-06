"""
Export the wav2vec2-large-960h ENCODER (the .wav2vec2 module the Python
pipeline uses for acoustic embeddings) to ONNX for browser inference.

Outputs (to web/public/models/wav2vec2-960h-encoder/):
  config.json               copied from the HF snapshot
  onnx/model.onnx           fp32 export (~1.4 GB, not for git)
  onnx/model_quantized.onnx int8 dynamic quantization (~360 MB, browser default)

Run with the project venv:
    .venv/bin/python web/scripts/export_embedding_model.py

Note: needs `pip install onnx onnxruntime` in the venv for quantization.
"""
import json
import shutil
from pathlib import Path

import torch
from transformers import Wav2Vec2Model

OUT_DIR = Path(__file__).resolve().parent.parent / "public" / "models" / "wav2vec2-960h-encoder"
MODEL_ID = "facebook/wav2vec2-large-960h"


def main() -> None:
    out_onnx = OUT_DIR / "onnx" / "model.onnx"
    out_quantized = OUT_DIR / "onnx" / "model_quantized.onnx"
    if out_quantized.exists():
        print(f"already exported: {out_quantized}")
        return

    print(f"loading {MODEL_ID} encoder...")
    model = Wav2Vec2Model.from_pretrained(MODEL_ID)
    model.eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "onnx").mkdir(parents=True, exist_ok=True)

    config_src = Path(
        shutil.which("python") or ""
    )  # config fetched via transformers cache below instead
    from huggingface_hub import hf_hub_download

    config_file = hf_hub_download(MODEL_ID, "config.json")
    shutil.copy(config_file, OUT_DIR / "config.json")

    if not out_onnx.exists():
        print("exporting to ONNX (fp32)... this writes ~1.4 GB")
        dummy = torch.randn(1, 32000)
        torch.onnx.export(
            model,
            (dummy,),
            str(out_onnx),
            input_names=["input_values"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_values": {0: "batch", 1: "sequence"},
                "last_hidden_state": {0: "batch", 1: "sequence"},
            },
            opset_version=17,
            dynamo=False,
        )

    print("quantizing (int8 dynamic, MatMul/Gemm only — ConvInteger is unsupported in ORT wasm/node)...")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(out_onnx),
        str(out_quantized),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    size_mb = out_quantized.stat().st_size / 1e6
    print(f"done: {out_quantized} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
