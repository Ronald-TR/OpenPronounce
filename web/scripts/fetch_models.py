"""
Fetch every model used by the client-side app into web/public/models/ using
the layout the front-end expects (also the layout to upload to S3):

    models/phone-recognizer/     (wav2vec2-lv-60-espeak-cv-ft, ONNX int8+fp16)
    models/word-transcription/   (wav2vec2-base-960h, ONNX int8+fp16)
    models/wav2vec2-960h-encoder/  (already produced by export_embedding_model.py)

Run with the project venv (resumable: existing files are skipped):
    .venv/bin/python web/scripts/fetch_models.py
"""
from pathlib import Path

from huggingface_hub import hf_hub_download

WEB = Path(__file__).resolve().parent.parent
MODELS_DIR = WEB / "public" / "models"

SMALL_FILES = [
    "config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
]

REPOS = {
    "phone-recognizer": ("onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX", [
        *SMALL_FILES,
        "onnx/model_quantized.onnx",
        "onnx/model_fp16.onnx",
    ]),
    "word-transcription": ("onnx-community/wav2vec2-base-960h-ONNX", [
        *SMALL_FILES,
        "tokenizer.json",
        "onnx/model_quantized.onnx",
        "onnx/model_fp16.onnx",
    ]),
}


def main() -> None:
    for model_id, (repo, files) in REPOS.items():
        target = MODELS_DIR / model_id
        target.mkdir(parents=True, exist_ok=True)
        for filename in files:
            destination = target / filename
            if destination.exists() and destination.stat().st_size > 0:
                print(f"skip {model_id}/{filename} (exists)")
                continue
            print(f"downloading {repo}/{filename}...")
            downloaded = hf_hub_download(repo, filename, local_dir=target)
            print(f"  -> {downloaded}")

    total = sum(f.stat().st_size for f in MODELS_DIR.rglob("*") if f.is_file())
    print(f"\nmodels dir: {MODELS_DIR} ({total / 1e6:.0f} MB total)")


if __name__ == "__main__":
    main()
