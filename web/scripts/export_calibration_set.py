"""
Export the speechocean762 benchmark sample for the CLIENT-SIDE calibration:
audio wavs (learner + gTTS reference) and a JSON manifest with the Python
pipeline's stored components, so the Node/browser harness can recompute the
acoustic term with the quantized ONNX encoder and recalibrate the constants.

Run with the project venv (network on first run: ~300 MB dataset download):
    .venv/bin/python web/scripts/export_calibration_set.py
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "benchmarks"))
import speechocean762 as bench  # noqa: E402

from openpronounce import audio  # noqa: E402

CSV_PATH = REPO / "benchmarks" / "results" / "speechocean762-v0.3.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "calibration"


def main() -> None:
    wav_dir = OUT_DIR / "wav"
    ref_dir = OUT_DIR / "ref"
    for directory in (wav_dir, ref_dir):
        directory.mkdir(parents=True, exist_ok=True)

    with open(CSV_PATH, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    print(f"CSV rows: {len(rows)}")

    by_utt = {row["utt"]: row for row in load_parquet_rows(rows)}

    manifest = []
    ref_index = {}
    for index, row in enumerate(rows):
        utt = row["utt"]
        entry = by_utt.get(utt)
        if entry is None:
            print(f"skip {utt}: not found in parquet")
            continue

        learner_path = wav_dir / f"{utt}.wav"
        if not learner_path.exists():
            waveform = resample_to_16k(entry["audio"]["bytes"])
            sf.write(learner_path, waveform, 16000)

        # gTTS reference, one per distinct sentence (cached by audio.text2speech)
        text = row["text"]
        if text not in ref_index:
            ref_path = ref_dir / f"ref-{len(ref_index)}.wav"
            sf.write(ref_path, audio.load(audio.text2speech(text, lang="en")), 16000)
            ref_index[text] = ref_path.name

        manifest.append({
            "utt": utt,
            "text": text,
            "humanTotal": float(row["human_total"]),
            "pythonAcoustic": float(row["acoustic_distance"]),
            "per": float(row["phoneme_error_rate"]),
            "wer": float(row["word_error_rate"]),
            "learnerFile": learner_path.name,
            "referenceFile": ref_index[text],
        })
        if (index + 1) % 50 == 0:
            print(f"{index + 1}/{len(rows)} exported")

    out_manifest = OUT_DIR / "manifest.json"
    out_manifest.write_text(json.dumps({
        "source": "benchmarks/results/speechocean762-v0.3.csv (seed 0, stratified 500)",
        "pythonConstants": {"weights": [0.3, 0.4, 0.3], "acousticGood": 6.0, "acousticBad": 15.0},
        "entries": manifest,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"manifest: {out_manifest} ({len(manifest)} entries)")


def load_parquet_rows(rows):
    """Load only the parquet rows matching the CSV utterances (keeps memory sane)."""
    wanted = {row["utt"] for row in rows}
    for entry in bench.load_rows("test"):
        if entry["utt"] in wanted:
            yield entry


def resample_to_16k(raw: bytes) -> np.ndarray:
    import io

    import librosa

    waveform, _ = librosa.load(io.BytesIO(raw), sr=16000, mono=True)
    return waveform


if __name__ == "__main__":
    main()
