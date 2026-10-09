"""
Generate phase-2 parity fixtures:
  - acoustic.json: DTW distances over wav2vec2-large-960h embeddings (torch fp32)
    for (learner fixture, gTTS reference) pairs — the browser/ONNX numbers are
    validated against these with a tolerance for int8 drift
  - score.json: word error rates and scores computed by the Python pipeline on
    transcription pairs (deterministic strings)
  - transcribe.json: word transcriptions of the audio fixtures (large-960h)

Run with the project venv:
    .venv/bin/python web/scripts/generate_fixtures_phase2.py
"""
import json
from pathlib import Path

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from openpronounce import audio, speech

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

TEXTS = {
    "hello": "Hello, how are you?",
    "coffee": "I would like a cup of coffee, please.",
    "weather": "The weather is lovely today, and I am going for a walk in the park.",
}


def acoustic_fixtures() -> dict:
    # gTTS fixture vs piper reading the same sentence: the "good" (different
    # voices, same text) calibration case, mirroring languages.py's method.
    piper_ref = audio.text2speech(TEXTS["coffee"], lang="en", backend="piper")
    learner = audio.load(str(FIXTURES_DIR / "en-coffee.wav"))
    reference = audio.load(piper_ref, sr=16000)
    emb_learner = speech.extract_embeddings(learner)
    emb_reference = speech.extract_embeddings(reference)
    distance, path = fastdtw(emb_learner, emb_reference, dist=euclidean)
    good_case = {
        "learner": "en-coffee.wav",
        "referenceText": TEXTS["coffee"] + " [piper voice]",
        "distance": round(float(distance), 4),
        "pathLength": int(len(path)),
        "meanPerStep": round(float(distance / max(1, len(path))), 4),
        "learnerFrames": int(emb_learner.shape[0]),
        "referenceFrames": int(emb_reference.shape[0]),
        "dim": int(emb_learner.shape[1]),
    }
    print(f"coffee gTTS vs piper (good): {good_case['meanPerStep']:.3f}")

    cases = []
    for name, text in TEXTS.items():
        learner = audio.load(str(FIXTURES_DIR / f"en-{name}.wav"))
        reference_file = audio.text2speech(text, lang="en")
        reference = audio.load(reference_file, sr=16000)

        emb_learner = speech.extract_embeddings(learner)
        emb_reference = speech.extract_embeddings(reference)
        distance, path = fastdtw(emb_learner, emb_reference, dist=euclidean)
        acoustic_distance = distance / max(1, len(path))
        cases.append({
            "learner": f"en-{name}.wav",
            "referenceText": text,
            "distance": round(float(distance), 4),
            "pathLength": int(len(path)),
            "meanPerStep": round(float(acoustic_distance), 4),
            "learnerFrames": int(emb_learner.shape[0]),
            "referenceFrames": int(emb_reference.shape[0]),
            "dim": int(emb_learner.shape[1]),
        })
        print(f"{name}: mean per-step {acoustic_distance:.3f} over {len(path)} steps")

    cases.append(good_case)

    # wrong-sentence case: coffee audio against the weather reference
    learner = audio.load(str(FIXTURES_DIR / "en-coffee.wav"))
    reference = audio.load(audio.text2speech(TEXTS["weather"], lang="en"), sr=16000)
    emb_learner = speech.extract_embeddings(learner)
    emb_reference = speech.extract_embeddings(reference)
    distance, path = fastdtw(emb_learner, emb_reference, dist=euclidean)
    cases.append({
        "learner": "en-coffee.wav",
        "referenceText": TEXTS["weather"],
        "distance": round(float(distance), 4),
        "pathLength": int(len(path)),
        "meanPerStep": round(float(distance / max(1, len(path))), 4),
        "learnerFrames": int(emb_learner.shape[0]),
        "referenceFrames": int(emb_reference.shape[0]),
        "dim": int(emb_reference.shape[1]),
    })
    print(f"coffee-vs-weather (wrong sentence): {distance / max(1, len(path)):.3f}")

    return {"lang": "en", "cases": cases}


def score_fixtures() -> dict:
    transcription_cases = [
        ("I would like a cup of coffee, please.", "I would like a cup of coffee, please."),
        ("I would like a cup of coffee, please.", "i would like a cup of coffee please"),
        ("I would like a cup of coffee, please.", "I would like a cup of tea and a piece of cake."),
        ("Hello, how are you?", "hello how are you"),
        ("Hello, how are you?", "hello how old are you"),
        ("The weather is lovely today.", "the weather in lovely today"),
        ("The weather is lovely today.", ""),
    ]
    cases = []
    for reference, transcription in transcription_cases:
        expected_words = [w.lower() for w in speech._words(reference)]
        heard_words = [w.lower() for w in speech._words(speech.clean_transcription(transcription))]
        wer = speech.compare_transcriptions(transcription, reference, "en")["word_error_rate"]
        for acoustic in (5.0, 7.5, 10.0, 13.0, 16.5):
            cases.append({
                "reference": reference,
                "transcription": transcription,
                "expectedWords": expected_words,
                "heardWords": heard_words,
                "wordErrorRate": wer,
                "acousticDistance": acoustic,
                "score": speech.compute_pronunciation_score(acoustic, 0.15, wer, "en"),
            })
    return {"lang": "en", "cases": cases}


def transcribe_fixtures() -> dict:
    cases = []
    for name in TEXTS:
        waveform = audio.load(str(FIXTURES_DIR / f"en-{name}.wav"))
        transcription = speech.transcribe(waveform, "en")
        cases.append({"learner": f"en-{name}.wav", "text": TEXTS[name], "transcription": transcription})
        print(f"{name}: {transcription!r}")
    return {"lang": "en", "cases": cases}


def main() -> None:
    (FIXTURES_DIR / "acoustic.json").write_text(
        json.dumps(acoustic_fixtures(), ensure_ascii=False, indent=2), encoding="utf-8")
    print("acoustic.json written")
    (FIXTURES_DIR / "score.json").write_text(
        json.dumps(score_fixtures(), ensure_ascii=False, indent=2), encoding="utf-8")
    print("score.json written")
    (FIXTURES_DIR / "transcribe.json").write_text(
        json.dumps(transcribe_fixtures(), ensure_ascii=False, indent=2), encoding="utf-8")
    print("transcribe.json written")


if __name__ == "__main__":
    main()
