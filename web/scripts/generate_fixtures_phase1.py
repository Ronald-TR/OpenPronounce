"""
Generate phase-1 parity fixtures: G2P, Levenshtein opcodes, FastDTW and the
full compare_phones output of the Python pipeline, so the TypeScript port can
be validated component by component.

Run with the project venv (after generate_fixtures.py, which needs network for gTTS):
    .venv/bin/python web/scripts/generate_fixtures_phase1.py
"""
import base64
import json
from pathlib import Path

import numpy as np
from fastdtw import fastdtw
from Levenshtein import distance as lev_distance, opcodes as lev_opcodes
from scipy.spatial.distance import euclidean

from openpronounce import audio, phones

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

G2P_TEXTS = [
    "Hello, how are you?",
    "I would like a cup of coffee, please.",
    "The weather is lovely today, and I am going for a walk in the park.",
    "She thought three thieves threw the theory out the theatre.",
    "Would you like a cup of tea and a piece of cake?",
    "The boy enjoys his toy and the coins he found around the house.",
]

# (audio fixture, expected text) pairs; the last one deliberately mismatches the
# audio so deletes/replaces across words are exercised.
COMPARE_CASES = [
    ("en-hello.wav", "Hello, how are you?"),
    ("en-coffee.wav", "I would like a cup of coffee, please."),
    ("en-weather.wav", "The weather is lovely today, and I am going for a walk in the park."),
    ("en-coffee.wav", "I would like a cup of tea and a piece of cake."),
]


def g2p_fixtures() -> dict:
    cases = []
    for text in G2P_TEXTS:
        words, groups = phones.get_expected_phones(text, "en")
        cases.append({"text": text, "words": words, "groups": groups})
    return {"lang": "en", "cases": cases}


def seq_fixtures() -> dict:
    """Levenshtein + FastDTW reference outputs on the real phone sequences."""
    cases = []
    for wav_name, text in COMPARE_CASES:
        waveform = audio.load(str(FIXTURES_DIR / wav_name))
        recognition = phones.recognize_phones(waveform, 16000, lang="en")
        words, groups = phones.get_expected_phones(text, "en")
        expected = [p for g in groups for p in g]
        heard = list(recognition.phones)
        if not expected or not heard:
            continue

        # speech.py joins phones with spaces and maps each CHARACTER (incl. spaces)
        # to its codepoint — mirror that exactly.
        distance, path = fastdtw(
            np.array([ord(c) for c in " ".join(expected)], dtype=float).reshape(-1, 1),
            np.array([ord(c) for c in " ".join(heard)], dtype=float).reshape(-1, 1),
            dist=euclidean,
        )
        cases.append({
            "wav": wav_name,
            "text": text,
            "expected": expected,
            "heard": heard,
            "levenshtein_distance": lev_distance(expected, heard),
            "opcodes": [[tag, i1, i2, j1, j2] for tag, i1, i2, j1, j2 in lev_opcodes(expected, heard)],
            "fastdtw_distance": round(float(distance), 6),
            "fastdtw_path_length": len(path),
        })
    # one tiny exact pair for the DTW base case
    small_distance, small_path = fastdtw(
        np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]),
        np.array([[2.0], [3.0], [4.0]]),
        dist=euclidean,
    )
    return {
        "cases": cases,
        "small_dtw": {"distance": round(float(small_distance), 9), "path": [list(p) for p in small_path]},
    }


def compare_fixtures() -> dict:
    # Save the recognizer vocab once: plausibility lookups in the TS port need
    # the same token->id mapping the Python pipeline used.
    vocab = phones.phone_vocab()
    (FIXTURES_DIR / "vocab.json").write_text(
        json.dumps(list(vocab), ensure_ascii=False), encoding="utf-8"
    )

    cases = []
    for wav_name, text in COMPARE_CASES:
        waveform = audio.load(str(FIXTURES_DIR / wav_name))
        recognition = phones.recognize_phones(waveform, 16000, lang="en")
        result = phones.compare_phones(recognition, text, "en")

        log_posteriors = np.asarray(recognition.log_posteriors, dtype=np.float32)
        frames, vocab_size = log_posteriors.shape
        cases.append({
            "wav": wav_name,
            "text": text,
            "recognition": {
                "phones": recognition.phones,
                "confidences": [round(c, 6) for c in recognition.confidences],
                "spans": [list(s) for s in recognition.spans],
                "frames": int(frames),
                "vocabSize": int(vocab_size),
                "logPosteriorsB64": base64.b64encode(log_posteriors.tobytes()).decode("ascii"),
            },
            "compare": result,
        })
        flagged = [e["word"] for e in result["errors"]]
        print(f"{wav_name} vs {text!r}: {len(result['errors'])} errors {flagged}")
    return {"lang": "en", "cases": cases}


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    g2p = g2p_fixtures()
    (FIXTURES_DIR / "g2p.json").write_text(json.dumps(g2p, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"g2p: {len(g2p['cases'])} cases")

    seq = seq_fixtures()
    (FIXTURES_DIR / "sequences.json").write_text(json.dumps(seq, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"sequences: {len(seq['cases'])} cases")

    compare = compare_fixtures()
    (FIXTURES_DIR / "compare.json").write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"compare: {len(compare['cases'])} cases")


if __name__ == "__main__":
    main()
