"""
Generate spike fixtures: TTS reference samples + the expected phone output of
the Python pipeline, used by the browser/Node parity tests.

Run with the project venv:
    .venv/bin/python web/scripts/generate_fixtures.py
"""
import json
from pathlib import Path

import soundfile as sf

from openpronounce import audio, phones

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

SENTENCES = [
    ("hello", "Hello, how are you?"),
    ("coffee", "I would like a cup of coffee, please."),
    ("weather", "The weather is lovely today, and I am going for a walk in the park."),
]


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    fixtures = []
    for name, text in SENTENCES:
        wav_path = audio.text2speech(text, lang="en")
        waveform = audio.load(wav_path)

        out_path = FIXTURES_DIR / f"en-{name}.wav"
        sf.write(out_path, waveform, 16000)

        recognition = phones.recognize_phones(waveform, 16000, lang="en")
        fixtures.append({
            "name": name,
            "text": text,
            "file": out_path.name,
            "phones": recognition.phones,
            "confidences": [round(c, 4) for c in recognition.confidences],
        })
        print(f"{name}: {len(recognition.phones)} phones -> {out_path.name}")

    payload = {
        "model": phones.PHONE_MODEL_NAME,
        "lang": "en",
        "fixtures": fixtures,
    }
    (FIXTURES_DIR / "expected.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {FIXTURES_DIR / 'expected.json'}")


if __name__ == "__main__":
    main()
