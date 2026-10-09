"""
Analyse the client-side calibration results: correlate the browser scores with
the human ratings of speechocean762, compare against the Python (torch fp32)
baseline, and grid-search the acoustic constants for the quantized encoder.

Run with the project venv (after `npm run calibrate`):
    .venv/bin/python web/scripts/analyze_calibration.py
"""
import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "calibration"
CSV_PATH = OUT_DIR / "calibration-results.csv"

GRID_WEIGHTS = [(a / 10, p / 10, round((10 - a - p) / 10, 1)) for a in range(11) for p in range(11 - a)]
GRID_GOOD = [float(g) for g in range(3, 9)]
GRID_BAD = [float(b) for b in range(12, 23)]

DEFAULTS = ((0.3, 0.4, 0.3), 6.0, 15.0)


def score_from_components(ad, per, wer, good, bad, w_ac, w_ph, w_wo):
    acoustic = np.clip(100 * (1 - (ad - good) / (bad - good)), 0, 100)
    phon = np.clip(100 * (1 - per), 0, 100)
    word = np.clip(100 * (1 - wer), 0, 100)
    return np.clip(w_ac * acoustic + w_ph * phon + w_wo * word, 0, 100)


def main() -> None:
    with open(CSV_PATH, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    human = np.array([float(r["human_total"]) for r in rows])
    py_ad = np.array([float(r["python_acoustic"]) for r in rows])
    br_ad = np.array([float(r["browser_acoustic"]) for r in rows])
    per = np.array([float(r["per"]) for r in rows])
    wer = np.array([float(r["wer"]) for r in rows])
    n = len(rows)
    print(f"rows: {n}")

    (w_ac, w_ph, w_wo), good, bad = DEFAULTS
    py_score = score_from_components(py_ad, per, wer, good, bad, w_ac, w_ph, w_wo)
    br_score = score_from_components(br_ad, per, wer, good, bad, w_ac, w_ph, w_wo)

    print("\ncorrelation with human total (Spearman / Pearson), shipped constants:")
    print(f"  python (torch fp32): {spearmanr(py_score, human).correlation:.4f} / {pearsonr(py_score, human).correlation:.4f}")
    print(f"  browser (onnx int8): {spearmanr(br_score, human).correlation:.4f} / {pearsonr(br_score, human).correlation:.4f}")

    print("\nacoustic distance drift (browser vs python):")
    diff = br_ad - py_ad
    print(f"  mean {diff.mean():+.3f} · median {np.median(diff):+.3f} · p10 {np.percentile(diff, 10):+.3f} · p90 {np.percentile(diff, 90):+.3f}")
    print(f"  distance correlation: {pearsonr(py_ad, br_ad).correlation:.4f}")

    # grid search over the browser distances, mirroring benchmarks/speechocean762.py
    results = []
    for weights in GRID_WEIGHTS:
        for g in GRID_GOOD:
            for b in GRID_BAD:
                if weights[0] == 0 and not (g == GRID_GOOD[0] and b == GRID_BAD[0]):
                    continue
                s = score_from_components(br_ad, per, wer, g, b, *weights)
                results.append((spearmanr(s, human).correlation, weights, g, b))
    results.sort(reverse=True)

    print("\ngrid search on browser acoustic distances (top 8 + defaults):")
    print(f"{'spearman':>9} {'w_acoustic':>10} {'w_phonemes':>10} {'w_words':>8} {'GOOD':>6} {'BAD':>5}")
    for rho, weights, g, b in results[:8]:
        print(f"{rho:9.4f} {weights[0]:10.1f} {weights[1]:10.1f} {weights[2]:8.1f} {g:6.1f} {b:5.1f}")
    rho_default = spearmanr(br_score, human).correlation
    print(f"{rho_default:9.4f} {w_ac:10.1f} {w_ph:10.1f} {w_wo:8.1f} {good:6.1f} {bad:5.1f}  <- shipped defaults")

    # stability check: best combo of a random half scored on the other half (3 seeds)
    print("\ntwo-fold cross-validation of the best combo (3 seeds):")
    best_rho, best_weights, best_good, best_bad = results[0]
    stable = []
    for seed in range(3):
        perm = np.random.RandomState(seed).permutation(n)
        half_a, half_b = perm[: n // 2], perm[n // 2:]
        s_a = score_from_components(br_ad[half_a], per[half_a], wer[half_a], best_good, best_bad, *best_weights)
        s_b = score_from_components(br_ad[half_b], per[half_b], wer[half_b], best_good, best_bad, *best_weights)
        rho_a = spearmanr(s_a, human[half_a]).correlation
        rho_b = spearmanr(s_b, human[half_b]).correlation
        stable.append((rho_a + rho_b) / 2)
        print(f"  seed {seed}: held-out spearman {(rho_a + rho_b) / 2:.4f}")
    print(f"  mean held-out: {np.mean(stable):.4f} (full-sample best {best_rho:.4f})")

    print("\nrecommended constants for the browser (int8 encoder):")
    print(f"  SCORE_WEIGHTS = {{ acoustic: {best_weights[0]}, phonemes: {best_weights[1]}, words: {best_weights[2]} }}")
    print(f"  ACOUSTIC_GOOD (en) = {best_good} · ACOUSTIC_DISTANCE_BAD = {best_bad}")


if __name__ == "__main__":
    main()
