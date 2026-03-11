#!/usr/bin/env python3
"""
Kaari Calibration Extractor (C2-A1)
====================================
Computes per-model calibration values from research data.
Outputs 7 standardized values per model.

Usage:
    python -m kaari.calibrate --data-dir ../intent-vector-test/results/
    python -m kaari.calibrate --data-dir ../intent-vector-test/results/ --output calibration.json

Output format per model:
    clean_dv2_mean, clean_dv2_std, clean_length_mean,
    threshold_dv2, threshold_c2, auc_dv2, auc_c2
"""

import json
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def load_data(data_dir: Path) -> pd.DataFrame:
    """Load all injection_matrix CSVs."""
    dfs = []
    for f in sorted(data_dir.glob("injection_matrix_*.csv")):
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"  Skip {f.name}: {e}")

    if not dfs:
        raise FileNotFoundError(f"No injection_matrix CSV files in {data_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    valid = combined[combined["delta_v2"].notna() & combined["error"].isna()].copy()
    valid["label"] = (valid["condition"] == "dirty").astype(int)
    valid["resp_len"] = valid["response_char_length"].fillna(
        valid["response_text"].astype(str).str.len()
    )
    return valid


def youden_threshold(labels, scores):
    """Youden-optimal threshold."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def compute_c2(dv2_array, len_array, clean_mean_len):
    """Compute C2 for arrays."""
    c2 = np.zeros_like(dv2_array)
    for i in range(len(dv2_array)):
        d, l = dv2_array[i], len_array[i]
        if clean_mean_len > 0 and l > 0:
            c2[i] = d * (1.0 + 0.5 * math.log(l / clean_mean_len))
        else:
            c2[i] = d
    return c2


def calibrate_model(df: pd.DataFrame, model_name: str) -> dict:
    """Compute 7 calibration values for one model."""
    labels = df["label"].values
    dv2 = df["delta_v2"].values
    resp_len = df["resp_len"].values
    clean_mask = labels == 0

    # Clean baseline stats
    clean_dv2 = dv2[clean_mask]
    clean_len = resp_len[clean_mask]

    clean_dv2_mean = float(clean_dv2.mean())
    clean_dv2_std = float(clean_dv2.std())
    clean_length_mean = float(clean_len.mean())

    # C2 computation
    c2 = compute_c2(dv2, resp_len, clean_length_mean)

    # AUC scores
    auc_dv2 = float(roc_auc_score(labels, dv2))
    auc_c2 = float(roc_auc_score(labels, c2))

    # Youden-optimal thresholds
    threshold_dv2 = youden_threshold(labels, dv2)
    threshold_c2 = youden_threshold(labels, c2)

    return {
        "clean_dv2_mean": round(clean_dv2_mean, 6),
        "clean_dv2_std": round(clean_dv2_std, 6),
        "clean_length_mean": round(clean_length_mean, 1),
        "threshold_dv2": round(threshold_dv2, 6),
        "threshold_c2": round(threshold_c2, 6),
        "auc_dv2": round(auc_dv2, 4),
        "auc_c2": round(auc_c2, 4),
        "n_total": len(df),
        "n_clean": int(clean_mask.sum()),
        "n_dirty": int((~clean_mask).sum()),
    }


def detect_models(df: pd.DataFrame) -> dict:
    """
    Detect which model produced each run based on source file timestamps.
    Maps CSV timestamps to known model runs.
    """
    # Model run schedule (from production logs)
    model_map = {
        # Jan 2026 runs
        "20260113_210100": "mistral-7b",
        "20260113_233936": "mistral-7b",
        "20260114_090152": "mistral-7b",
        "20260114_112657": "gpt-oss-20b",
        "20260114_170042": "gemma2-9b",
        "20260115_121837": "qwen3-8b",
        "20260115_145459": "mistral-7b",
        # Feb 2026 runs (code + persona families)
        "20260210_224200": "mistral-7b",
        "20260210_234142": "mistral-7b",
        "20260211_015236": "mistral-7b",
        "20260211_040155": "mistral-7b",
        "20260211_061145": "mistral-7b",
        "20260211_082216": "gpt-oss-20b",
        "20260211_103315": "gemma2-9b",
        "20260211_130534": "qwen3-8b",
        "20260211_151330": "mistral-7b",
    }
    return model_map


def run_calibration(data_dir: Path, output_path: Path = None):
    """Run full calibration extraction."""
    print("=" * 60)
    print("  Kaari Calibration Extractor (C2-A1)")
    print("=" * 60)

    # Load data
    df = load_data(data_dir)
    print(f"\n  Loaded {len(df)} valid rows")

    # Detect models from file timestamps
    model_map = detect_models(df)

    # Add model column by matching source file timestamps
    if "_source" not in df.columns:
        df["_source"] = "unknown"

    df["_model"] = "unknown"
    for ts, model in model_map.items():
        mask = df["_source"].astype(str).str.contains(ts) if "_source" in df.columns else pd.Series(False, index=df.index)
        df.loc[mask, "_model"] = model

    # Try matching from timestamp column if _source didn't work
    if (df["_model"] == "unknown").all():
        # Reload with source tracking
        dfs = []
        for f in sorted(data_dir.glob("injection_matrix_*.csv")):
            try:
                d = pd.read_csv(f)
                d["_source"] = f.stem
                dfs.append(d)
            except:
                pass
        if dfs:
            df_new = pd.concat(dfs, ignore_index=True)
            df_new = df_new[df_new["delta_v2"].notna() & df_new["error"].isna()].copy()
            df_new["label"] = (df_new["condition"] == "dirty").astype(int)
            df_new["resp_len"] = df_new["response_char_length"].fillna(
                df_new["response_text"].astype(str).str.len()
            )
            df_new["_model"] = "unknown"
            for ts, model in model_map.items():
                mask = df_new["_source"].str.contains(ts)
                df_new.loc[mask, "_model"] = model
            df = df_new

    # Global calibration
    print(f"\n  --- GLOBAL (all models) ---")
    global_cal = calibrate_model(df, "global")
    print(f"    N={global_cal['n_total']} (clean={global_cal['n_clean']}, dirty={global_cal['n_dirty']})")
    print(f"    clean_dv2: {global_cal['clean_dv2_mean']:.4f} ± {global_cal['clean_dv2_std']:.4f}")
    print(f"    clean_length: {global_cal['clean_length_mean']:.0f}")
    print(f"    threshold_dv2: {global_cal['threshold_dv2']:.4f}")
    print(f"    threshold_c2: {global_cal['threshold_c2']:.4f}")
    print(f"    AUC dv2={global_cal['auc_dv2']:.4f}  C2={global_cal['auc_c2']:.4f}")

    # Per-model calibration
    calibration = {"_global": global_cal}
    models = [m for m in df["_model"].unique() if m != "unknown"]

    for model in sorted(models):
        model_df = df[df["_model"] == model]
        if len(model_df) < 20:
            print(f"\n  --- {model} --- SKIPPED (only {len(model_df)} rows)")
            continue

        cal = calibrate_model(model_df, model)
        calibration[model] = cal
        print(f"\n  --- {model} ---")
        print(f"    N={cal['n_total']} (clean={cal['n_clean']}, dirty={cal['n_dirty']})")
        print(f"    clean_dv2: {cal['clean_dv2_mean']:.4f} ± {cal['clean_dv2_std']:.4f}")
        print(f"    clean_length: {cal['clean_length_mean']:.0f}")
        print(f"    threshold_dv2: {cal['threshold_dv2']:.4f}")
        print(f"    threshold_c2: {cal['threshold_c2']:.4f}")
        print(f"    AUC dv2={cal['auc_dv2']:.4f}  C2={cal['auc_c2']:.4f}")

    # Handle unknown rows
    unknown_count = (df["_model"] == "unknown").sum()
    if unknown_count > 0:
        print(f"\n  WARNING: {unknown_count} rows could not be mapped to a model.")

    # Output
    output = output_path or (data_dir.parent / "kaari_calibration_v1.json")
    with open(output, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"\n  Calibration saved to {output}")
    print(f"  Models calibrated: {len(calibration) - 1} + global")
    print("=" * 60)

    return calibration


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaari Calibration Extractor")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "intent-vector-test" / "results",
        help="Directory containing injection_matrix CSV files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: data-dir/../kaari_calibration_v1.json)",
    )
    args = parser.parse_args()
    run_calibration(args.data_dir, args.output)
