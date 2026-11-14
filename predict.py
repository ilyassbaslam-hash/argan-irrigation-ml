#!/usr/bin/env python3
"""
Score new plot-days with the trained argan irrigation models.

The input must contain the 14 columns listed in schema.FEATURES (extra columns
such as the labels are ignored). One row = one plot on one morning.

Examples
--------
    python predict.py --demo
    python predict.py --csv data/argan_dataset_train.csv --out predictions.csv
    python predict.py --json '{"day_of_year": 200, "month": 7, ...}'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from schema import CATEGORICAL, FEATURES

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = HERE / "models"

DEMO_ROWS = [
    {  # hot July morning, dry-ish sandy loam, no rain in sight
        "day_of_year": 195, "month": 7, "temp_max": 38.4, "temp_min": 21.0,
        "rain_mm": 0.0, "et0_mm": 6.4, "rain_forecast_3d_mm": 0.0,
        "argan_stage": "fruit_ripening", "tree_age_years": 18,
        "trees_per_ha": 204, "canopy_cover_pct": 52.0, "soil_type": "sandy_loam",
        "soil_depth_cm": 80, "soil_moisture_pct": 12.5,
    },
    {  # mild January morning, already wet, rain coming
        "day_of_year": 12, "month": 1, "temp_max": 18.1, "temp_min": 6.2,
        "rain_mm": 4.5, "et0_mm": 1.6, "rain_forecast_3d_mm": 14.0,
        "argan_stage": "winter_rest", "tree_age_years": 18,
        "trees_per_ha": 204, "canopy_cover_pct": 52.0, "soil_type": "sandy_loam",
        "soil_depth_cm": 80, "soil_moisture_pct": 21.0,
    },
    {  # young plot on shallow clay soil, dry
        "day_of_year": 140, "month": 5, "temp_max": 31.0, "temp_min": 15.5,
        "rain_mm": 0.0, "et0_mm": 5.1, "rain_forecast_3d_mm": 1.2,
        "argan_stage": "flowering", "tree_age_years": 4,
        "trees_per_ha": 400, "canopy_cover_pct": 15.3, "soil_type": "clay",
        "soil_depth_cm": 40, "soil_moisture_pct": 24.0,
    },
]


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Score plot-days with the argan models.")
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--csv", type=Path, help="CSV with the feature columns")
    source.add_argument("--json", type=str,
                        help="JSON object or list of objects with the feature columns")
    source.add_argument("--demo", action="store_true", help="score three built-in example rows")
    ap.add_argument("--out", type=Path, help="write the predictions to this CSV")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="probability above which the classifier says 'irrigate' (default 0.5)")
    ap.add_argument("--mode", choices=["two-stage", "direct"], default="two-stage",
                    help="two-stage: classifier gate + litres model trained on irrigation days; "
                         "direct: litres model trained on all rows (default: two-stage)")
    return ap.parse_args(argv)


def load_input(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv is not None:
        if not args.csv.exists():
            sys.exit(f"[input] file not found: {args.csv}")
        return pd.read_csv(args.csv)
    if args.json is not None:
        try:
            payload = json.loads(args.json)
        except json.JSONDecodeError as exc:
            sys.exit(f"[input] invalid JSON: {exc}")
        return pd.DataFrame(payload if isinstance(payload, list) else [payload])
    return pd.DataFrame(DEMO_ROWS)


def load_models(model_dir: Path):
    files = {
        "classifier": model_dir / "irrigation_classifier.joblib",
        "regressor": model_dir / "irrigation_regressor.joblib",
        "regressor_positive_only": model_dir / "irrigation_regressor_positive_only.joblib",
    }
    bundles = {}
    for key, path in files.items():
        if not path.exists():
            sys.exit(f"[models] {path} not found — run train_model.py first")
        bundles[key] = joblib.load(path)
    return bundles


def check_input(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        sys.exit(f"[input] missing feature columns: {missing}")
    if df[FEATURES].isna().any().any():
        bad = df[FEATURES].isna().sum()
        sys.exit(f"[input] missing values:\n{bad[bad > 0]}")
    for col in CATEGORICAL:
        if not pd.api.types.is_string_dtype(df[col]) and df[col].dtype != object:
            sys.exit(f"[input] column '{col}' must contain strings, got {df[col].dtype}")
    return df


def main(argv=None) -> None:
    args = parse_args(argv)
    if not 0.0 < args.threshold < 1.0:
        sys.exit("[input] --threshold must be between 0 and 1")

    bundles = load_models(args.model_dir)
    clf_bundle = bundles["classifier"]
    reg_bundle = bundles["regressor"]
    reg_pos_bundle = bundles["regressor_positive_only"]
    df = check_input(load_input(args))
    X = df[clf_bundle.get("features", FEATURES)]

    proba = clf_bundle["pipeline"].predict_proba(X)[:, 1]
    irrigate = (proba >= args.threshold).astype(int)
    size_bundle = reg_pos_bundle if args.mode == "two-stage" else reg_bundle
    litres = size_bundle["pipeline"].predict(X)

    out = df.copy()
    out["pred_irrigate_probability"] = proba.round(3)
    out["pred_irrigate_yes_no"] = irrigate
    out["pred_irrigation_L_per_tree"] = litres.round(1)
    out["pred_recommended_L_per_tree"] = (litres * irrigate).round(1)

    cols = ["pred_irrigate_yes_no", "pred_irrigate_probability",
            "pred_irrigation_L_per_tree", "pred_recommended_L_per_tree"]
    display = out[cols]
    if args.out is not None:
        out.to_csv(args.out, index=False)
        print(f"[out] wrote {len(out)} prediction(s) to {args.out}")
    print(f"[mode] {args.mode} — "
          f"{'classifier gate + irrigation-day regressor' if args.mode == 'two-stage' else 'regressor trained on all rows'}")
    print(display.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:   # e.g. piping the table into `head`
        sys.exit(0)
