#!/usr/bin/env python3
"""
Weekly planner: Regressor B -> weekly need -> scarcity allocator.

This is the glue between the machine-learning layer and the product layer:

  1. read 7 days of forecast features per plot (one row per plot-day),
  2. run the classifier and Regressor B on each day,
  3. turn that into one weekly net need per tree,
  4. apply the rain gate ("if rain is forecast, wait"),
  5. allocate the farmer's source-water budget with allocator.py,
  6. write the per-plot plan (litres, days, pump minutes, stress, loss score).

Regressor A is intentionally NOT used: the app ships the classifier plus the
regressor trained on irrigation days (Regressor B).

Input CSV (`--csv`): one row per plot-day, 7 rows per plot, containing:
  * the 14 model features (see schema.py),
  * plot metadata repeated on each row: trees, irrigation_system,
    depletion_mm, raw_mm, taw_mm, and optionally pump_flow_m3_per_h, priority,
  * rain_mm is the forecast rain for that day.

Run:
    python plan_week.py --demo --budget 150
    python plan_week.py --csv week_forecast.csv --budget 150 --out plan.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from allocator import (
    STAGES,
    SYSTEMS,
    AllocationConfig,
    Plot,
    allocate,
    print_result,
)
from schema import FEATURES, SOILS

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = HERE / "models"

WEEK = 7
REQUIRED_META = [
    "plot_id", "trees", "irrigation_system", "depletion_mm", "raw_mm", "taw_mm",
]


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a weekly irrigation plan under a water budget.")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="six built-in plots, 7 forecast days")
    source.add_argument("--csv", type=Path, help="one row per plot-day (7 days per plot)")
    ap.add_argument("--budget", type=float, required=True,
                    help="source water available for the week, m3")
    ap.add_argument("--out", type=Path, help="write the combined plan to this CSV")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="classifier probability above which a day counts as irrigation")
    ap.add_argument("--rain-skip-mm", type=float, default=10.0,
                    help="weekly rain forecast above which a non-stressed plot waits")
    ap.add_argument("--floor-fraction", type=float, default=0.35)
    ap.add_argument("--gamma", type=float, default=2.0)
    return ap.parse_args(argv)


# --------------------------------------------------------------------------
# Model layer
# --------------------------------------------------------------------------
def load_models(model_dir: Path):
    paths = {
        "classifier": model_dir / "irrigation_classifier.joblib",
        "regressor": model_dir / "irrigation_regressor_positive_only.joblib",
    }
    bundles = {}
    for key, path in paths.items():
        if not path.exists():
            sys.exit(f"[models] {path} not found — run train_model.py first")
        bundles[key] = joblib.load(path)
    return bundles


def check_week_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_META + FEATURES if c not in df.columns]
    if missing:
        sys.exit(f"[input] missing columns: {missing}")
    if df[FEATURES + REQUIRED_META].isna().any().any():
        bad = df[FEATURES + REQUIRED_META].isna().sum()
        sys.exit(f"[input] missing values:\n{bad[bad > 0]}")

    # Canonicalise the categorical values the model saw during training.
    df = df.copy()
    for col in ("argan_stage", "soil_type"):
        df[col] = (df[col].astype(str).str.strip().str.lower()
                   .str.replace("-", "_", regex=False)
                   .str.replace(" ", "_", regex=False))
    unknown_stage = sorted(set(df["argan_stage"]) - set(STAGES))
    if unknown_stage:
        sys.exit(f"[input] unknown argan_stage values: {unknown_stage}")
    unknown_soil = sorted(set(df["soil_type"]) - set(SOILS))
    if unknown_soil:
        sys.exit(f"[input] unknown soil_type values: {unknown_soil}")

    counts = df.groupby("plot_id").size()
    if (counts != WEEK).any():
        print(f"[input] warning: these plots do not have exactly {WEEK} rows: "
              f"{counts[counts != WEEK].to_dict()}")
    return df


def weekly_plan(df: pd.DataFrame, bundles: dict, threshold: float,
                rain_skip_mm: float) -> tuple[list[Plot], pd.DataFrame]:
    """Aggregate 7 daily model calls into one weekly need per plot."""
    clf = bundles["classifier"]["pipeline"]
    reg = bundles["regressor"]["pipeline"]
    features = bundles["classifier"].get("features", FEATURES)

    plots: list[Plot] = []
    diagnostics = []
    for plot_id, g in df.groupby("plot_id", sort=False):
        if "date" in g.columns:
            g = g.sort_values("date")
        X = g[features]
        proba = clf.predict_proba(X)[:, 1]
        litres = reg.predict(X)
        daily_yes = proba >= threshold

        # The model predicts an EVENT SIZE ("litres to give today"), not daily
        # consumption.  Summing 7 daily events would assume irrigating every
        # day and overstate the weekly need ~7x.  The weekly target is the
        # first recommended event in the window; the app re-plans after it.
        candidates = np.where(daily_yes & (litres > 0))[0]
        need_week = float(litres[candidates[0]]) if len(candidates) else 0.0
        first_event_day = int(candidates[0]) + 1 if len(candidates) else 1
        irrigate = bool(len(candidates) and need_week > 0)
        rain_week = float(g["rain_mm"].sum())

        first = g.iloc[0]
        stage = (str(first["argan_stage"]).strip().lower()
                 .replace("-", "_").replace(" ", "_"))
        system = (str(first["irrigation_system"]).strip().lower()
                  .replace("-", "_").replace(" ", "_"))
        if stage not in STAGES:
            sys.exit(f"[input] {plot_id}: unknown argan_stage '{first['argan_stage']}'")
        if system not in SYSTEMS:
            sys.exit(f"[input] {plot_id}: unknown irrigation_system "
                     f"'{first['irrigation_system']}'")
        depletion = float(first["depletion_mm"])
        raw = float(first["raw_mm"])
        reason = ""
        if rain_week >= rain_skip_mm and depletion <= raw:
            irrigate = False
            need_week = 0.0
            reason = f"wait for rain ({rain_week:.0f} mm)"
        elif not len(candidates):
            irrigate = False
            need_week = 0.0
            reason = "classifier says no"
        elif need_week <= 0:
            irrigate = False
            reason = "no litres predicted"
        elif rain_week >= rain_skip_mm and depletion > raw:
            reason = f"first event day {candidates[0] + 1}, irrigate despite rain (already dry)"
        else:
            reason = f"first event day {candidates[0] + 1} of 7"

        plots.append(Plot(
            plot_id=str(plot_id),
            trees=int(first["trees"]),
            stage=stage,
            tree_age_years=float(first["tree_age_years"]),
            irrigation_system=system,
            need_L_per_tree_week=round(need_week, 1),
            depletion_mm=depletion,
            raw_mm=raw,
            taw_mm=float(first["taw_mm"]),
            pump_flow_m3_per_h=(float(first["pump_flow_m3_per_h"])
                                if "pump_flow_m3_per_h" in g.columns
                                and pd.notna(first["pump_flow_m3_per_h"]) else None),
            irrigate=irrigate,
            rain_forecast_mm=rain_week,
            priority=(float(first["priority"]) if "priority" in g.columns
                      and pd.notna(first["priority"]) else 1.0),
            first_day=first_event_day,
        ))
        diagnostics.append({
            "plot": str(plot_id),
            "stage": stage,
            "max_prob": round(float(proba.max()), 3),
            "first_event_day": first_event_day if len(candidates) else None,
            "days_over_threshold": int(daily_yes.sum()),
            "rain_week_mm": round(rain_week, 1),
            "need_L/tree": round(need_week, 1),
            "reason": reason,
        })
    diag = pd.DataFrame(diagnostics)
    if "first_event_day" in diag.columns:
        diag["first_event_day"] = diag["first_event_day"].astype("Int64")
    return plots, diag


# --------------------------------------------------------------------------
# Demo week (clearly synthetic; production uses Open-Meteo)
# --------------------------------------------------------------------------
def _canopy_cover(age: float, tph: int) -> float:
    area = 10000.0 / tph
    d = min(6.0 * (1.0 - math.exp(-age / 12.0)), 1.1 * math.sqrt(area))
    return round(min(0.85, (math.pi * d * d / 4.0) / area) * 100.0, 1)


DEMO_PROFILES = [
    # id, trees, age, tph, soil, depth, stage, month, moisture, system, flow, dep, raw, taw
    ("P01", 120, 18, 204, "sandy_loam", 80, "fruit_ripening", 7, 18.0, "drip", 12.0, 40, 62, 96),
    ("P02", 400, 4, 400, "loam", 40, "fruit_ripening", 7, 18.0, "drip", 8.0, 28, 34, 52),
    ("P03", 80, 30, 100, "clay", 120, "fruit_ripening", 7, 30.0, "surface", 15.0, 72, 109, 168),
    ("P04", 200, 12, 250, "sandy", 80, "winter_rest", 1, 10.0, "drip", 6.0, 16, 39, 60),
    ("P05", 100, 25, 150, "clay_loam", 120, "flowering", 4, 24.0, "micro_sprinkler", 10.0, 84, 109, 168),
    ("P06", 350, 2, 400, "sandy_loam", 40, "flowering", 4, 16.0, "drip", 8.0, 28, 31, 48),
]

WEATHER = {
    7: dict(temp_max=37.0, temp_min=21.0, et0=6.4, day0=195),
    4: dict(temp_max=27.0, temp_min=13.0, et0=4.7, day0=100),
    1: dict(temp_max=18.0, temp_min=7.0, et0=1.7, day0=10),
}
RAIN = {  # forecast rain by day for the winter plot
    "P04": [5.0, 4.0, 3.0, 0.0, 0.0, 0.0, 0.0],
}


def demo_week() -> pd.DataFrame:
    rows = []
    for (pid, trees, age, tph, soil, depth, stage, month, moisture,
         system, flow, dep, raw, taw) in DEMO_PROFILES:
        wx = WEATHER[month]
        cover = _canopy_cover(age, tph)
        for d in range(WEEK):
            rain = RAIN.get(pid, [0.0] * WEEK)[d]
            forecast3 = sum(RAIN.get(pid, [0.0] * WEEK)[d + 1:d + 4])
            rows.append({
                "plot_id": pid,
                "day_of_year": wx["day0"] + d,
                "month": month,
                "temp_max": wx["temp_max"] + 0.4 * d,
                "temp_min": wx["temp_min"],
                "rain_mm": rain,
                "et0_mm": wx["et0"] + 0.08 * d,
                "rain_forecast_3d_mm": forecast3,
                "argan_stage": stage,
                "tree_age_years": age,
                "trees_per_ha": tph,
                "canopy_cover_pct": cover,
                "soil_type": soil,
                "soil_depth_cm": depth,
                "soil_moisture_pct": max(5.0, moisture - 0.10 * d),
                "trees": trees,
                "irrigation_system": system,
                "pump_flow_m3_per_h": flow,
                "depletion_mm": dep,
                "raw_mm": raw,
                "taw_mm": taw,
                "priority": 1.0,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv=None) -> None:
    args = parse_args(argv)
    if args.budget < 0:
        sys.exit("[input] budget cannot be negative")
    bundles = load_models(args.model_dir)
    df = check_week_frame(demo_week() if args.demo else pd.read_csv(args.csv))

    plots, diagnostics = weekly_plan(df, bundles, args.threshold, args.rain_skip_mm)
    config = AllocationConfig(
        floor_fraction=args.floor_fraction,
        gamma=args.gamma,
        rain_skip_mm=args.rain_skip_mm,
    )
    result = allocate(plots, args.budget, config)

    print(f"[models] classifier + Regressor B (positive-only), threshold {args.threshold}")
    print(f"[week] {len(plots)} plots, {len(df)} plot-days")
    print(diagnostics.to_string(index=False))
    print_result(result)

    if args.out is not None:
        diag = {d["plot"]: d for d in diagnostics.to_dict("records")}
        plan = pd.DataFrame([{
            "plot": r.plot_id,
            "stage": r.stage,
            "trees": r.trees,
            "max_prob": diag[r.plot_id]["max_prob"],
            "first_event_day": diag[r.plot_id]["first_event_day"],
            "days_over_threshold": diag[r.plot_id]["days_over_threshold"],
            "rain_week_mm": diag[r.plot_id]["rain_week_mm"],
            "need_L_per_tree_week": r.need_L_per_tree_week,
            "allocated_L_per_tree_week": r.allocated_L_per_tree_week,
            "pct_of_need": r.pct_of_need,
            "allocated_m3": r.allocated_m3,
            "events": r.events,
            "days": r.irrigation_days,
            "litres_per_event": r.litres_per_event,
            "pump_minutes_total": r.pump_minutes_total,
            "pump_minutes_per_event": r.pump_minutes_per_event,
            "stress": r.stress_level,
            "loss_score": r.loss_score_pct,
            "status": r.status,
        } for r in result.rows])
        plan.to_csv(args.out, index=False)
        print(f"\n[out] wrote {args.out}")


if __name__ == "__main__":
    main()
