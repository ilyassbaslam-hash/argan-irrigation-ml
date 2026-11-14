#!/usr/bin/env python3
"""
Train the Random Forest models for the argan irrigation recommender.

  * irrigation_classifier.joblib            -> irrigate_yes_no       (0 / 1)
  * irrigation_regressor.joblib             -> litres, all rows      (benchmark)
  * irrigation_regressor_positive_only.joblib -> litres, irrigation days

The production app uses the classifier plus the positive-only regressor
(Regressor B).  The all-rows regressor (Regressor A) is trained and reported
only as a comparison.

The 50 daily rows of one plot are strongly related, so the test set is built
from *plots* the model never saw (GroupShuffleSplit on plot_id) whenever the
full dataset (which still contains plot_id) is available. Training on random
rows would leak a plot's behaviour into the test set and inflate the scores.

Usage
-----
    python train_model.py
    python train_model.py --data data/argan_dataset_train.csv \
                          --group-file data/argan_dataset_full.csv
    python train_model.py --no-groups      # ignore plot_id (random row split)
    python train_model.py --no-figures     # skip the PNG reports
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from schema import CATEGORICAL, FEATURES, LABELS

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data" / "argan_dataset_train.csv"
DEFAULT_GROUP_FILE = HERE / "data" / "argan_dataset_full.csv"
MODEL_DIR = HERE / "models"
REPORT_DIR = HERE / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 600

# Hyperparameters chosen with grouped 5-fold cross-validation on the training
# plots (see README): higher max_features than the usual sqrt because the 14
# inputs are few and several of them are strongly interacting.
CLASSIFIER_PARAMS = dict(
    n_estimators=N_ESTIMATORS,
    max_features=1.0,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
REGRESSOR_PARAMS = dict(
    n_estimators=N_ESTIMATORS,
    max_features=0.66,
    min_samples_leaf=3,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train the argan irrigation models.")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA,
                    help="training CSV with the 14 features and 2 labels")
    ap.add_argument("--group-file", type=Path, default=DEFAULT_GROUP_FILE,
                    help="full dataset containing plot_id; only used for splitting")
    ap.add_argument("--no-groups", action="store_true",
                    help="split random rows instead of whole plots (not recommended)")
    ap.add_argument("--output-dir", type=Path, default=HERE,
                    help="where models/ and reports/ are written")
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    ap.add_argument("--test-size", type=float, default=TEST_SIZE)
    ap.add_argument("--no-figures", action="store_true",
                    help="do not write the PNG report figures")
    return ap.parse_args(argv)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[data] file not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + LABELS if c not in df.columns]
    if missing:
        sys.exit(f"[data] missing columns: {missing}")
    if df[FEATURES + LABELS].isna().any().any():
        bad = df[FEATURES + LABELS].isna().sum()
        sys.exit(f"[data] missing values:\n{bad[bad > 0]}")
    unknown_stage = set(df["argan_stage"]) - set(
        ["winter_rest", "flowering", "fruit_growth", "fruit_ripening"])
    if unknown_stage:
        sys.exit(f"[data] unknown argan_stage values: {sorted(unknown_stage)}")
    unknown_soil = set(df["soil_type"]) - set(
        ["sandy", "sandy_loam", "loam", "clay_loam", "clay"])
    if unknown_soil:
        sys.exit(f"[data] unknown soil_type values: {sorted(unknown_soil)}")
    if not df["irrigate_yes_no"].isin([0, 1]).all():
        sys.exit("[data] irrigate_yes_no must be 0 or 1")
    if (df["irrigation_L_per_tree"] < 0).any():
        sys.exit("[data] irrigation_L_per_tree contains negative values")
    return df


def load_groups(df: pd.DataFrame, group_path: Path) -> pd.Series | None:
    """Return plot_id aligned with df, or None when it cannot be trusted."""
    if not group_path.exists():
        print(f"[split] no group file at {group_path}; falling back to a random row split")
        return None
    gdf = pd.read_csv(group_path)
    if len(gdf) != len(df) or "plot_id" not in gdf.columns:
        print("[split] group file does not match the training rows; using a random row split")
        return None
    same = gdf[FEATURES + LABELS].reset_index(drop=True).equals(df[FEATURES + LABELS].reset_index(drop=True))
    if not same:
        print("[split] group file rows differ from the training file; using a random row split")
        return None
    print(f"[split] using plot_id from {group_path.name}: {gdf['plot_id'].nunique()} plots")
    return gdf["plot_id"].astype(str)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
        remainder="passthrough",
        verbose_feature_names_out=True,
    )


def build_classifier(seed: int) -> Pipeline:
    params = dict(CLASSIFIER_PARAMS, random_state=seed)
    return Pipeline([
        ("pre", make_preprocessor()),
        ("model", RandomForestClassifier(**params)),
    ])


def build_regressor(seed: int) -> Pipeline:
    params = dict(REGRESSOR_PARAMS, random_state=seed)
    return Pipeline([
        ("pre", make_preprocessor()),
        ("model", RandomForestRegressor(**params)),
    ])


def split_rows(df: pd.DataFrame, groups: pd.Series | None, test_size: float,
               seed: int) -> tuple[np.ndarray, np.ndarray]:
    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        return next(splitter.split(df[FEATURES], df["irrigate_yes_no"], groups=groups))
    idx_tr, idx_te = train_test_split(
        np.arange(len(df)), test_size=test_size, random_state=seed,
        stratify=df["irrigate_yes_no"],
    )
    return idx_tr, idx_te


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def classification_metrics(y_true, y_pred, proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "brier_score": float(brier_score_loss(y_true, proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "predicted_positive_rate": float(np.mean(y_pred)),
        "actual_positive_rate": float(np.mean(y_true)),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def evaluate(clf: Pipeline, reg: Pipeline, reg_pos: Pipeline, df: pd.DataFrame,
             idx_tr: np.ndarray, idx_te: np.ndarray) -> dict:
    X_tr, X_te = df[FEATURES].iloc[idx_tr], df[FEATURES].iloc[idx_te]
    y_cls = df["irrigate_yes_no"].iloc[idx_te]
    y_reg = df["irrigation_L_per_tree"].iloc[idx_te]

    proba = clf.predict_proba(X_te)[:, 1]
    pred_cls = clf.predict(X_te)
    pred_reg = reg.predict(X_te)

    dummy_cls = DummyClassifier(strategy="most_frequent").fit(
        X_tr, df["irrigate_yes_no"].iloc[idx_tr])
    dummy_reg = DummyRegressor(strategy="mean").fit(
        X_tr, df["irrigation_L_per_tree"].iloc[idx_tr])

    metrics = {
        "classifier": classification_metrics(y_cls, pred_cls, proba),
        "classifier_baseline": {
            "accuracy": float(accuracy_score(y_cls, dummy_cls.predict(X_te))),
            "f1": float(f1_score(y_cls, dummy_cls.predict(X_te), zero_division=0)),
        },
        "regressor": regression_metrics(y_reg.to_numpy(), pred_reg),
        "regressor_baseline": regression_metrics(
            y_reg.to_numpy(), dummy_reg.predict(X_te)),
    }

    positive = y_reg.to_numpy() > 0
    if positive.any():
        metrics["regressor_on_true_positive_rows"] = regression_metrics(
            y_reg.to_numpy()[positive], pred_reg[positive])
        metrics["regressor_positive_only"] = regression_metrics(
            y_reg.to_numpy()[positive], reg_pos.predict(X_te)[positive])

    return metrics


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def original_feature(name: str) -> str:
    body = name.split("__", 1)[-1]
    if name.startswith("cat__"):
        for col in CATEGORICAL:
            if body.startswith(col + "_"):
                return col
    return body


def feature_importance_table(clf: Pipeline, reg: Pipeline) -> pd.DataFrame:
    names = clf.named_steps["pre"].get_feature_names_out()
    table = pd.DataFrame({
        "feature": [original_feature(n) for n in names],
        "classifier_importance": clf.named_steps["model"].feature_importances_,
        "regressor_importance": reg.named_steps["model"].feature_importances_,
    })
    table = table.groupby("feature", as_index=False).sum()
    table["mean_importance"] = table[["classifier_importance", "regressor_importance"]].mean(axis=1)
    return table.sort_values("mean_importance", ascending=False).reset_index(drop=True)


def save_figures(clf: Pipeline, reg: Pipeline, df: pd.DataFrame, idx_te: np.ndarray,
                 importance: pd.DataFrame) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve
    except ImportError:
        print("[figures] matplotlib is not installed; skipping PNG reports")
        return []

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    X_te = df[FEATURES].iloc[idx_te]
    y_cls = df["irrigate_yes_no"].iloc[idx_te]
    y_reg = df["irrigation_L_per_tree"].iloc[idx_te]
    proba = clf.predict_proba(X_te)[:, 1]
    pred_reg = reg.predict(X_te)

    # 1. confusion matrix
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ConfusionMatrixDisplay.from_predictions(
        y_cls, clf.predict(X_te), labels=[0, 1],
        display_labels=["no irrigation", "irrigate"], cmap="Greens",
        ax=ax, colorbar=False)
    ax.set_title("Classifier — test plots")
    fig.tight_layout()
    path = FIGURE_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    # 2. ROC and precision-recall
    fpr, tpr, _ = roc_curve(y_cls, proba)
    prec, rec, _ = precision_recall_curve(y_cls, proba)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    axes[0].plot(fpr, tpr, color="#2e7d32", lw=2)
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    axes[0].set(xlabel="false positive rate", ylabel="true positive rate",
                title=f"ROC (AUC = {roc_auc_score(y_cls, proba):.3f})")
    axes[1].plot(rec, prec, color="#2e7d32", lw=2)
    axes[1].set(xlabel="recall", ylabel="precision",
                title=f"Precision-recall (AP = {average_precision_score(y_cls, proba):.3f})")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = FIGURE_DIR / "roc_precision_recall.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    # 3. predicted vs actual litres
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    axes[0].scatter(y_reg, pred_reg, s=9, alpha=0.35, color="#2e7d32", edgecolors="none")
    lim = max(y_reg.max(), pred_reg.max()) * 1.05
    axes[0].plot([0, lim], [0, lim], "--", color="grey", lw=1)
    axes[0].set(xlabel="label (L per tree)", ylabel="prediction (L per tree)",
                title=f"Regressor — test plots (R² = {r2_score(y_reg, pred_reg):.3f})",
                xlim=(0, lim), ylim=(0, lim))
    axes[1].hist(pred_reg - y_reg, bins=50, color="#2e7d32", alpha=0.8)
    axes[1].set(xlabel="error (prediction − label), L per tree", ylabel="rows",
                title="Residuals")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = FIGURE_DIR / "prediction_vs_actual.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    # 4. feature importances
    top = importance.head(12).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), sharey=True)
    axes[0].barh(top["feature"], top["classifier_importance"], color="#2e7d32")
    axes[0].set_title("Classifier importance")
    axes[1].barh(top["feature"], top["regressor_importance"], color="#1565c0")
    axes[1].set_title("Regressor importance")
    fig.tight_layout()
    path = FIGURE_DIR / "feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))
    return written


def print_summary(metrics: dict, n_rows: int, groups: pd.Series | None,
                  idx_te: np.ndarray, paths: dict) -> None:
    c, cb = metrics["classifier"], metrics["classifier_baseline"]
    r, rb = metrics["regressor"], metrics["regressor_baseline"]
    print("\n" + "=" * 68)
    print("TRAINING SUMMARY")
    print("=" * 68)
    print(f"Rows: {n_rows:,}  |  irrigation days: {metrics['dataset']['positive_rate']:.1%}")
    if groups is not None:
        print(f"Plots: {groups.nunique()}  |  test rows from "
              f"{groups.iloc[idx_te].nunique()} unseen plots")
    print(f"Split: {metrics['dataset']['n_train']:,} train / "
          f"{metrics['dataset']['n_test']:,} test rows")
    print("\nClassifier — will the farmer irrigate today?")
    print(f"  accuracy {c['accuracy']:.3f} | balanced {c['balanced_accuracy']:.3f} | "
          f"precision {c['precision']:.3f} | recall {c['recall']:.3f}")
    print(f"  F1 {c['f1']:.3f} | ROC-AUC {c['roc_auc']:.3f} | PR-AUC {c['average_precision']:.3f}")
    print(f"  baseline (always 'no'): accuracy {cb['accuracy']:.3f} | F1 {cb['f1']:.3f}")
    print("\nRegressor A — litres per tree (trained on all rows, zeros included)")
    print(f"  R² {r['r2']:.3f} | MAE {r['mae']:.1f} L | RMSE {r['rmse']:.1f} L | "
          f"median AE {r['median_absolute_error']:.1f} L")
    print(f"  baseline (predict the mean): R² {rb['r2']:.3f} | MAE {rb['mae']:.1f} L")
    if "regressor_positive_only" in metrics:
        rp = metrics["regressor_positive_only"]
        ra = metrics["regressor_on_true_positive_rows"]
        print("\nRegressor B — litres on irrigation days only (use after the classifier says yes)")
        print(f"  on rows that truly irrigate: R² {rp['r2']:.3f} | MAE {rp['mae']:.1f} L "
              f"({rp['n']} rows)")
        print(f"  same rows, model A:          R² {ra['r2']:.3f} | MAE {ra['mae']:.1f} L")
    print("\nSaved:")
    for label, path in paths.items():
        print(f"  {label:<22} {path}")
    print("\nReminder: the model reproduces the dataset's label logic, which is itself")
    print("an FAO-56-style approximation for argan. Validate on real field data before use.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv=None) -> None:
    args = parse_args(argv)
    out_dir = args.output_dir.resolve()
    model_dir = out_dir / "models"
    report_dir = out_dir / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    groups = None if args.no_groups else load_groups(df, args.group_file)
    idx_tr, idx_te = split_rows(df, groups, args.test_size, args.seed)

    print(f"[train] fitting Random Forest on {len(idx_tr):,} rows ...")
    clf = build_classifier(args.seed)
    clf.fit(df[FEATURES].iloc[idx_tr], df["irrigate_yes_no"].iloc[idx_tr])

    print(f"[train] fitting Random Forest regressor on {len(idx_tr):,} rows ...")
    reg = build_regressor(args.seed)
    reg.fit(df[FEATURES].iloc[idx_tr], df["irrigation_L_per_tree"].iloc[idx_tr])

    pos_mask = df["irrigate_yes_no"].iloc[idx_tr].to_numpy() == 1
    print(f"[train] fitting litres regressor on the {pos_mask.sum():,} irrigation days ...")
    reg_pos = build_regressor(args.seed)
    reg_pos.fit(df[FEATURES].iloc[idx_tr][pos_mask],
                df["irrigation_L_per_tree"].iloc[idx_tr][pos_mask])

    metrics = evaluate(clf, reg, reg_pos, df, idx_tr, idx_te)
    metrics["dataset"] = {
        "rows": int(len(df)),
        "plots": int(groups.nunique()) if groups is not None else None,
        "n_train": int(len(idx_tr)),
        "n_test": int(len(idx_te)),
        "positive_rate": float(df["irrigate_yes_no"].mean()),
        "test_plots": int(groups.iloc[idx_te].nunique()) if groups is not None else None,
        "features": FEATURES,
        "labels": LABELS,
    }
    metrics["model"] = {
        "type": "RandomForestClassifier + RandomForestRegressor",
        "classifier_params": {k: v for k, v in CLASSIFIER_PARAMS.items() if k != "n_jobs"},
        "regressor_params": {k: v for k, v in REGRESSOR_PARAMS.items() if k != "n_jobs"},
        "split": "GroupShuffleSplit on plot_id" if groups is not None else "random rows",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_state": args.seed,
    }

    importance = feature_importance_table(clf, reg)
    paths = {
        "classifier": model_dir / "irrigation_classifier.joblib",
        "regressor (all rows)": model_dir / "irrigation_regressor.joblib",
        "regressor (positive days)": model_dir / "irrigation_regressor_positive_only.joblib",
        "metrics": report_dir / "metrics.json",
        "feature report": report_dir / "feature_importance.csv",
    }

    joblib.dump({
        "pipeline": clf, "features": FEATURES, "categorical": CATEGORICAL,
        "target": "irrigate_yes_no", "metrics": metrics["classifier"],
    }, paths["classifier"], compress=3)
    joblib.dump({
        "pipeline": reg, "features": FEATURES, "categorical": CATEGORICAL,
        "target": "irrigation_L_per_tree", "trained_on": "all rows",
        "metrics": metrics["regressor"],
    }, paths["regressor (all rows)"], compress=3)
    joblib.dump({
        "pipeline": reg_pos, "features": FEATURES, "categorical": CATEGORICAL,
        "target": "irrigation_L_per_tree", "trained_on": "irrigation days only",
        "metrics": metrics.get("regressor_positive_only"),
    }, paths["regressor (positive days)"], compress=3)

    importance.round(5).to_csv(paths["feature report"], index=False)
    if not args.no_figures:
        figures = save_figures(clf, reg, df, idx_te, importance)
        if figures:
            paths["figures"] = report_dir / "figures"

    paths["metrics"].write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print_summary(metrics, len(df), groups, idx_te, paths)


if __name__ == "__main__":
    main()
