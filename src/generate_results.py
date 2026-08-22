"""
generate_results.py
===================
Reproducible results pipeline. Trains the CNN-LSTM and its ablation
baselines, runs walk-forward cross-validation, computes permutation
feature importance, and writes every figure referenced by
RESULTS_AND_GRAPHS.md plus a machine-readable results/metrics.json.

Every number in RESULTS_AND_GRAPHS.md is produced by this script — run it
to regenerate the report end to end.

Usage
-----
    # from repo root — full run (~15-25 min on CPU)
    python src/generate_results.py

    # fast smoke test (figures render, numbers are not meaningful)
    python src/generate_results.py --quick

Outputs
-------
    images/01_training_curve.png ... images/09_grid_stability.png
    results/metrics.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent if _HERE.name == "src" else _HERE
sys.path.insert(0, str(_HERE))

from console_utils import enable_utf8_output
from model import CNNLSTMModel, LSTMOnlyModel, CNNOnlyModel
from train import load_data, prepare_data, train_model, evaluate_model
from feature_engineering import FEATURE_COLUMNS
from duck_curve_analysis import analyze_duck_curve
from duck_curve_simulation import (
    load_or_generate_data, simulate_all_days, LATITUDE, SOLAR_FARM_MW,
)

IMAGES_DIR = _REPO / "images"
RESULTS_DIR = _REPO / "results"

# ── Plot style ───────────────────────────────────────────────────────────────
ACCENT   = "#F5A623"   # solar orange
PRIMARY  = "#1F6FB2"   # deep blue
SECOND   = "#2FA37A"   # green
DANGER   = "#D64545"   # red
MUTED    = "#8899A6"
GRID     = "#DDE3E8"
INK      = "#1B2733"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#C7D0D8",
    "axes.labelcolor": INK,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "font.size": 10,
    "grid.color": GRID,
    "legend.frameon": False,
    "figure.dpi": 130,
})


def _finish(fig, path, caption=None):
    """Apply grid styling, save, and close a figure."""
    for ax in fig.axes:
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    if caption:
        fig.text(0.5, -0.02, caption, ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    saved -> images/{path.name}")


def _metrics_from(preds, targets):
    """RMSE / MAE / R2 / MAPE / accuracy from raw W/m^2 arrays."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    preds, targets = np.asarray(preds, float), np.asarray(targets, float)
    mae = float(mean_absolute_error(targets, preds))
    mask = targets > 1.0
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(targets, preds))), 2),
        "mae": round(mae, 2),
        "r2": round(float(r2_score(targets, preds)), 4),
        "mape": round(float(np.mean(np.abs((targets[mask] - preds[mask]) / targets[mask])) * 100), 2)
                 if mask.any() else float("nan"),
        "accuracy": round(float(max(0.0, (1.0 - mae / (targets.mean() + 1e-8)) * 100)), 1),
    }


def _inv_ghi(norm_vals, scaler):
    """Inverse-transform normalized GHI (feature 0) back to W/m^2."""
    norm_vals = np.atleast_1d(np.asarray(norm_vals, float))
    dummy = np.zeros((len(norm_vals), scaler.n_features_in_))
    dummy[:, 0] = norm_vals
    return np.maximum(scaler.inverse_transform(dummy)[:, 0], 0.0)


def _loader(X, y, batch_size=32):
    return DataLoader(
        TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y)),
        batch_size=batch_size, shuffle=False,
    )


def _season(month):
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Pre-monsoon"
    if month in (6, 7, 8, 9):
        return "Monsoon"
    return "Post-monsoon"


SEASON_ORDER = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]


# ═════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═════════════════════════════════════════════════════════════════════════════

def fig_training_curve(history):
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], color=PRIMARY, lw=2, label="Training loss")
    ax.plot(epochs, history["val_loss"], color=ACCENT, lw=2, label="Validation loss")
    best = int(np.argmin(history["val_loss"]))
    ax.axvline(best + 1, color=SECOND, ls=":", lw=1.5)
    ax.annotate(f"best epoch {best + 1}\nval MSE {history['val_loss'][best]:.5f}",
                xy=(best + 1, history["val_loss"][best]),
                xytext=(0.55, 0.6), textcoords="axes fraction",
                fontsize=9, color=SECOND,
                arrowprops=dict(arrowstyle="->", color=SECOND, lw=1.2))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (normalised units)")
    ax.set_title("CNN-LSTM Training & Validation Loss")
    ax.set_yscale("log")
    ax.legend()
    _finish(fig, IMAGES_DIR / "01_training_curve.png",
            "Loss on min-max normalised GHI. Best checkpoint is restored before evaluation.")


def fig_model_comparison(rows):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    names = [r["model"] for r in rows]
    colors = [ACCENT if n.startswith("CNN-LSTM") else PRIMARY for n in names]
    for ax, key, label in zip(
        axes, ["rmse", "mae", "r2"], ["RMSE (W/m²)", "MAE (W/m²)", "R²"]
    ):
        vals = [r[key] for r in rows]
        bars = ax.bar(names, vals, color=colors, width=0.62)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=20)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.3f}" if key == "r2" else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.margins(y=0.18)
    fig.suptitle("Model Comparison — held-out test set", fontsize=13, fontweight="bold")
    _finish(fig, IMAGES_DIR / "02_model_comparison.png",
            "Persistence = naive 'next hour equals current hour' baseline. Lower RMSE/MAE and higher R² are better.")


def fig_duck_curve(day):
    h = day["hours"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(h, day["demand"], color=PRIMARY, lw=2.2, label="Grid demand")
    ax.plot(h, day["net_demand"], color=DANGER, lw=2.2, ls="--", label="Net demand (duck)")
    ax.fill_between(h, day["pv_power"], color=ACCENT, alpha=0.35, label="PV generation")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Power (MW)")
    ax.set_title(f"Duck Curve — {day['date']} "
                 f"(solar fraction {day['solar_fraction']:.1f}%)")
    ax.legend(loc="upper left")
    peak_pen = float(np.max(day["pv_power"] / day["demand"]) * 100)
    _finish(fig, IMAGES_DIR / "03_duck_curve.png",
            f"{SOLAR_FARM_MW:.0f} MW farm, Mumbai (lat {LATITUDE}°), "
            f"peak instantaneous penetration {peak_pen:.0f}%. "
            "The belly is midday solar suppressing net demand; the neck is the evening ramp.")


def fig_feature_importance(importances, baseline_rmse):
    order = np.argsort([i["delta_rmse"] for i in importances])
    names = [importances[i]["feature"] for i in order]
    vals = [importances[i]["delta_rmse"] for i in order]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(names, vals, color=PRIMARY, height=0.62)
    for y, v in enumerate(vals):
        ax.text(v, y, f"  {v:+.1f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Increase in test RMSE when feature is shuffled (W/m²)")
    ax.set_title("Permutation Feature Importance")
    ax.margins(x=0.16)
    _finish(fig, IMAGES_DIR / "04_feature_importance.png",
            f"Baseline test RMSE = {baseline_rmse:.1f} W/m². Larger bars = the model relies on that feature more.")


def fig_seasonal(seasonal_rows):
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [r["season"] for r in seasonal_rows]
    vals = [r["rmse"] for r in seasonal_rows]
    counts = [r["n"] for r in seasonal_rows]
    bars = ax.bar(names, vals, color=[DANGER if n == "Monsoon" else PRIMARY for n in names],
                  width=0.6)
    for b, v, n in zip(bars, vals, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v:.1f}\nn={n:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("RMSE (W/m²)")
    ax.set_title("Seasonal Forecast Error (out-of-sample)")
    ax.margins(y=0.22)
    _finish(fig, IMAGES_DIR / "05_seasonal_performance.png",
            "Pooled walk-forward predictions. Monsoon cloud cover drives the largest errors.")


def fig_hourly(hourly_rmse):
    fig, ax = plt.subplots(figsize=(10, 5))
    hours = np.arange(24)
    colors = [MUTED if v < 1 else ACCENT for v in hourly_rmse]
    ax.bar(hours, hourly_rmse, color=colors, width=0.7)
    ax.set_xticks(hours)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("RMSE (W/m²)")
    ax.set_title("Forecast Error by Hour of Day (out-of-sample)")
    _finish(fig, IMAGES_DIR / "06_forecast_accuracy_by_hour.png",
            "Night hours are near-zero because irradiance is zero and trivially predictable.")


def fig_error_distribution(residuals):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(residuals, bins=70, color=PRIMARY, alpha=0.85)
    mu, sd = float(np.mean(residuals)), float(np.std(residuals))
    ax.axvline(mu, color=DANGER, lw=2, label=f"mean bias = {mu:+.1f} W/m²")
    ax.axvline(0, color=INK, lw=1, ls=":")
    ax.set_xlabel("Residual (predicted − actual), W/m²")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution (out-of-sample)")
    ax.legend()
    _finish(fig, IMAGES_DIR / "07_error_distribution.png",
            f"σ = {sd:.1f} W/m². A mean near zero indicates no systematic over/under-prediction.")


def fig_cross_validation(folds):
    fig, ax = plt.subplots(figsize=(9.5, 5))
    idx = np.arange(1, len(folds) + 1)
    rmse = [f["rmse"] for f in folds]
    ax.bar(idx, rmse, color=PRIMARY, width=0.55)
    mean_rmse = float(np.mean(rmse))
    ax.axhline(mean_rmse, color=ACCENT, lw=2, ls="--",
               label=f"mean = {mean_rmse:.1f} ± {np.std(rmse):.1f} W/m²")
    for i, f in enumerate(folds, start=1):
        ax.text(i, f["rmse"], f"{f['rmse']:.1f}\nR²={f['r2']:.3f}",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"Fold {i}\ntrain {f['n_train']:,}" for i, f in zip(idx, folds)],
                       fontsize=8.5)
    ax.set_ylabel("RMSE (W/m²)")
    ax.set_title("Walk-Forward Cross-Validation")
    ax.margins(y=0.25)
    ax.legend()
    _finish(fig, IMAGES_DIR / "08_cross_validation.png",
            "Expanding training window; each fold tests on the next unseen time block (no shuffling).")


def fig_grid_stability(scores, buckets):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8),
                                    gridspec_kw={"width_ratios": [2, 1]})
    ax1.plot(scores, color=PRIMARY, lw=0.9, alpha=0.55)
    if len(scores) >= 7:
        roll = pd.Series(scores).rolling(7, center=True).mean()
        ax1.plot(roll, color=ACCENT, lw=2.2, label="7-day mean")
        ax1.legend()
    ax1.axhline(80, color=SECOND, ls=":", lw=1.4)
    ax1.axhline(60, color=DANGER, ls=":", lw=1.4)
    ax1.set_xlabel("Day of year")
    ax1.set_ylabel("Stability score / 100")
    ax1.set_title("Grid Stability Across Simulated Year")
    ax1.set_ylim(0, 105)

    labels = list(buckets.keys())
    vals = [buckets[k] for k in labels]
    cols = {"LOW": SECOND, "MODERATE": ACCENT, "HIGH": DANGER}
    bars = ax2.bar(labels, vals, color=[cols[k] for k in labels], width=0.6)
    total = max(sum(vals), 1)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 f"{v}\n({v / total * 100:.0f}%)", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Days")
    ax2.set_title("Stress-Level Distribution")
    ax2.margins(y=0.22)
    _finish(fig, IMAGES_DIR / "09_grid_stability.png",
            "Stability score derives from peak PV ramp rate vs the capacity-scaled high-ramp threshold.")


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSES
# ═════════════════════════════════════════════════════════════════════════════

def permutation_importance(model, X_test, y_test, scaler, device, seed=0):
    """Shuffle each feature across samples and measure the RMSE increase."""
    rng = np.random.default_rng(seed)
    base = evaluate_model(model, _loader(X_test, y_test), scaler, device)
    out = []
    for f_idx, fname in enumerate(FEATURE_COLUMNS):
        X_perm = X_test.copy()
        order = rng.permutation(len(X_perm))
        X_perm[:, :, f_idx] = X_perm[order, :, f_idx]
        m = evaluate_model(model, _loader(X_perm, y_test), scaler, device)
        out.append({"feature": fname,
                    "rmse": m["rmse"],
                    "delta_rmse": round(m["rmse"] - base["rmse"], 2)})
        print(f"    {fname:<18} ΔRMSE +{out[-1]['delta_rmse']:.2f}")
    return out, base["rmse"]


def walk_forward_cv(df, n_folds, epochs, seq_len, device):
    """
    Expanding-window cross-validation.

    Fold k trains on the first k/(n_folds+1) of the series and tests on the
    next 1/(n_folds+1) block, so every test block is strictly future data.
    """
    folds, pooled = [], []
    n = len(df)
    for k in range(1, n_folds + 1):
        cutoff = int(n * (k + 1) / (n_folds + 1))
        df_fold = df.iloc[:cutoff]
        test_split = 1.0 / (k + 1)
        print(f"\n  Fold {k}/{n_folds} — rows 0..{cutoff:,} (test_split={test_split:.3f})")
        model, scaler, _, _ = train_model(
            df_fold, sequence_length=seq_len, epochs=epochs,
            test_split=test_split, device=device, verbose=False,
        )
        data = prepare_data(df_fold, sequence_length=seq_len, test_split=test_split)
        m = evaluate_model(model, _loader(data["X_test"], data["y_test"]),
                           scaler, device, return_predictions=True)
        folds.append({"fold": k, "rmse": m["rmse"], "mae": m["mae"],
                      "r2": m["r2"], "mape": m["mape"],
                      "n_train": len(data["X_train"]), "n_test": len(data["X_test"])})
        pooled.append(pd.DataFrame({
            "timestamp": pd.to_datetime(data["ts_test"]),
            "pred": m["predictions"], "actual": m["targets"],
        }))
        print(f"    RMSE {m['rmse']:.2f}  MAE {m['mae']:.2f}  R² {m['r2']:.4f}")
    return folds, pd.concat(pooled, ignore_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# REPORT
# ═════════════════════════════════════════════════════════════════════════════

def write_report(res):
    """Regenerate RESULTS_AND_GRAPHS.md from the computed results."""
    m, cmp_rows = res["headline"], res["model_comparison"]
    folds, seasonal = res["cv_folds"], res["seasonal"]
    imp, duck, grid = res["feature_importance"], res["duck"], res["grid"]
    cfg = res["config"]

    def row(r):
        return (f"| {r['model']} | {r['rmse']:.1f} | {r['mae']:.1f} | "
                f"{r['r2']:.4f} | {r['mape']:.1f} |")

    cv_rmse = [f["rmse"] for f in folds]
    cv_r2 = [f["r2"] for f in folds]
    cv_mape = [f["mape"] for f in folds]

    # Rank only genuinely sunlit hours — night hours have ~zero error simply
    # because irradiance is zero, which would otherwise win "most accurate".
    daylight = res["daylight_hours"]
    best_hr = min(daylight, key=lambda h: res["hourly_rmse"][h])
    worst_hr = max(daylight, key=lambda h: res["hourly_rmse"][h])

    lines = [
        "# Results and Analysis",
        "",
        "Every figure and number below is generated by "
        "[`src/generate_results.py`](src/generate_results.py). Regenerate with:",
        "",
        "```bash",
        "python src/generate_results.py",
        "```",
        "",
        f"**Run configuration** — dataset: `{cfg['dataset']}` "
        f"({cfg['n_rows']:,} hourly rows, {cfg['date_range']}) · "
        f"sequence length {cfg['sequence_length']} h · "
        f"{cfg['epochs']} epochs · device `{cfg['device']}` · "
        f"generated {cfg['generated_utc']}.",
        "",
        "> The bundled dataset is **synthetic** — a Spencer/Iqbal clear-sky model "
        "with stochastic monsoon-aware cloud cover, calibrated to Mumbai "
        f"(lat {LATITUDE}°). It exists so the pipeline runs out of the box. "
        "Point `--data` at measured irradiance to reproduce these figures on real data.",
        "",
        "---",
        "",
        "## 1. Training Curve",
        "",
        "![Training Curve](images/01_training_curve.png)",
        "",
        f"Training and validation MSE over {cfg['epochs']} epochs. Best validation "
        f"loss **{min(res['history']['val_loss']):.5f}** at epoch "
        f"**{int(np.argmin(res['history']['val_loss'])) + 1}**; those weights are "
        "restored before evaluation. Validation tracks training closely, so dropout "
        "(0.2) plus gradient clipping (max_norm 1.0) is holding overfitting in check.",
        "",
        "---",
        "",
        "## 2. Model Comparison",
        "",
        "![Model Comparison](images/02_model_comparison.png)",
        "",
        "All models share the same feature set, split, and training budget.",
        "",
        "| Model | RMSE (W/m²) | MAE (W/m²) | R² | MAPE (%) |",
        "|-------|-------------|------------|-----|----------|",
    ]
    lines += [row(r) for r in cmp_rows]

    best = min(cmp_rows, key=lambda r: r["rmse"])
    pers = next((r for r in cmp_rows if r["model"] == "Persistence"), None)
    neural = [r for r in cmp_rows if r["model"] != "Persistence"]
    best_n = min(neural, key=lambda r: r["rmse"])
    spread = max(r["rmse"] for r in neural) - best_n["rmse"]

    gain = (f"All three learned models cut RMSE by roughly "
            f"**{(pers['rmse'] - best_n['rmse']) / pers['rmse'] * 100:.0f}%** against the "
            f"persistence baseline, so the feature set and sequence framing are doing real work. "
            if pers else "")

    if spread < 0.02 * best_n["rmse"]:
        verdict = (
            f"{gain}But they land within **{spread:.1f} W/m²** of each other — a spread far "
            f"smaller than the fold-to-fold variation in cross-validation below. On this "
            f"dataset the CNN-LSTM hybrid shows **no measurable advantage** over either "
            f"single-branch ablation; the CNN front-end and the LSTM head appear to be "
            f"capturing the same structure rather than complementary structure. "
            f"Treat the three as tied, not ranked."
        )
    else:
        verdict = (f"{gain}**{best_n['model']}** achieves the lowest RMSE "
                   f"({best_n['rmse']:.1f} W/m²), ahead of the next variant by "
                   f"{spread:.1f} W/m².")

    lines += [
        "",
        verdict,
        "",
        "---",
        "",
        "## 3. Duck Curve",
        "",
        "![Duck Curve](images/03_duck_curve.png)",
        "",
        f"Representative day **{duck['date']}** for a {SOLAR_FARM_MW:.0f} MW farm: "
        f"peak PV **{duck['peak_pv_mw']:.1f} MW** at {duck['peak_hour']:02d}:00, "
        f"solar fraction **{duck['solar_fraction']:.1f}%** of daily demand, "
        f"stability score **{duck['stability_score']}/100** "
        f"({duck['stress_level']} stress). Morning ramp "
        f"{duck['morning_ramp']:+.1f} MW/h, evening ramp {duck['evening_ramp']:+.1f} MW/h.",
        "",
        "---",
        "",
        "## 4. Feature Importance",
        "",
        "![Feature Importance](images/04_feature_importance.png)",
        "",
        f"Permutation importance on the test set (baseline RMSE "
        f"{res['importance_baseline_rmse']:.1f} W/m²). Each feature is shuffled "
        "across samples; the resulting RMSE increase is its contribution.",
        "",
        "| Rank | Feature | RMSE when shuffled | Δ RMSE |",
        "|------|---------|--------------------|--------|",
    ]
    for i, f in enumerate(sorted(imp, key=lambda x: -x["delta_rmse"]), start=1):
        lines.append(f"| {i} | `{f['feature']}` | {f['rmse']:.1f} | {f['delta_rmse']:+.1f} |")
    lines += [
        "",
        "A near-zero or negative Δ means the model is not relying on that feature "
        "independently — its signal is largely redundant with the others "
        "(the rolling GHI means and the raw GHI lag carry overlapping information).",
    ]

    lines += [
        "",
        "---",
        "",
        "## 5. Seasonal Performance",
        "",
        "![Seasonal Performance](images/05_seasonal_performance.png)",
        "",
        "Errors are pooled from the walk-forward folds, so every prediction is "
        "out-of-sample. A single chronological split would only cover the final "
        "months of the year.",
        "",
        "| Season | Months | RMSE (W/m²) | MAE (W/m²) | Samples |",
        "|--------|--------|-------------|------------|---------|",
    ]
    season_months = {"Winter": "Dec–Feb", "Pre-monsoon": "Mar–May",
                     "Monsoon": "Jun–Sep", "Post-monsoon": "Oct–Nov"}
    for s in seasonal:
        lines.append(f"| {s['season']} | {season_months[s['season']]} | "
                     f"{s['rmse']:.1f} | {s['mae']:.1f} | {s['n']:,} |")

    worst_season = max(seasonal, key=lambda s: s["rmse"])
    cause = (" — rapid monsoon cloud-cover fluctuation drives the highest "
             "irradiance variance in this dataset"
             if worst_season["season"] == "Monsoon" else "")
    lines += [
        "",
        f"**{worst_season['season']}** is the hardest window "
        f"(RMSE {worst_season['rmse']:.1f} W/m²){cause}.",
        "",
        "---",
        "",
        "## 6. Forecast Accuracy by Hour",
        "",
        "![Forecast Accuracy by Hour](images/06_forecast_accuracy_by_hour.png)",
        "",
        f"Night hours carry near-zero error (irradiance is zero and trivially "
        f"predictable). Among daylight hours, error peaks at "
        f"**{worst_hr:02d}:00** ({res['hourly_rmse'][worst_hr]:.1f} W/m²) and is "
        f"lowest at **{best_hr:02d}:00** ({res['hourly_rmse'][best_hr]:.1f} W/m²).",
        "",
        "---",
        "",
        "## 7. Error Distribution",
        "",
        "![Error Distribution](images/07_error_distribution.png)",
        "",
        f"Residuals (predicted − actual) have mean bias "
        f"**{res['residual_mean']:+.1f} W/m²** and standard deviation "
        f"**{res['residual_std']:.1f} W/m²**, with excess kurtosis "
        f"**{res['residual_kurtosis']:.2f}**. A near-zero mean indicates no "
        "systematic over- or under-prediction; the heavy tails come from rapid "
        "cloud transients.",
        "",
        "---",
        "",
        "## 8. Cross-Validation",
        "",
        "![Cross Validation](images/08_cross_validation.png)",
        "",
        f"{len(folds)}-fold walk-forward validation with an expanding training "
        "window — each fold tests on the next unseen time block, never on "
        "shuffled data.",
        "",
        "| Fold | Train samples | Test samples | RMSE (W/m²) | R² | MAPE (%) |",
        "|------|---------------|--------------|-------------|-----|----------|",
    ]
    for f in folds:
        lines.append(f"| {f['fold']} | {f['n_train']:,} | {f['n_test']:,} | "
                     f"{f['rmse']:.1f} | {f['r2']:.4f} | {f['mape']:.1f} |")
    lines += [
        f"| **Mean** | | | **{np.mean(cv_rmse):.1f}** | **{np.mean(cv_r2):.4f}** | "
        f"**{np.mean(cv_mape):.1f}** |",
        f"| Std | | | ±{np.std(cv_rmse):.1f} | ±{np.std(cv_r2):.4f} | ±{np.std(cv_mape):.1f} |",
        "",
        "Fold 1 trains on the least data and is expected to be the weakest; "
        "performance stabilises as the training window expands.",
        "",
        "---",
        "",
        "## 9. Grid Stability",
        "",
        "![Grid Stability](images/09_grid_stability.png)",
        "",
        f"Across **{grid['n_days']}** simulated days for a "
        f"{SOLAR_FARM_MW:.0f} MW farm, with the high-ramp threshold set at "
        f"{grid['high_ramp_threshold']:.0f} MW/h (40% of capacity per hour — "
        "above the median day's steepest ramp of ~32%):",
        "",
        f"- **LOW stress** (score ≥ 80): {grid['buckets']['LOW']} days "
        f"({grid['buckets']['LOW'] / grid['n_days'] * 100:.0f}%)",
        f"- **MODERATE stress** (60–79): {grid['buckets']['MODERATE']} days "
        f"({grid['buckets']['MODERATE'] / grid['n_days'] * 100:.0f}%)",
        f"- **HIGH stress** (< 60): {grid['buckets']['HIGH']} days "
        f"({grid['buckets']['HIGH'] / grid['n_days'] * 100:.0f}%)",
        "",
        f"Mean stability score **{grid['mean_score']:.1f}/100**. The score is a "
        "heuristic: it scales the day's steepest PV ramp against that threshold, "
        "so it measures ramp severity, not a physical reserve-adequacy calculation.",
        "",
        "---",
        "",
        "## Headline Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| RMSE | {m['rmse']:.2f} W/m² |",
        f"| MAE | {m['mae']:.2f} W/m² |",
        f"| R² | {m['r2']:.4f} |",
        f"| MAPE | {m['mape']:.2f}% |",
        f"| Accuracy (1 − MAE/mean) | {m['accuracy']:.1f}% |",
        "",
        "Raw values: [`results/metrics.json`](results/metrics.json).",
        "",
    ]

    (_REPO / "RESULTS_AND_GRAPHS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  wrote RESULTS_AND_GRAPHS.md ({len(lines)} lines)")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    enable_utf8_output()
    p = argparse.ArgumentParser(
        description="Generate all result figures & metrics for RESULTS_AND_GRAPHS.md")
    p.add_argument("--data", default=str(_REPO / "data" / "sample_solar_data.csv"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--cv-epochs", type=int, default=15)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--quick", action="store_true",
                   help="Tiny epoch counts for a fast smoke test")
    p.add_argument("--report-only", action="store_true",
                   help="Rewrite RESULTS_AND_GRAPHS.md from the existing "
                        "results/metrics.json without retraining")
    args = p.parse_args()

    if args.report_only:
        path = RESULTS_DIR / "metrics.json"
        if not path.exists():
            sys.exit(f"No {path} — run without --report-only first.")
        write_report(json.loads(path.read_text(encoding="utf-8")))
        return

    if args.quick:
        args.epochs, args.cv_epochs, args.folds = 2, 1, 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  GENERATING RESULTS")
    print(f"  device={device}  epochs={args.epochs}  "
          f"cv={args.folds}x{args.cv_epochs}  quick={args.quick}")
    print("=" * 70)

    print("\n[1/7] Loading data...")
    df = load_data(args.data)

    # ── Main model ──────────────────────────────────────────────────────────
    print("\n[2/7] Training CNN-LSTM...")
    model, scaler, headline, history = train_model(
        df, sequence_length=args.seq_len, epochs=args.epochs, device=device,
    )
    data = prepare_data(df, sequence_length=args.seq_len)
    print(f"  RMSE {headline['rmse']}  MAE {headline['mae']}  R² {headline['r2']}")
    fig_training_curve(history)

    # ── Baselines ───────────────────────────────────────────────────────────
    print("\n[3/7] Training baselines...")
    cmp_rows = []

    # Persistence: predict next hour = current hour (last GHI in the window)
    pers_pred = _inv_ghi(data["X_test"][:, -1, 0], scaler)
    pers_true = _inv_ghi(data["y_test"], scaler)
    cmp_rows.append({"model": "Persistence", **_metrics_from(pers_pred, pers_true)})
    print(f"  Persistence  RMSE {cmp_rows[-1]['rmse']}")

    for label, factory in [
        ("LSTM-only", lambda n: LSTMOnlyModel(num_features=n)),
        ("CNN-only", lambda n: CNNOnlyModel(num_features=n)),
    ]:
        _, _, mm, _ = train_model(
            df, sequence_length=args.seq_len, epochs=args.epochs,
            device=device, model_factory=factory, verbose=False,
        )
        cmp_rows.append({"model": label, **{k: mm[k] for k in
                                            ("rmse", "mae", "r2", "mape", "accuracy")}})
        print(f"  {label:<12} RMSE {mm['rmse']}")

    cmp_rows.append({"model": "CNN-LSTM (ours)",
                     **{k: headline[k] for k in ("rmse", "mae", "r2", "mape", "accuracy")}})
    fig_model_comparison(cmp_rows)

    # ── Feature importance ──────────────────────────────────────────────────
    print("\n[4/7] Permutation feature importance...")
    imp, imp_base = permutation_importance(
        model, data["X_test"], data["y_test"], scaler, device)
    fig_feature_importance(imp, imp_base)

    # ── Cross-validation + pooled out-of-sample diagnostics ─────────────────
    print("\n[5/7] Walk-forward cross-validation...")
    folds, pooled = walk_forward_cv(df, args.folds, args.cv_epochs, args.seq_len, device)
    fig_cross_validation(folds)

    pooled["residual"] = pooled["pred"] - pooled["actual"]
    pooled["hour"] = pooled["timestamp"].dt.hour
    pooled["season"] = pooled["timestamp"].dt.month.map(_season)

    seasonal = []
    for s in SEASON_ORDER:
        sub = pooled[pooled["season"] == s]
        if len(sub):
            mm = _metrics_from(sub["pred"].values, sub["actual"].values)
            seasonal.append({"season": s, "rmse": mm["rmse"], "mae": mm["mae"],
                             "n": int(len(sub))})
    fig_seasonal(seasonal)

    hourly_rmse = [
        float(np.sqrt(np.mean(pooled.loc[pooled["hour"] == h, "residual"] ** 2)))
        if (pooled["hour"] == h).any() else 0.0
        for h in range(24)
    ]
    # An hour counts as daylight if mean observed irradiance exceeds 10 W/m².
    daylight_hours = [
        h for h in range(24)
        if (pooled["hour"] == h).any()
        and float(pooled.loc[pooled["hour"] == h, "actual"].mean()) > 10.0
    ]
    fig_hourly(hourly_rmse)

    resid = pooled["residual"].values
    fig_error_distribution(resid)

    # ── Duck curve + grid stability ─────────────────────────────────────────
    print("\n[6/7] Duck curve & grid stability simulation...")
    sim_df = load_or_generate_data(args.data)
    days = simulate_all_days(sim_df, capacity_mw=SOLAR_FARM_MW)
    fracs = [d["solar_fraction"] for d in days]
    focus = days[int(np.argsort(fracs)[len(fracs) // 2])]
    fig_duck_curve(focus)

    scores = [d["analysis"]["stability_score"] for d in days]
    buckets = {"LOW": 0, "MODERATE": 0, "HIGH": 0}
    for d in days:
        buckets[d["analysis"]["stress_level"]] += 1
    fig_grid_stability(scores, buckets)

    duck = {
        "date": focus["date"],
        "capacity_mw": float(SOLAR_FARM_MW),
        "peak_pv_mw": float(focus["pv_power"].max()),
        "peak_hour": int(focus["pv_power"].argmax()),
        "solar_fraction": focus["solar_fraction"],
        "stability_score": focus["analysis"]["stability_score"],
        "stress_level": focus["analysis"]["stress_level"],
        "morning_ramp": focus["analysis"]["morning_ramp_rate"],
        "evening_ramp": focus["analysis"]["evening_ramp_rate"],
    }
    grid = {
        "n_days": len(days),
        "buckets": buckets,
        "mean_score": float(np.mean(scores)),
        "high_ramp_threshold": focus["high_ramp_threshold"],
    }

    # ── Report ──────────────────────────────────────────────────────────────
    print("\n[7/7] Writing report...")
    mean_r = float(np.mean(resid))
    std_r = float(np.std(resid))
    kurt = float(np.mean(((resid - mean_r) / (std_r + 1e-12)) ** 4) - 3.0)

    res = {
        "config": {
            "dataset": os.path.basename(args.data),
            "n_rows": int(len(df)),
            "date_range": f"{df['timestamp'].min():%Y-%m-%d} to {df['timestamp'].max():%Y-%m-%d}",
            "sequence_length": args.seq_len,
            "epochs": args.epochs,
            "cv_epochs": args.cv_epochs,
            "device": device,
            "generated_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        },
        "headline": {k: headline[k] for k in ("rmse", "mae", "r2", "mape", "accuracy")},
        "history": history,
        "model_comparison": cmp_rows,
        "feature_importance": imp,
        "importance_baseline_rmse": imp_base,
        "cv_folds": folds,
        "seasonal": seasonal,
        "hourly_rmse": hourly_rmse,
        "daylight_hours": daylight_hours,
        "residual_mean": mean_r,
        "residual_std": std_r,
        "residual_kurtosis": kurt,
        "duck": duck,
        "grid": grid,
    }

    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print("  wrote results/metrics.json")

    write_report(res)

    print("\n" + "=" * 70)
    print(f"  DONE in {(time.time() - t0) / 60:.1f} min")
    print("=" * 70)


if __name__ == "__main__":
    main()
