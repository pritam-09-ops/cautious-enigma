"""
24-Hour GHI Forecasting Module

Provides:
- 24-hour ahead GHI predictions
- Confidence interval estimation
- Optimal dispatch scheduling
- Grid stability assessment

Research Project: IIT Bombay (December 2025 - February 2026)
"""

import numpy as np
import torch
import pandas as pd

from feature_engineering import (
    engineer_features,
    normalize_features,
    create_sequences,
    FEATURE_COLUMNS,
)
from duck_curve_analysis import analyze_duck_curve, predict_curtailment_strategy


def forecast_24h(model, last_sequence, scaler, n_steps=24, device='cpu',
                 mc_samples=50):
    """
    Generate 24-hour ahead GHI forecasts with confidence intervals.

    Uses Monte Carlo dropout for uncertainty estimation.

    Args:
        model: Trained CNNLSTMModel
        last_sequence: Seed sequence array of shape (seq_len, n_features)
        scaler: Fitted MinMaxScaler for inverse transform
        n_steps: Forecast horizon (default 24 hours)
        device: Torch device string
        mc_samples: Number of MC dropout passes for uncertainty

    Returns:
        dict with 'predictions', 'lower_ci', 'upper_ci' (all W/m²)
    """
    model.train()  # Keep dropout active for MC estimation
    all_preds = []

    seq = last_sequence.copy()

    for _ in range(mc_samples):
        preds_run = []
        current_seq = seq.copy()
        for _ in range(n_steps):
            x = torch.FloatTensor(current_seq).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_norm = model(x).cpu().numpy().flatten()[0]
            preds_run.append(pred_norm)

            # Build next step: shift window and append predicted GHI
            next_step = current_seq[-1].copy()
            next_step[0] = pred_norm  # GHI is the first feature
            current_seq = np.vstack([current_seq[1:], next_step])

        all_preds.append(preds_run)

    all_preds = np.array(all_preds)  # (mc_samples, n_steps)

    mean_pred_norm = all_preds.mean(axis=0)
    std_pred_norm = all_preds.std(axis=0)

    # Inverse-transform: create dummy array with correct feature count
    def inverse_ghi(norm_vals):
        dummy = np.zeros((len(norm_vals), scaler.n_features_in_))
        dummy[:, 0] = norm_vals
        inv = scaler.inverse_transform(dummy)
        return np.maximum(inv[:, 0], 0.0)

    predictions = inverse_ghi(mean_pred_norm)
    lower_ci = inverse_ghi(np.maximum(mean_pred_norm - 1.96 * std_pred_norm, 0.0))
    upper_ci = inverse_ghi(mean_pred_norm + 1.96 * std_pred_norm)

    model.eval()
    return {
        "predictions": predictions.tolist(),
        "lower_ci": lower_ci.tolist(),
        "upper_ci": upper_ci.tolist(),
    }


def build_dispatch_schedule(forecast_result):
    """
    Build an optimal hourly dispatch schedule from the 24-h forecast.

    Args:
        forecast_result: Output from forecast_24h

    Returns:
        List of dicts with hour, forecast_ghi, recommended_dispatch
    """
    preds = forecast_result["predictions"]
    lower = forecast_result["lower_ci"]
    upper = forecast_result["upper_ci"]

    schedule = []
    for h, (p, lo, hi) in enumerate(zip(preds, lower, upper)):
        uncertainty = hi - lo
        if uncertainty > 200:
            confidence = "LOW"
            dispatch_fraction = 0.80
        elif uncertainty > 100:
            confidence = "MEDIUM"
            dispatch_fraction = 0.90
        else:
            confidence = "HIGH"
            dispatch_fraction = 1.00

        schedule.append({
            "hour": h,
            "forecast_ghi": round(float(p), 1),
            "lower_ci": round(float(lo), 1),
            "upper_ci": round(float(hi), 1),
            "confidence": confidence,
            "recommended_dispatch_pct": int(dispatch_fraction * 100),
        })
    return schedule


def assess_grid_stability(forecast_result):
    """
    Perform high-level grid stability assessment from 24-h forecast.

    Args:
        forecast_result: Output from forecast_24h

    Returns:
        dict with stability metrics
    """
    preds = np.array(forecast_result["predictions"])
    analysis = analyze_duck_curve(preds)
    curtailment = predict_curtailment_strategy(analysis, preds)

    return {
        "duck_curve_analysis": analysis,
        "curtailment_strategy": curtailment,
    }


def print_forecast_summary(forecast_result, schedule):
    """Pretty-print the 24-hour forecast results."""
    preds = forecast_result["predictions"]
    lower = forecast_result["lower_ci"]
    upper = forecast_result["upper_ci"]

    print("\n" + "=" * 70)
    print("24-HOUR GHI FORECAST")
    print("=" * 70)
    print(f"  {'Hour':>4} | {'GHI (W/m²)':>11} | {'95% CI':>20} | {'Dispatch':>8}")
    print("  " + "-" * 56)
    for s in schedule:
        h = s["hour"]
        print(f"  {h:02d}:00 | {preds[h]:>9.1f}   | "
              f"[{lower[h]:>7.1f}, {upper[h]:>7.1f}] | "
              f"{s['recommended_dispatch_pct']:>6}%")
    print("=" * 70)
    print(f"  Peak forecast: {max(preds):.1f} W/m² at "
          f"hour {int(np.argmax(preds)):02d}:00")
    print(f"  Daily total  : {sum(preds):.0f} Wh/m²")
    print("=" * 70)
