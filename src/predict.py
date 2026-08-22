import numpy as np
import torch
import pandas as pd

from feature_engineering import (
    compute_solar_zenith_angle,
    compute_clearness_index,
)
from duck_curve_analysis import analyze_duck_curve, predict_curtailment_strategy


def _inverse_ghi(norm_vals, scaler):
    """Inverse-transform normalized GHI values (feature index 0) back to W/m²."""
    norm_vals = np.atleast_1d(norm_vals)
    dummy = np.zeros((len(norm_vals), scaler.n_features_in_))
    dummy[:, 0] = norm_vals
    return np.maximum(scaler.inverse_transform(dummy)[:, 0], 0.0)


def forecast_24h(model, last_sequence, scaler, last_timestamp, n_steps=24,
                 device='cpu', mc_samples=50, latitude=19.076):
    """
    Generate 24-hour ahead GHI forecasts with confidence intervals.

    Uses Monte Carlo dropout for uncertainty estimation. At each recursive
    step the full feature vector (not just GHI) is rebuilt for the
    predicted hour — cyclical time encoding, solar zenith, clearness index,
    and rolling GHI statistics — so the model actually sees the correct
    time-of-day signal as the forecast advances instead of a frozen copy
    of the seed window's last hour.

    Args:
        model: Trained CNNLSTMModel
        last_sequence: Seed sequence array of shape (seq_len, n_features)
        scaler: Fitted MinMaxScaler for inverse transform
        last_timestamp: Timestamp of the seed sequence's final (most recent)
            hour — forecasts start at last_timestamp + 1h.
        n_steps: Forecast horizon (default 24 hours)
        device: Torch device string
        mc_samples: Number of MC dropout passes for uncertainty
        latitude: Observer latitude in degrees, for solar position features

    Returns:
        dict with 'predictions', 'lower_ci', 'upper_ci' (all W/m²)
    """
    model.train()  # Keep dropout active for MC estimation
    all_preds = []

    last_timestamp = pd.Timestamp(last_timestamp)
    seed_ghi_raw = list(_inverse_ghi(last_sequence[-6:, 0], scaler))

    for _ in range(mc_samples):
        preds_run = []
        current_seq = last_sequence.copy()
        ghi_hist = list(seed_ghi_raw)  # rolling raw-GHI history for this MC run

        for step in range(n_steps):
            x = torch.FloatTensor(current_seq).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_norm = model(x).cpu().numpy().flatten()[0]
            preds_run.append(pred_norm)

            pred_raw = float(_inverse_ghi(pred_norm, scaler)[0])
            ghi_hist.append(pred_raw)

            future_time = last_timestamp + pd.Timedelta(hours=step + 1)
            hour = future_time.hour + future_time.minute / 60.0
            doy = future_time.dayofyear
            hour_sin = np.sin(2 * np.pi * hour / 24.0)
            hour_cos = np.cos(2 * np.pi * hour / 24.0)
            zenith = compute_solar_zenith_angle(
                pd.DatetimeIndex([future_time]), latitude=latitude
            )[0]
            cos_zenith = np.cos(np.radians(zenith))
            clearness = compute_clearness_index(
                np.array([pred_raw]), np.array([doy]), np.array([zenith])
            )[0]

            roll3 = float(np.mean(ghi_hist[-3:]))
            roll6 = float(np.mean(ghi_hist[-6:]))
            ghi_diff = ghi_hist[-1] - ghi_hist[-2] if len(ghi_hist) > 1 else 0.0

            raw_feat_vec = np.array([[pred_raw, hour_sin, hour_cos, cos_zenith,
                                       clearness, roll3, roll6, ghi_diff]])
            next_step = scaler.transform(raw_feat_vec)[0]

            # Build next step: shift window and append the recomputed features
            current_seq = np.vstack([current_seq[1:], next_step])

        all_preds.append(preds_run)

    all_preds = np.array(all_preds)  # (mc_samples, n_steps)

    mean_pred_norm = all_preds.mean(axis=0)
    std_pred_norm = all_preds.std(axis=0)

    predictions = _inverse_ghi(mean_pred_norm, scaler)
    lower_ci = _inverse_ghi(np.maximum(mean_pred_norm - 1.96 * std_pred_norm, 0.0), scaler)
    upper_ci = _inverse_ghi(mean_pred_norm + 1.96 * std_pred_norm, scaler)

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
    """Pretty-print the 24-hour operational forecast with grid advisories."""
    preds = forecast_result["predictions"]
    lower = forecast_result["lower_ci"]
    upper = forecast_result["upper_ci"]

    def _period(h):
        if h < 6:   return "Night"
        if h < 10:  return "Morning ramp"
        if h < 15:  return "Midday peak"
        if h < 19:  return "Afternoon"
        if h < 22:  return "Evening ramp"
        return "Night"

    def _advisory(s):
        if s["recommended_dispatch_pct"] == 100:
            return "Full dispatch"
        if s["recommended_dispatch_pct"] >= 90:
            return "Slight derating"
        return "Curtail — high uncertainty"

    peak_val  = max(preds)
    peak_hour = int(preds.index(peak_val)) if isinstance(preds, list) else int(
        np.argmax(preds))
    daily_kwh = sum(preds)

    print("\n" + "=" * 78)
    print("  24-HOUR GHI OPERATIONAL FORECAST")
    print("=" * 78)
    print(f"  {'Hr':>3} | {'Period':<15} | {'GHI W/m²':>10} | "
          f"{'95% CI':>22} | {'Dispatch':>9} | Advisory")
    print("  " + "-" * 72)
    for s in schedule:
        h  = s["hour"]
        ci = f"[{lower[h]:>7.1f}, {upper[h]:>7.1f}]"
        print(f"  {h:02d}:00 | {_period(h):<15} | {preds[h]:>9.1f}   | "
              f"{ci:>22} | {s['recommended_dispatch_pct']:>7}%  | {_advisory(s)}")
    print("=" * 78)
    print(f"  Peak forecast  : {peak_val:.1f} W/m²  at {peak_hour:02d}:00")
    print(f"  Daily integral : {daily_kwh:.0f} Wh/m²  "
          f"({daily_kwh / 1000:.2f} kWh/m²/day)")
    low_conf = sum(1 for s in schedule if s["confidence"] == "LOW")
    if low_conf:
        print(f"  ⚠  Low-confidence windows: {low_conf} h — consider reserve dispatch")
    else:
        print("  ✓  All forecast windows meet HIGH or MEDIUM confidence threshold")
    print("=" * 78)
