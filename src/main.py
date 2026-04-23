
import os
import sys
import argparse
import datetime
import numpy as np
import pandas as pd
import torch

from model import CNNLSTMModel
from feature_engineering import (
    engineer_features,
    normalize_features,
    create_sequences,
    FEATURE_COLUMNS,
)
from duck_curve_analysis import (
    analyze_duck_curve,
    predict_curtailment_strategy,
    print_duck_curve_summary,
)
from train import load_data, train_model, print_metrics
from predict import forecast_24h, build_dispatch_schedule, print_forecast_summary


# Default sample data path (relative to repo root)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(_REPO_ROOT, "data", "sample_solar_data.csv")


def generate_sample_data(filepath, n_days=365):
    """
    Generate a realistic synthetic GHI dataset calibrated to Mumbai (IIT Bombay).

    Uses the Spencer/Iqbal clear-sky model with:
      • Seasonal cloud-cover modulation (heavy monsoon Jun–Sep)
      • Stochastic Kt drawn from a climatology-consistent normal distribution
      • Night-time zeroing based on solar zenith angle
    """
    print(f"   Generating {n_days}-day synthetic dataset → {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    lat = 19.076          # IIT Bombay latitude
    solar_const = 1361.0  # W/m²

    timestamps = pd.date_range(start="2025-01-01", periods=n_days * 24, freq="h")
    ts = timestamps
    hour   = ts.hour
    doy    = ts.dayofyear
    month  = ts.month

    lat_r  = np.radians(lat)
    ha     = np.radians(15.0 * (hour - 12.0))
    decl   = np.radians(23.45 * np.sin(np.radians(360.0 / 365.0 * (doy - 81))))
    cos_z  = np.maximum(
        np.sin(lat_r) * np.sin(decl) + np.cos(lat_r) * np.cos(decl) * np.cos(ha),
        0.0,
    )

    # Spencer eccentricity correction → extraterrestrial irradiance
    B  = 2 * np.pi * (doy - 1) / 365.0
    E0 = (1.000110 + 0.034221 * np.cos(B) + 0.001280 * np.sin(B)
          + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))
    et_rad = solar_const * E0
    clear_sky = et_rad * cos_z * 0.78   # 78 % clear-sky transmittance

    # Clearness index: monsoon (Jun–Sep) heavily overcast, dry season clear
    rng = np.random.default_rng(42)
    is_monsoon = (month >= 6) & (month <= 9)
    kt_mu  = np.where(is_monsoon, 0.45, 0.72)
    kt_sig = np.where(is_monsoon, 0.18, 0.12)
    kt     = np.clip(rng.normal(kt_mu, kt_sig), 0.05, 1.0)

    # Night-time: zero out below threshold zenith
    kt  = np.where(cos_z < 0.05, 0.0, kt)
    ghi = np.round(np.maximum(clear_sky * kt, 0.0), 2)

    df_out = pd.DataFrame({"timestamp": timestamps, "ghi": ghi})
    df_out.to_csv(filepath, index=False)
    print(f"   ✓ Synthetic dataset saved  ({len(df_out):,} hourly rows, "
          f"GHI range {ghi[ghi>0].min():.1f}–{ghi.max():.1f} W/m²)")


def main():
    parser = argparse.ArgumentParser(
        description="Solar Irradiance & PV Power Prediction — CNN-LSTM Pipeline"
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help="Path to CSV file with 'timestamp' and 'ghi' columns",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Training epochs (default: 50)"
    )
    parser.add_argument(
        "--save-model",
        default=os.path.join(_REPO_ROOT, "models", "cnn_lstm.pt"),
        help="Path to save trained model weights",
    )
    args = parser.parse_args()

    run_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    print("=" * 70)
    print("  Solar Irradiance & PV Power Prediction System")
    print("  Architecture : CNN-LSTM Hybrid Deep Learning")
    print("  Institution  : IIT Bombay — Energy Systems Lab")
    print("  Run ID       :", run_id)
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  ✓ Compute device : {device.upper()}")
    print(f"  ✓ PyTorch version: {torch.__version__}")

    # ------------------------------------------------------------------ #
    # Step 1 — Data loading                                               #
    # ------------------------------------------------------------------ #
    print("\n[1/7] Loading data...")
    data_path = args.data
    if not os.path.exists(data_path):
        print(f"   Data file not found at {data_path}.")
        generate_sample_data(data_path)
    df = load_data(data_path)

    # ------------------------------------------------------------------ #
    # Step 2 — Feature engineering                                        #
    # ------------------------------------------------------------------ #
    print("\n[2/7] Feature Engineering...")
    df_feat = engineer_features(df)
    print("   ✓ Clearness Index (Kt)     — atmospheric transparency (0–1)")
    print("   ✓ Solar Zenith Angle       — sun position & air-mass correction")
    print("   ✓ Rolling Statistics       — 3 h and 6 h lagged moving averages")
    print("   ✓ Cyclical time encoding   — hour sin/cos (preserves periodicity)")
    print(f"   ✓ Feature matrix shape    — {len(df_feat):,} rows × {len(FEATURE_COLUMNS)} features")

    # ------------------------------------------------------------------ #
    # Step 3 — Data processing                                            #
    # ------------------------------------------------------------------ #
    print("\n[3/7] Data Processing Pipeline...")
    sequence_length = 24
    batch_size = 32
    n_features = len(FEATURE_COLUMNS)
    print(f"   ✓ Sliding window length : {sequence_length} h (look-back horizon)")
    print(f"   ✓ Mini-batch size       : {batch_size} samples")
    print(f"   ✓ Input feature count   : {n_features}")
    print("   ✓ Scaling method        : MinMaxScaler → [0, 1]")
    print("   ✓ Gap handling          : linear interpolation, then clip to 0")

    # ------------------------------------------------------------------ #
    # Step 4 — Model training                                             #
    # ------------------------------------------------------------------ #
    print("\n[4/7] Training CNN-LSTM Model...")
    print(f"   ✓ Epochs         : {args.epochs}")
    print("   ✓ Optimizer      : Adam  (lr=1e-3, betas=(0.9, 0.999))")
    print("   ✓ Loss function  : MSELoss")
    print("   ✓ Regularisation : Dropout 0.2 + gradient clipping (max_norm=1.0)")
    print("   ✓ LR schedule    : ReduceLROnPlateau (patience=5, factor=0.5)")
    print("   ✓ Early stopping : Best checkpoint retained via val-loss tracking")

    model, scaler, metrics = train_model(
        df,
        sequence_length=sequence_length,
        batch_size=batch_size,
        epochs=args.epochs,
        learning_rate=0.001,
        test_split=0.2,
        device=device,
        save_path=args.save_model,
    )

    # ------------------------------------------------------------------ #
    # Step 5 — Evaluation                                                 #
    # ------------------------------------------------------------------ #
    print("\n[5/7] Model Evaluation...")
    print_metrics(metrics)

    # ------------------------------------------------------------------ #
    # Step 6 — Duck Curve analysis                                        #
    # ------------------------------------------------------------------ #
    print("\n[6/7] Duck Curve & Grid Stability Analysis...")

    # Use last 24 hours of the dataset as the representative profile
    X_norm, _, _ = normalize_features(df_feat, scaler=scaler)
    last_seq = X_norm[-sequence_length:]
    x_tensor = torch.FloatTensor(last_seq).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        sample_pred_norm = model(x_tensor).cpu().numpy().flatten()

    # Build a toy 24-h profile from repeated inference on sliding recent data
    representative_24h = []
    window = X_norm[-sequence_length:].copy()
    for _ in range(24):
        xt = torch.FloatTensor(window).unsqueeze(0).to(device)
        with torch.no_grad():
            p = model(xt).cpu().numpy().flatten()[0]
        representative_24h.append(p)
        next_step = window[-1].copy()
        next_step[0] = p
        window = np.vstack([window[1:], next_step])

    # Inverse transform
    dummy = np.zeros((24, scaler.n_features_in_))
    dummy[:, 0] = representative_24h
    representative_ghi = np.maximum(scaler.inverse_transform(dummy)[:, 0], 0.0)

    analysis = analyze_duck_curve(representative_ghi)
    curtailment = predict_curtailment_strategy(analysis, representative_ghi)
    print_duck_curve_summary(analysis, curtailment)

    # ------------------------------------------------------------------ #
    # Step 7 — 24-hour forecast                                           #
    # ------------------------------------------------------------------ #
    print("\n[7/7] Generating 24-Hour Forecast...")
    seed_sequence = X_norm[-sequence_length:]
    forecast = forecast_24h(
        model, seed_sequence, scaler, n_steps=24, device=device, mc_samples=30
    )
    schedule = build_dispatch_schedule(forecast)
    print_forecast_summary(forecast, schedule)

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Run ID   : {run_id}")
    print(f"  RMSE     : {metrics['rmse']:.2f} W/m²")
    print(f"  MAE      : {metrics['mae']:.2f} W/m²")
    print(f"  R²       : {metrics['r2']:.4f}")
    print(f"  MAPE     : {metrics['mape']:.2f}%")
    print(f"  Accuracy : {metrics['accuracy']:.1f}%")
    print(f"  Stability: {analysis['stability_score']}/100  [{analysis['stress_level']} GRID STRESS]")
    print("=" * 70)
    print("\n  Key Research Contributions:")
    print("   • CNN-LSTM hybrid architecture for multi-step GHI forecasting")
    print("   • Clearness Index (Kt) for physics-informed atmospheric modeling")
    print("   • Solar Zenith Angle encoding for air-mass & path-length effects")
    print("   • Duck Curve dynamics analysis for grid stability assessment")
    print("   • Predictive curtailment with Monte Carlo confidence intervals")


if __name__ == "__main__":
    main()
