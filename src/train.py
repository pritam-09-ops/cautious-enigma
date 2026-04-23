
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from model import CNNLSTMModel
from feature_engineering import (
    engineer_features,
    normalize_features,
    create_sequences,
    FEATURE_COLUMNS,
)


def load_data(filepath):
    """
    Load solar irradiance CSV data.

    The CSV must have at minimum:
        - 'timestamp': parseable datetime string
        - 'ghi': Global Horizontal Irradiance (W/m²)

    Missing GHI values are filled with linear interpolation.

    Args:
        filepath: Path to the CSV file

    Returns:
        Cleaned DataFrame with 'timestamp' and 'ghi' columns
    """
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Fill missing GHI values
    df['ghi'] = df['ghi'].interpolate(method='linear').fillna(0.0)

    # Clip negative values (physically impossible)
    df['ghi'] = df['ghi'].clip(lower=0.0)

    print(f"   ✓ Loaded {len(df):,} records from {filepath}")
    print(f"   ✓ Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"   ✓ GHI range : {df['ghi'].min():.1f} – {df['ghi'].max():.1f} W/m²")

    return df


def train_model(df, sequence_length=24, batch_size=32, epochs=50,
                learning_rate=0.001, test_split=0.2, device='cpu',
                save_path=None):
    """
    Full training pipeline.

    Args:
        df: DataFrame with 'timestamp' and 'ghi' columns
        sequence_length: Sliding window length (hours)
        batch_size: Mini-batch size
        epochs: Training epochs
        learning_rate: Adam optimizer learning rate
        test_split: Fraction of data for testing
        device: 'cpu' or 'cuda'
        save_path: Path to save model weights (None = don't save)

    Returns:
        (model, scaler, metrics_dict)
    """
    print("\n   Feature engineering...")
    df_feat = engineer_features(df)
    n_features = len(FEATURE_COLUMNS)

    X_norm, scaler, _ = normalize_features(df_feat)

    # GHI is the first column — use it as target
    y = df_feat['ghi'].values.astype(float)
    # Normalize target consistently with the feature scaler
    y_norm = X_norm[:, 0]

    X_seq, y_seq = create_sequences(X_norm, y_norm, seq_len=sequence_length)

    # Train/test split (chronological)
    split_idx = int(len(X_seq) * (1 - test_split))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    print(f"   ✓ Training samples : {len(X_train):,}")
    print(f"   ✓ Test samples     : {len(X_test):,}")
    print(f"   ✓ Feature count    : {n_features}")

    # Build PyTorch datasets
    train_ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
    )
    test_ds = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = CNNLSTMModel(
        input_channels=1,
        lstm_hidden=64,
        output_size=1,
        num_features=n_features,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    print(f"\n   Training CNN-LSTM for {epochs} epochs...")
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).squeeze(-1)
                val_loss += criterion(pred, yb).item() * len(xb)
        val_loss /= len(test_ds)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"   Epoch [{epoch:3d}/{epochs}] "
                  f"Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Evaluation ---
    metrics = evaluate_model(model, test_loader, scaler, device)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"\n   ✓ Model weights saved to {save_path}")

    return model, scaler, metrics


def evaluate_model(model, test_loader, scaler, device='cpu'):
    """
    Compute evaluation metrics on the test set.

    Returns:
        dict with RMSE, MAE, R², MAPE, accuracy
    """
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).squeeze(-1).cpu().numpy()
            all_preds.extend(pred)
            all_targets.extend(yb.numpy())

    preds_norm = np.array(all_preds)
    targets_norm = np.array(all_targets)

    # Inverse transform (GHI is feature index 0)
    def inv(vals):
        dummy = np.zeros((len(vals), scaler.n_features_in_))
        dummy[:, 0] = vals
        return np.maximum(scaler.inverse_transform(dummy)[:, 0], 0.0)

    preds = inv(preds_norm)
    targets = inv(targets_norm)

    rmse = float(np.sqrt(mean_squared_error(targets, preds)))
    mae = float(mean_absolute_error(targets, preds))
    r2 = float(r2_score(targets, preds))

    # MAPE — avoid division by zero
    mask = targets > 1.0
    mape = (
        float(np.mean(np.abs((targets[mask] - preds[mask]) / targets[mask])) * 100)
        if mask.any() else float('nan')
    )

    accuracy = max(0.0, (1.0 - mae / (targets.mean() + 1e-8)) * 100)

    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2": round(r2, 4),
        "mape": round(mape, 2),
        "accuracy": round(accuracy, 1),
    }


def print_metrics(metrics):
    """Pretty-print evaluation metrics with interpretation."""
    rmse = metrics['rmse']
    mae  = metrics['mae']
    r2   = metrics['r2']
    mape = metrics['mape']
    acc  = metrics['accuracy']

    # Qualitative rating bands (literature benchmarks for GHI forecasting)
    def rmse_rating(v):
        return "Excellent" if v < 40 else ("Good" if v < 80 else ("Acceptable" if v < 130 else "Needs improvement"))

    def r2_rating(v):
        return "Excellent" if v > 0.95 else ("Good" if v > 0.90 else ("Acceptable" if v > 0.80 else "Needs improvement"))

    def mape_rating(v):
        return "Excellent" if v < 5 else ("Good" if v < 10 else ("Acceptable" if v < 20 else "Needs improvement"))

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION — TEST SET PERFORMANCE")
    print("=" * 60)
    print(f"  {'Metric':<22} {'Value':>12}   {'Rating'}")
    print("  " + "-" * 54)
    print(f"  {'RMSE':<22} {rmse:>10.2f}   W/m²  →  {rmse_rating(rmse)}")
    print(f"  {'MAE':<22} {mae:>10.2f}   W/m²  →  (mean abs error)")
    print(f"  {'R²  (coeff. of det.)':<22} {r2:>12.4f}   {r2_rating(r2)}")
    print(f"  {'MAPE':<22} {mape:>10.2f}   %     →  {mape_rating(mape)}")
    print(f"  {'Accuracy (1−MAE/μ)':<22} {acc:>10.1f}   %")
    print("=" * 60)
    print("  Benchmark: state-of-the-art CNN-LSTM models for")
    print("  hourly GHI forecasting typically achieve RMSE 30–80 W/m²")
    print("  and R² > 0.92 on clear-sky-normalised datasets.")
    print("=" * 60)
