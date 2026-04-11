import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Solar constant (W/m²)
SOLAR_CONSTANT = 1361.0


def compute_extraterrestrial_radiation(day_of_year):
    """
    Compute extraterrestrial radiation on a horizontal surface (W/m²).

    Args:
        day_of_year: Integer or array of day-of-year values (1–365)

    Returns:
        Extraterrestrial radiation array (W/m²)
    """
    B = 2 * np.pi * (day_of_year - 1) / 365.0
    E0 = (1.000110 + 0.034221 * np.cos(B) + 0.001280 * np.sin(B)
          + 0.000719 * np.cos(2 * B) + 0.000077 * np.sin(2 * B))
    return SOLAR_CONSTANT * E0


def compute_solar_zenith_angle(timestamps, latitude=19.076):
    """
    Approximate solar zenith angle (degrees) for given timestamps.

    Args:
        timestamps: DatetimeIndex or Series of timestamps
        latitude: Observer latitude in degrees (default: Mumbai, IIT Bombay)

    Returns:
        Array of solar zenith angles in degrees
    """
    lat_rad = np.radians(latitude)

    ts_dt = pd.DatetimeIndex(timestamps)
    hour = ts_dt.hour + ts_dt.minute / 60.0
    day_of_year = ts_dt.dayofyear

    # Hour angle (solar noon = 0)
    hour_angle = np.radians(15.0 * (hour - 12.0))

    # Declination angle (approximate)
    declination = np.radians(23.45 * np.sin(
        np.radians(360.0 / 365.0 * (day_of_year - 81))
    ))

    # Zenith angle
    cos_zenith = (np.sin(lat_rad) * np.sin(declination)
                  + np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle))
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    zenith_angle = np.degrees(np.arccos(cos_zenith))

    return zenith_angle


def compute_clearness_index(ghi, day_of_year, zenith_angle):
    """
    Compute Clearness Index (Kt = GHI / Extraterrestrial Horizontal Radiation).

    Args:
        ghi: Array of GHI values (W/m²)
        day_of_year: Array of day-of-year values
        zenith_angle: Array of solar zenith angles (degrees)

    Returns:
        Array of clearness index values (0–1 range)
    """
    et_radiation = compute_extraterrestrial_radiation(day_of_year)
    cos_zenith = np.cos(np.radians(zenith_angle))
    # Horizontal extraterrestrial irradiance
    eth = et_radiation * np.maximum(cos_zenith, 0.0)
    # Avoid division by zero for night hours
    kt = np.where(eth > 10.0, ghi / eth, 0.0)
    return np.clip(kt, 0.0, 1.0)


def engineer_features(df):
    """
    Build feature matrix from a DataFrame with 'timestamp' and 'ghi' columns.

    Args:
        df: DataFrame with columns ['timestamp', 'ghi']

    Returns:
        DataFrame of engineered features
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    ts = df['timestamp']
    ghi = df['ghi'].values.astype(float)

    # Time features
    df['hour'] = ts.dt.hour
    df['day_of_year'] = ts.dt.dayofyear
    df['month'] = ts.dt.month

    # Cyclical encoding of hour to capture periodicity
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

    # Solar position features
    zenith = compute_solar_zenith_angle(ts)
    df['zenith_angle'] = zenith
    df['cos_zenith'] = np.cos(np.radians(zenith))

    # Clearness Index
    df['clearness_index'] = compute_clearness_index(
        ghi, df['day_of_year'].values, zenith
    )

    # Rolling statistics (3-hour and 6-hour windows)
    df['ghi_roll3_mean'] = df['ghi'].rolling(window=3, min_periods=1).mean()
    df['ghi_roll6_mean'] = df['ghi'].rolling(window=6, min_periods=1).mean()
    df['ghi_roll3_std'] = df['ghi'].rolling(window=3, min_periods=1).std().fillna(0)

    # GHI difference (rate of change)
    df['ghi_diff'] = df['ghi'].diff().fillna(0)

    return df


# Feature columns used for model input
FEATURE_COLUMNS = [
    'ghi',
    'hour_sin',
    'hour_cos',
    'cos_zenith',
    'clearness_index',
    'ghi_roll3_mean',
    'ghi_roll6_mean',
    'ghi_diff',
]


def normalize_features(df, feature_cols=None, scaler=None):
    """
    Normalize feature columns using MinMaxScaler.

    Args:
        df: DataFrame with engineered features
        feature_cols: List of column names to normalize (default: FEATURE_COLUMNS)
        scaler: Pre-fitted scaler (None to fit a new one)

    Returns:
        (normalized_array, scaler, feature_cols)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    X = df[feature_cols].values.astype(float)

    if scaler is None:
        scaler = MinMaxScaler()
        X_norm = scaler.fit_transform(X)
    else:
        X_norm = scaler.transform(X)

    return X_norm, scaler, feature_cols


def create_sequences(X, y, seq_len=24):
    """
    Create sliding-window sequences for time-series model input.

    Args:
        X: Feature array of shape (n_samples, n_features)
        y: Target array of shape (n_samples,)
        seq_len: Sequence (window) length

    Returns:
        (X_seq, y_seq) arrays suitable for model training
    """
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)
