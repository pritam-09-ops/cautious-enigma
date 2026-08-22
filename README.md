# Solar Irradiance & PV Power Prediction

Hourly global horizontal irradiance (GHI) forecasting with a CNN-LSTM hybrid,
plus duck-curve and grid-stability analysis for a utility-scale PV farm.

## Project Overview
This project predicts solar irradiance and photovoltaic (PV) power generation
using a hybrid CNN-LSTM sequence model with physics-informed features
(clearness index, solar zenith angle). It also simulates the resulting
duck-curve dynamics and ramp-driven curtailment needs on a Mumbai-shaped
demand profile.

## Research Details
**Institution**: IIT Bombay
**Duration**: December 2025 - February 2026
**Architecture**: CNN-LSTM hybrid, 24-hour look-back window

## Current Results

Measured on the bundled **synthetic** dataset (8,760 hourly rows, 40 epochs) —
see [RESULTS_AND_GRAPHS.md](RESULTS_AND_GRAPHS.md) for all nine figures and
[`results/metrics.json`](results/metrics.json) for raw values.

| Model | RMSE (W/m²) | MAE (W/m²) | R² |
|-------|-------------|------------|-----|
| Persistence baseline | 133.8 | 76.4 | 0.640 |
| LSTM-only | 80.7 | 43.8 | 0.869 |
| CNN-only | 80.1 | 42.1 | 0.871 |
| CNN-LSTM | 80.1 | 44.9 | 0.871 |

The learned models cut RMSE ~40% versus persistence. **They do not
meaningfully differ from each other** — the three land within 0.6 W/m², well
inside cross-validation fold variance, so the hybrid shows no measurable
advantage over its single-branch ablations on this data. Reproduce with:

```bash
python src/generate_results.py
```

> **On the data:** the bundled dataset is synthetic — a Spencer/Iqbal
> clear-sky model with stochastic monsoon-aware cloud cover, calibrated to
> Mumbai (lat 19.076°), so the pipeline runs out of the box. These numbers
> characterise model behaviour on that series, not measured-irradiance
> benchmark performance. Point `--data` at real observations to evaluate
> properly.

## Feature Engineering
### Clearness Index (Kt)
The clearness index quantifies the fraction of global solar radiation received at Earth's surface compared to the total solar radiation at the top of the atmosphere:
```
Kt = GHI / Extraterrestrial Radiation
```
This parameter helps understand atmospheric transparency and its impact on solar generation.

### Solar Zenith Angle
The solar zenith angle is the angle between the sun and a point on Earth's surface, varying throughout the day and affecting solar radiation intensity. Accurate modeling of this angle enhances prediction performance.

## Duck Curve Analysis
The duck curve illustrates daily variation in electricity demand alongside solar energy output. Our analysis includes:
- **Morning Ramp Analysis**: Peak generation period identification
- **Evening Ramp Analysis**: Rapid decline in solar generation  
- **Peak Hour Prediction**: Optimal dispatch scheduling
- **Proactive Grid Management**: Mitigating midday grid instability

## Curtailment Strategies
To balance supply and demand during peak solar output:
- **Predictive Curtailment**: Reduce generation by forecast-driven percentages
- **Ramp Management**: Proactive strategies to prevent grid instability
- **Stability Scoring**: Real-time grid stability assessment
- **High Ramp Event Detection**: Early warning system for critical events

## Installation Instructions
```bash
# Clone the repository
git clone https://github.com/pritam-09-ops/cautious-enigma.git
cd cautious-enigma

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Run the complete pipeline (trains a model, forecasts, and analyzes grid stability)
python src/main.py

# Multi-day duck-curve simulation with 6-panel plot (standalone, no torch needed)
python src/duck_curve_simulation.py --date 2025-06-15 --save plots/duck_curve.png
```

## Project Structure
```
cautious-enigma/
├── src/
│   ├── main.py                  # Pipeline orchestration
│   ├── model.py                 # CNN-LSTM plus LSTM-only / CNN-only ablations
│   ├── feature_engineering.py   # Clearness Index, Zenith Angle & normalization
│   ├── train.py                 # Data prep, training loop & evaluation metrics
│   ├── predict.py               # 24-hour forecasting with MC-dropout uncertainty
│   ├── duck_curve_analysis.py   # Core grid-stability / ramp-rate analysis
│   ├── duck_curve_simulation.py # Multi-day PV/demand simulation & plotting
│   ├── generate_results.py      # Reproduces every figure & number in the report
│   └── console_utils.py         # UTF-8 safe console output
├── data/
│   └── sample_solar_data.csv    # Synthetic hourly GHI data (Mumbai-calibrated)
├── images/                      # Generated figures (do not edit by hand)
├── results/metrics.json         # Generated metrics
├── requirements.txt
└── README.md
```

## Key Findings
- **~40% RMSE reduction** versus a persistence baseline (133.8 → 80.1 W/m²)
- **Hybrid gives no measurable gain** over CNN-only or LSTM-only ablations here
- **Solar geometry dominates**: `hour_sin` and `cos_zenith` are by far the
  highest-impact features under permutation importance
- **Accuracy improves with history**: walk-forward CV RMSE falls 108.9 → 77.6
  as the training window expands
- **Forecast horizon**: 24 hours ahead, with Monte-Carlo dropout confidence bands

## Author
Pritam-09-ops  
IIT Bombay Research Project