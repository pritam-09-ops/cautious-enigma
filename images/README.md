# Images Directory

Figures for [`RESULTS_AND_GRAPHS.md`](../RESULTS_AND_GRAPHS.md). **These are
generated artifacts — do not edit them by hand.** Regenerate the whole set with:

```bash
python src/generate_results.py
```

| File | Figure |
|------|--------|
| `01_training_curve.png` | Training vs validation loss per epoch |
| `02_model_comparison.png` | CNN-LSTM vs LSTM-only, CNN-only, persistence |
| `03_duck_curve.png` | Demand, PV generation, and net demand over a day |
| `04_feature_importance.png` | Permutation importance per input feature |
| `05_seasonal_performance.png` | Out-of-sample RMSE by season |
| `06_forecast_accuracy_by_hour.png` | Out-of-sample RMSE by hour of day |
| `07_error_distribution.png` | Residual histogram with mean bias |
| `08_cross_validation.png` | Walk-forward fold RMSE |
| `09_grid_stability.png` | Stability score trend and stress-level split |
