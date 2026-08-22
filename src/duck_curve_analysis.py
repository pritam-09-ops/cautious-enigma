"""
duck_curve_analysis.py
=======================
Core duck-curve / grid-stability analysis used by the CNN-LSTM pipeline
(main.py, train.py, predict.py).

For the standalone multi-day plotting & PV-power simulation, see
duck_curve_simulation.py — it imports these functions rather than
duplicating them.
"""

import numpy as np

import console_utils  # noqa: F401  (configures UTF-8 console output on import)

# Default ramp-rate thresholds, calibrated for GHI (W/m^2) input.
# Callers working in other units (e.g. MW of PV power) should pass an
# explicit threshold scaled to their own value range.
HIGH_RAMP_THRESHOLD = 150.0
CURTAILMENT_RAMP_THRESHOLD = 100.0


def analyze_duck_curve(predictions, timestamps=None, high_ramp_threshold=HIGH_RAMP_THRESHOLD):
    """
    Analyze a 24-hour generation profile for duck-curve / ramp-rate behaviour.

    Args:
        predictions: Sequence of hourly generation values (GHI in W/m^2, or
            PV power in MW — any consistent unit).
        timestamps: Unused placeholder for future timestamp-aware analysis.
        high_ramp_threshold: Ramp rate (per hour, same unit as predictions)
            above which an hour is flagged as a high-ramp event.

    Returns:
        dict with ramp rates, peak info, and a 0-100 stability score.
    """
    arr = np.asarray(predictions, dtype=float)
    ramp_rates = np.diff(arr, prepend=arr[0])
    mid = len(arr) // 2

    stability = max(0.0, 100.0 - (np.max(np.abs(ramp_rates)) / high_ramp_threshold) * 30)
    stress = "LOW" if stability >= 80 else ("MODERATE" if stability >= 60 else "HIGH")

    return dict(
        morning_ramp_rate=float(ramp_rates[:mid].max()) if mid else 0.0,
        evening_ramp_rate=float(ramp_rates[mid:].min()) if mid < len(arr) else 0.0,
        peak_generation=float(arr.max()),
        peak_hour=int(arr.argmax()),
        high_ramp_events=np.where(np.abs(ramp_rates) > high_ramp_threshold)[0].tolist(),
        stability_score=round(stability, 1),
        stress_level=stress,
        ramp_rates=ramp_rates.tolist(),
        high_ramp_threshold=high_ramp_threshold,
    )


def predict_curtailment_strategy(analysis, predictions,
                                  high_threshold=None, low_threshold=CURTAILMENT_RAMP_THRESHOLD):
    """
    Derive an hourly curtailment schedule from a duck-curve analysis.

    Args:
        analysis: Output of analyze_duck_curve().
        predictions: The same generation profile passed to analyze_duck_curve().
        high_threshold: Ramp rate above which action is "CURTAIL". Defaults
            to the threshold actually used by analyze_duck_curve().
        low_threshold: Ramp rate above which action is "REDUCE".

    Returns:
        dict with the per-hour curtailment schedule and an overall recommendation.
    """
    if high_threshold is None:
        high_threshold = analysis.get("high_ramp_threshold", HIGH_RAMP_THRESHOLD)

    rr = np.asarray(analysis["ramp_rates"])
    schedule = []
    for i, (p, r) in enumerate(zip(predictions, rr)):
        ar = abs(r)
        if ar > high_threshold:
            action = "CURTAIL"
            pct = min(40.0, (ar / high_threshold - 1) * 20 + 20)
        elif ar > low_threshold:
            action = "REDUCE"
            pct = min(20.0, (ar / low_threshold - 1) * 10 + 5)
        else:
            action = "NORMAL"
            pct = 0.0
        schedule.append(dict(hour=i, ghi_forecast=round(float(p), 2),
                              ramp_rate=round(float(r), 2),
                              action=action, curtailment_pct=round(pct, 1)))

    s = analysis["stability_score"]
    rec = ("Full dispatch recommended — grid is stable." if s >= 80
           else "Moderate curtailment advised during high-ramp windows." if s >= 60
           else "Significant curtailment required. Coordinate with grid operator.")

    return dict(curtailment_schedule=schedule, dispatch_recommendation=rec,
                high_ramp_hours=analysis["high_ramp_events"],
                stability_score=s, stress_level=analysis["stress_level"])


def print_duck_curve_summary(analysis, curtailment, unit="W/m²"):
    """Pretty-print a duck-curve / curtailment summary. `unit` labels the ramp-rate axis."""
    W = 60
    print("\n" + "=" * W)
    print("  DUCK CURVE & GRID STABILITY SUMMARY")
    print("=" * W)
    print(f"  Morning ramp rate  : {analysis['morning_ramp_rate']:+.1f} {unit}/h")
    print(f"  Evening ramp rate  : {analysis['evening_ramp_rate']:+.1f} {unit}/h")
    print(f"  Peak generation    : {analysis['peak_generation']:.1f} {unit}"
          f"  at {analysis['peak_hour']:02d}:00")
    print(f"  Stability score    : {analysis['stability_score']}/100"
          f"  [{analysis['stress_level']} STRESS]")
    if analysis["high_ramp_events"]:
        hrs = ", ".join(f"{h:02d}:00" for h in analysis["high_ramp_events"])
        print(f"  High-ramp hours    : {hrs}")
    else:
        print("  High-ramp hours    : None detected ✓")
    print(f"\n  Dispatch advisory  : {curtailment['dispatch_recommendation']}")
    curtailed = [s for s in curtailment["curtailment_schedule"]
                 if s["action"] != "NORMAL"]
    if curtailed:
        print(f"  Curtailment slots  : {len(curtailed)} h requiring REDUCE/CURTAIL action")
    else:
        print("  Curtailment slots  : None required ✓")
    print("=" * W)
