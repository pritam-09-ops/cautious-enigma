import numpy as np
import pandas as pd


# Thresholds (W/m² per hour)
HIGH_RAMP_THRESHOLD = 150.0
CURTAILMENT_RAMP_THRESHOLD = 100.0


def analyze_duck_curve(predictions, timestamps=None):
    """
    Perform duck-curve analysis on a 24-hour GHI profile.

    Args:
        predictions: Array of 24 GHI predictions (W/m²)
        timestamps: Optional array/list of timestamp strings or hours

    Returns:
        dict with analysis results
    """
    predictions = np.array(predictions, dtype=float)
    n = len(predictions)

    if timestamps is None:
        hours = np.arange(n)
    else:
        hours = np.arange(n)

    # Ramp rates (change per time step)
    ramp_rates = np.diff(predictions, prepend=predictions[0])

    # Morning ramp: first half of day
    mid = n // 2
    morning_ramp = ramp_rates[:mid]
    evening_ramp = ramp_rates[mid:]

    morning_ramp_rate = float(np.max(morning_ramp))
    evening_ramp_rate = float(np.min(evening_ramp))

    # Peak generation
    peak_idx = int(np.argmax(predictions))
    peak_value = float(predictions[peak_idx])

    # High ramp events
    high_ramp_events = np.where(np.abs(ramp_rates) > HIGH_RAMP_THRESHOLD)[0].tolist()

    # Stability score: 100 = perfect, decreases with ramp severity
    max_abs_ramp = float(np.max(np.abs(ramp_rates))) if n > 1 else 0.0
    stability_score = max(0.0, 100.0 - (max_abs_ramp / HIGH_RAMP_THRESHOLD) * 30.0)
    stability_score = round(stability_score, 1)

    # Grid stress level
    if stability_score >= 80:
        stress_level = "LOW"
    elif stability_score >= 60:
        stress_level = "MODERATE"
    else:
        stress_level = "HIGH"

    return {
        "morning_ramp_rate": morning_ramp_rate,
        "evening_ramp_rate": evening_ramp_rate,
        "peak_generation": peak_value,
        "peak_hour": int(peak_idx),
        "high_ramp_events": high_ramp_events,
        "stability_score": stability_score,
        "stress_level": stress_level,
        "ramp_rates": ramp_rates.tolist(),
    }


def predict_curtailment_strategy(analysis_result, predictions):
    """
    Generate a predictive curtailment strategy based on duck curve analysis.

    Args:
        analysis_result: Output dict from analyze_duck_curve
        predictions: Array of 24 GHI predictions (W/m²)

    Returns:
        dict with curtailment recommendations
    """
    predictions = np.array(predictions, dtype=float)
    n = len(predictions)
    ramp_rates = np.array(analysis_result["ramp_rates"])

    curtailment_schedule = []
    for i in range(n):
        abs_ramp = abs(ramp_rates[i])
        if abs_ramp > HIGH_RAMP_THRESHOLD:
            curtailment_pct = min(40.0, (abs_ramp / HIGH_RAMP_THRESHOLD - 1.0) * 20.0 + 20.0)
            action = "CURTAIL"
        elif abs_ramp > CURTAILMENT_RAMP_THRESHOLD:
            curtailment_pct = min(20.0, (abs_ramp / CURTAILMENT_RAMP_THRESHOLD - 1.0) * 10.0 + 5.0)
            action = "REDUCE"
        else:
            curtailment_pct = 0.0
            action = "NORMAL"

        curtailment_schedule.append({
            "hour": i,
            "ghi_forecast": round(float(predictions[i]), 2),
            "ramp_rate": round(float(ramp_rates[i]), 2),
            "action": action,
            "curtailment_pct": round(curtailment_pct, 1),
        })

    # Dispatch recommendation based on stability
    stability = analysis_result["stability_score"]
    if stability >= 80:
        dispatch_recommendation = "Full dispatch recommended — grid is stable."
    elif stability >= 60:
        dispatch_recommendation = (
            "Moderate curtailment advised during high-ramp windows."
        )
    else:
        dispatch_recommendation = (
            "Significant curtailment required. Coordinate with grid operator."
        )

    high_ramp_hours = [
        e for e in analysis_result["high_ramp_events"]
    ]

    return {
        "curtailment_schedule": curtailment_schedule,
        "dispatch_recommendation": dispatch_recommendation,
        "high_ramp_hours": high_ramp_hours,
        "stability_score": analysis_result["stability_score"],
        "stress_level": analysis_result["stress_level"],
    }


def print_duck_curve_summary(analysis, curtailment):
    """Pretty-print the duck curve analysis results."""
    print("\n" + "=" * 60)
    print("DUCK CURVE ANALYSIS RESULTS")
    print("=" * 60)
    print(f"  Peak Generation  : {analysis['peak_generation']:.1f} W/m² "
          f"at hour {analysis['peak_hour']:02d}:00")
    print(f"  Morning Ramp Rate: {analysis['morning_ramp_rate']:.1f} W/m²/h")
    print(f"  Evening Ramp Rate: {analysis['evening_ramp_rate']:.1f} W/m²/h")
    print(f"  Stability Score  : {analysis['stability_score']}/100  "
          f"[{analysis['stress_level']} STRESS]")

    if analysis["high_ramp_events"]:
        hours_str = ", ".join(f"{h:02d}:00" for h in analysis["high_ramp_events"])
        print(f"  High Ramp Events : {hours_str}")
    else:
        print("  High Ramp Events : None detected ✓")

    print("\nCURTAILMENT STRATEGY:")
    print(f"  {curtailment['dispatch_recommendation']}")

    curtailed = [
        s for s in curtailment["curtailment_schedule"]
        if s["action"] != "NORMAL"
    ]
    if curtailed:
        print("\n  Hour  |  Action  | Curtailment")
        print("  ------|----------|------------")
        for s in curtailed:
            print(f"  {s['hour']:02d}:00 | {s['action']:<8} | {s['curtailment_pct']:.1f}%")
    else:
        print("  No curtailment required for this forecast period. ✓")
    print("=" * 60)
