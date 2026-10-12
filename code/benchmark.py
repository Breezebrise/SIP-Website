"""
AI Efficiency Benchmarking for Cybersecurity Analysis
Proof-of-concept skeleton (SIP311, design phase)

Author: Nicole Welch

Runs the same labeled network flow data through two detection pipelines:
  1. Traditional: rule-based thresholds (stands in for Zeek detection rules)
  2. AI: a scikit-learn Random Forest classifier

Each pipeline is wrapped in a CodeCarbon tracker so energy use and
emissions are recorded alongside detection accuracy and false positives.

Input: a CSV of flow features with a "label" column (1 = malicious, 0 = benign).
Usage: python benchmark.py --data flows.csv
"""

import argparse
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

try:
    from codecarbon import EmissionsTracker
except ImportError:
    EmissionsTracker = None


def measure(name, func):
    """Run func() while tracking time and, if available, energy and emissions."""
    tracker = None
    if EmissionsTracker is not None:
        tracker = EmissionsTracker(project_name=name, log_level="error", save_to_file=False)
        tracker.start()
    start = time.perf_counter()
    predictions = func()
    seconds = time.perf_counter() - start
    emissions_kg = None
    energy_kwh = None
    if tracker is not None:
        emissions_kg = tracker.stop()
        energy_kwh = tracker.final_emissions_data.energy_consumed
    return predictions, seconds, energy_kwh, emissions_kg


def traditional_pipeline(test):
    """Rule-based detection. Thresholds mirror simple Zeek-style heuristics."""
    rules = (
        (test["conn_count"] > 100)            # beaconing or scanning volume
        | (test["bytes_out"] > 5_000_000)     # large outbound transfer
        | (test["dst_port"].isin([6667, 4444]))  # IRC and common C2 ports
    )
    return rules.astype(int).values


def ai_pipeline(train, test, features):
    """Train and run a Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(train[features], train["label"])
    return model.predict(test[features])


def score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return accuracy_score(y_true, y_pred), fpr, tp


def main():
    parser = argparse.ArgumentParser(description="Compare AI and traditional detection on accuracy and energy.")
    parser.add_argument("--data", required=True, help="CSV of labeled flow features")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    features = ["conn_count", "bytes_out", "bytes_in", "duration", "dst_port"]
    train, test = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])

    results = {}
    results["Traditional (rules)"] = measure("traditional", lambda: traditional_pipeline(test))
    results["AI (Random Forest)"] = measure("ai", lambda: ai_pipeline(train, test, features))

    print(f"\n{'Pipeline':<22}{'Accuracy':>10}{'FPR':>8}{'Time (s)':>10}{'kWh':>12}{'kg CO2eq':>12}{'Wh per TP':>12}")
    for name, (pred, secs, kwh, kg) in results.items():
        acc, fpr, tp = score(test["label"].values, pred)
        wh_per_tp = (kwh * 1000 / tp) if (kwh is not None and tp) else None
        fmt = lambda v, f: format(v, f) if v is not None else "n/a"
        print(f"{name:<22}{acc:>10.3f}{fpr:>8.3f}{secs:>10.3f}{fmt(kwh, '.6f'):>12}{fmt(kg, '.6f'):>12}{fmt(wh_per_tp, '.4f'):>12}")

    if EmissionsTracker is None:
        print("\nCodeCarbon is not installed, so energy columns show n/a. Install with: pip install codecarbon")


if __name__ == "__main__":
    main()
