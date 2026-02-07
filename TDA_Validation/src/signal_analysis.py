"""Signal-pattern analysis utilities.

Kept in a separate module so validator.py stays readable.
"""

from collections import Counter


def summarize_signal_patterns(event_results):
    """Aggregate signal patterns across all event detections.

    event_results is the dict returned by ValidationSuite.test_historical_events().
    """

    true_positive_detections = []
    for ev in event_results.get("details", []):
        for d in ev.get("all_detections", []) or []:
            if d.get("is_anomaly") and 0 <= d.get("days_before", 9999) <= 7:
                true_positive_detections.append(d)

    patterns = Counter()
    for d in true_positive_detections:
        b = bool(d.get("betti_signal"))
        w = bool(d.get("wasserstein_signal"))
        e = bool(d.get("entropy_signal"))
        key = "".join(["B" if b else "-", "W" if w else "-", "E" if e else "-"])
        patterns[key] += 1

    return {
        "total_true_positive_detections": len(true_positive_detections),
        "pattern_counts": dict(patterns),
    }
