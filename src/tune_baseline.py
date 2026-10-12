import itertools
import json
import pandas as pd

LABELS = ["none", "propulsion_failure", "navigation_drift", "communication_anomaly"]


def load(labels_csv):
    df = pd.read_csv(labels_csv)
    rows = []
    for r in df.itertuples():
        f = json.load(open(f"data/summaries/{r.id}.json", encoding="utf-8"))["facts"]
        rows.append((r.label, f))
    return rows


def rule(f, rpm, loss, lat, pos):
    if any(abs(v) > rpm for v in f["thruster_rpm_change_pct"].values()):
        return "propulsion_failure"
    if (f["packet_loss_end_pct"] - f["packet_loss_start_pct"] > loss
            or f["latency_end_ms"] - f["latency_start_ms"] > lat):
        return "communication_anomaly"
    if f["pos_error_end_m"] - f["pos_error_start_m"] > pos:
        return "navigation_drift"
    return "none"


def evaluate(rows, params):
    preds = [rule(f, *params) for _, f in rows]
    true = [t for t, _ in rows]
    acc = sum(p == t for p, t in zip(preds, true)) / len(rows)
    normal = [p for p, t in zip(preds, true) if t == "none"]
    fpr = sum(p != "none" for p in normal) / len(normal)
    return acc, fpr, true, preds


def show(name, true, preds):
    acc = sum(p == t for p, t in zip(preds, true)) / len(true)
    normal = [p for p, t in zip(preds, true) if t == "none"]
    fpr = sum(p != "none" for p in normal) / len(normal)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.1%}   FPR: {fpr:.1%}  (n={len(true)})")
    cm = pd.crosstab(pd.Series(true, name="true"), pd.Series(preds, name="pred"))
    print(cm.reindex(index=LABELS, columns=LABELS, fill_value=0).to_string())


dev = load("data/labels_hard.csv")
test = load("data/labels_test.csv")

grid = itertools.product(
    [10, 15, 20, 25, 30],            # RPM change %
    [3, 5, 8, 10, 15, 20],           # packet loss increase (percentage points)
    [50, 80, 120, 200, 300, 500],    # latency increase (ms)
    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],  # position error increase (m)
)

best = None
for params in grid:
    acc, fpr, _, _ = evaluate(dev, params)
    key = (acc, -fpr)                # pehle accuracy, phir kam FPR
    if best is None or key > best[0]:
        best = (key, params)

params = best[1]
print(f"Dev (h) par best thresholds: rpm>{params[0]}%, loss increase>{params[1]} pts, "
      f"latency increase>{params[2]} ms, pos error increase>{params[3]} m")

hand = (30, 20, 500, 3.0)   # purane haath se rakhe numbers (comm ke liye increase se)
for name, p in [("Hand-set baseline", hand), ("Dev-tuned baseline", params)]:
    for set_name, rows in [("DEV (h)", dev), ("TEST (t)", test)]:
        acc, fpr, true, preds = evaluate(rows, p)
        show(f"{name} on {set_name}", true, preds)