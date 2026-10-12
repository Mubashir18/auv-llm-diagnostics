import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(7)      # alag seed
OUT = Path("data/logs")
OUT.mkdir(parents=True, exist_ok=True)
N = 60
PER_CLASS = 10


def noisy_log(hard_normal=False):
    t = np.arange(N)
    df = pd.DataFrame({"t": t})
    for i in range(1, 5):
        df[f"thruster{i}_rpm"] = 1500 + rng.normal(0, 38, N)
    df["depth_m"] = 50 + rng.normal(0, 0.25, N)
    df["heading_deg"] = 90 + rng.normal(0, 1.2, N)
    pos0 = rng.uniform(2, 3) if hard_normal else 1.0
    loss0 = rng.uniform(3, 6) if hard_normal else 1.0
    lat0 = rng.uniform(250, 350) if hard_normal else 150.0
    df["pos_error_m"] = pos0 + np.abs(rng.normal(0, 0.4, N))
    df["packet_loss_pct"] = np.clip(rng.normal(loss0, 1.5, N), 0, None)
    df["latency_ms"] = lat0 + rng.normal(0, 45, N)
    return df


def mild_propulsion(df, t0):
    t = df["t"].values
    ramp = np.clip((t - t0) / 5, 0, 1)
    k = rng.integers(1, 5)
    sev = rng.uniform(0.20, 0.35)
    df[f"thruster{k}_rpm"] *= (1 - sev * ramp)
    sign = rng.choice([-1, 1])
    df["heading_deg"] += sign * rng.uniform(0.1, 0.25) * np.clip(t - t0, 0, None)
    df["depth_m"] += 0.02 * np.clip(t - t0, 0, None)


def mild_navigation(df, t0):
    t = df["t"].values
    df["pos_error_m"] += rng.uniform(0.12, 0.35) * np.clip(t - t0, 0, None)


def mild_communication(df, t0):
    t = df["t"].values
    ramp = np.clip((t - t0) / 4, 0, 1)
    df["packet_loss_pct"] += ramp * rng.uniform(8, 25)
    df["latency_ms"] += ramp * rng.uniform(200, 700)


CLASSES = {
    "none": None,
    "propulsion_failure": mild_propulsion,
    "navigation_drift": mild_navigation,
    "communication_anomaly": mild_communication,
}

labels = []
n = 0
for label, fn in CLASSES.items():
    for _ in range(PER_CLASS):
        n += 1
        sid = f"h{n:03d}"
        df = noisy_log(hard_normal=(fn is None))
        t0 = int(rng.integers(15, 51))     # 50 tak: kuch bahut late
        if fn is not None:
            fn(df, t0)
        df.round(2).to_csv(OUT / f"{sid}.csv", index=False)
        labels.append({"id": sid, "label": label,
                       "anomaly_start_t": t0 if fn else ""})

pd.DataFrame(labels).to_csv("data/labels_hard.csv", index=False)
print(f"{len(labels)} hard scenarios bane: data/logs/h*.csv, data/labels_hard.csv")