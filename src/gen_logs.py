import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)   # seed: har baar same data banega
OUT = Path("data/logs")
OUT.mkdir(parents=True, exist_ok=True)
N = 60                            # 60 second ka log
PER_CLASS = 15

def normal_log():
    t = np.arange(N)
    df = pd.DataFrame({"t": t})
    for i in range(1, 5):
        df[f"thruster{i}_rpm"] = 1500 + rng.normal(0, 15, N)
    df["depth_m"] = 50 + rng.normal(0, 0.1, N)
    df["heading_deg"] = 90 + rng.normal(0, 0.5, N)
    df["pos_error_m"] = 1 + np.abs(rng.normal(0, 0.3, N))
    df["packet_loss_pct"] = np.clip(rng.normal(1, 1, N), 0, None)
    df["latency_ms"] = 150 + rng.normal(0, 30, N)
    return df

def add_propulsion_failure(df, t0):
    t = df["t"].values
    ramp = np.clip((t - t0) / 5, 0, 1)            # 5 second mein girta hai
    k = rng.integers(1, 5)                         # kaun sa thruster
    sev = rng.uniform(0.5, 1.0)                    # kitna girta hai
    df[f"thruster{k}_rpm"] *= (1 - sev * ramp)
    sign = rng.choice([-1, 1])
    df["heading_deg"] += sign * rng.uniform(0.3, 0.8) * np.clip(t - t0, 0, None)
    df["depth_m"] += 0.05 * np.clip(t - t0, 0, None)

def add_navigation_drift(df, t0):
    t = df["t"].values
    growth = rng.uniform(0.2, 0.6)                 # meter per second
    df["pos_error_m"] += growth * np.clip(t - t0, 0, None)

def add_communication_anomaly(df, t0):
    t = df["t"].values
    ramp = np.clip((t - t0) / 3, 0, 1)
    df["packet_loss_pct"] += ramp * rng.uniform(30, 70)
    df["latency_ms"] += ramp * rng.uniform(600, 2500)

CLASSES = {
    "none": None,
    "propulsion_failure": add_propulsion_failure,
    "navigation_drift": add_navigation_drift,
    "communication_anomaly": add_communication_anomaly,
}

labels = []
n = 0
for label, fn in CLASSES.items():
    for _ in range(PER_CLASS):
        n += 1
        sid = f"s{n:03d}"
        df = normal_log()
        t0 = int(rng.integers(15, 35))
        if fn is not None:
            fn(df, t0)
        df.round(2).to_csv(OUT / f"{sid}.csv", index=False)
        labels.append({"id": sid, "label": label,
                       "anomaly_start_t": t0 if fn else ""})

pd.DataFrame(labels).to_csv("data/labels.csv", index=False)
print(f"{len(labels)} scenarios bane. Dekho: data/logs/ aur data/labels.csv")