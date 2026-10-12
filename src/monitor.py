import json
import pandas as pd
from pathlib import Path

LOGS = Path("data/logs")
OUT = Path("data/summaries")
OUT.mkdir(parents=True, exist_ok=True)
BASE_N = 10   # pehle 10 second = baseline
CUR_N = 10    # aakhri 10 second = current


def make_summary(sid):
    df = pd.read_csv(LOGS / f"{sid}.csv")
    base = df.iloc[:BASE_N]
    cur = df.iloc[-CUR_N:]
    t_start = int(cur["t"].iloc[0])
    t_end = int(cur["t"].iloc[-1])

    facts = {}
    sentences = []

    # --- Thrusters ---
    rpm_change = {}
    changed = []
    for i in range(1, 5):
        col = f"thruster{i}_rpm"
        b, c = base[col].mean(), cur[col].mean()
        pct = (c - b) / b * 100
        rpm_change[f"t{i}"] = round(pct, 1)
        if abs(pct) > 10:
            changed.append(f"Thruster {i} RPM changed from {b:.0f} to {c:.0f} ({pct:+.0f}%)")
    facts["thruster_rpm_change_pct"] = rpm_change
    if changed:
        sentences.append("; ".join(changed) + ". Other thrusters are within normal range.")
    else:
        mean_rpm = cur[[f"thruster{i}_rpm" for i in range(1, 5)]].mean().mean()
        sentences.append(f"All 4 thrusters are stable around {mean_rpm:.0f} RPM.")

    # --- Heading ---
    hd = cur["heading_deg"].mean() - base["heading_deg"].mean()
    facts["heading_change_deg"] = round(hd, 1)
    if abs(hd) > 3:
        sentences.append(f"Heading changed by {hd:+.1f} degrees.")
    else:
        sentences.append("Heading is steady.")

    # --- Depth ---
    dd = cur["depth_m"].mean() - base["depth_m"].mean()
    facts["depth_change_m"] = round(dd, 2)
    if abs(dd) > 0.8:
        sentences.append(f"Depth changed by {dd:+.1f} m.")
    else:
        sentences.append("Depth is steady.")

    # --- Position error ---
    pb, pc = base["pos_error_m"].mean(), cur["pos_error_m"].mean()
    facts["pos_error_start_m"] = round(pb, 2)
    facts["pos_error_end_m"] = round(pc, 2)
    if pc - pb > 2:
        sentences.append(f"Position error grew from {pb:.1f} m to {pc:.1f} m.")
    else:
        sentences.append(f"Position error is stable (about {pc:.1f} m).")

    # --- Communication ---
    lb, lc = base["packet_loss_pct"].mean(), cur["packet_loss_pct"].mean()
    mb, mc = base["latency_ms"].mean(), cur["latency_ms"].mean()
    facts["packet_loss_start_pct"] = round(lb, 1)
    facts["packet_loss_end_pct"] = round(lc, 1)
    facts["latency_start_ms"] = round(mb)
    facts["latency_end_ms"] = round(mc)
    
    if (lc - lb) > 5 or (mc - mb) > 120:
        sentences.append(
            f"Communication degraded: packet loss rose from {lb:.1f}% to {lc:.1f}%, "
            f"latency from {mb:.0f} ms to {mc:.0f} ms."
        )
    else:
        sentences.append(
            f"Communication shows no clear change (packet loss about {lc:.1f}%, latency about {mc:.0f} ms)."
        )

    return {
        "id": sid,
        "window": {"baseline": f"t=0-{BASE_N - 1}s", "current": f"t={t_start}-{t_end}s"},
        "summary": " ".join(sentences),
        "facts": facts,
    }


if __name__ == "__main__":
    ids = sorted(p.stem for p in LOGS.glob("*.csv"))
    for sid in ids:
        result = make_summary(sid)
        with open(OUT / f"{sid}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"{len(ids)} summaries bani: data/summaries/")