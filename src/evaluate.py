import argparse
import json
import time
from pathlib import Path
import pandas as pd
from diag import diagnose, LABELS


def threshold_rule(facts):
    """Simple baseline, bina LLM ke."""
    if any(abs(v) > 30 for v in facts["thruster_rpm_change_pct"].values()):
        return "propulsion_failure"
    if facts["packet_loss_end_pct"] > 20 or facts["latency_end_ms"] > 500:
        return "communication_anomaly"
    if facts["pos_error_end_m"] - facts["pos_error_start_m"] > 3:
        return "navigation_drift"
    return "none"


def metrics(name, y_true, y_pred):
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    acc = (df["true"] == df["pred"]).mean()
    normal = df[df["true"] == "none"]
    fpr = (normal["pred"] != "none").mean()
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.1%}   False Positive Rate: {fpr:.1%}  (n={len(df)})")
    cols = LABELS + ["invalid"]
    cm = pd.crosstab(df["true"], df["pred"]).reindex(index=LABELS, columns=cols, fill_value=0)
    print("Confusion matrix (rows = asli label, columns = prediction):")
    print(cm.to_string())
    return acc, fpr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=15)
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--results", default="data/results_llm.jsonl")
    args = ap.parse_args()

    RESULTS = Path(args.results)
    labels = pd.read_csv(args.labels)
    labels = labels.groupby("label", sort=False).head(args.per_class)

    done = {}
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["id"]] = r

    todo = labels[~labels["id"].isin(done)]
    print(f"Total {len(labels)} scenarios, baaki {len(todo)}")
    for i, row in enumerate(todo.itertuples(), 1):
        item = json.load(open(f"data/summaries/{row.id}.json", encoding="utf-8"))
        t0 = time.time()
        out = diagnose(item, k=4)
        secs = time.time() - t0
        rec = {
            "id": row.id,
            "pred": out["anomaly"] if out["valid"] else "invalid",
            "valid": out["valid"],
            "retried": out.get("retried", False),
            "secs": round(secs, 1),
            "raw": out.get("raw", "")[:300],
        }
        with open(RESULTS, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        done[row.id] = rec
        print(f"[{i}/{len(todo)}] {row.id} true={row.label} pred={rec['pred']} ({secs:.0f}s)")

    # --- Report ---
    ids = list(labels["id"])
    y_true = list(labels["label"])
    llm_pred = [done[i]["pred"] for i in ids]
    base_pred = [
        threshold_rule(json.load(open(f"data/summaries/{i}.json", encoding="utf-8"))["facts"])
        for i in ids
    ]
    a1, f1 = metrics("LLM (Mistral 7B + RAG)", y_true, llm_pred)
    a2, f2 = metrics("Threshold rule baseline", y_true, base_pred)

    recs = [done[i] for i in ids]
    print(f"\nInvalid outputs: {sum(not r['valid'] for r in recs)}   "
          f"Retried: {sum(r['retried'] for r in recs)}   "
          f"Avg time per sample: {sum(r['secs'] for r in recs) / len(recs):.0f} s")