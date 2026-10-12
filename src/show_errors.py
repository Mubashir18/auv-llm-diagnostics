import json
import pandas as pd

labels = pd.read_csv("data/labels_hard.csv")
res = {}
for line in open("data/results_hard.jsonl", encoding="utf-8"):
    r = json.loads(line)
    res[r["id"]] = r

for row in labels.itertuples():
    pred = res[row.id]["pred"]
    if pred != row.label:
        s = json.load(open(f"data/summaries/{row.id}.json", encoding="utf-8"))
        print(f"--- {row.id} true={row.label} pred={pred} start_t={row.anomaly_start_t}")
        print(s["summary"])
        print()