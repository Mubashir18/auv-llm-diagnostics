import json
import sys
from diag import diagnose

sid = sys.argv[1] if len(sys.argv) > 1 else "s016"
item = json.load(open(f"data/summaries/{sid}.json", encoding="utf-8"))
out = diagnose(item, k=4)
print(json.dumps(out, indent=2, ensure_ascii=False))