import shutil
from pathlib import Path
import chromadb

KB_DIR = Path("data/kb")
DB_DIR = "data/chroma"

DOCS = {
"propulsion_failure": """Fault: Propulsion failure (thruster failure)
Typical signs: One thruster's RPM drops sharply (by more than 30 to 50 percent) or goes to zero, while the other thrusters stay near normal RPM. Because thrust becomes unbalanced, the heading drifts steadily without any commanded turn. Depth may also change slowly. Position error and communication stay normal.
How to tell it apart: the key sign is that exactly one thruster RPM changed a lot. If position error and communication are normal but a thruster RPM dropped, it is a propulsion failure.
Recommended actions: reduce speed and switch to reduced-thrust mode; if heading cannot be held, abort the mission; return to surface or recovery point for inspection; check thruster power supply, motor controller and propeller blockage.""",

"navigation_drift": """Fault: Navigation drift (position estimation error)
Typical signs: All thrusters are stable around the same RPM. Heading and depth are steady. Position error is small (about 1 to 3 meters) and not growing. Packet loss of a few percent (up to about 6 percent) and latency of about 100 to 350 milliseconds can be normal if they stay at that level. Only a clear rise over time is an anomaly. Small random fluctuations are normal noise and are not anomalies.

Recommended actions: perform a position fix (surface GPS or acoustic transponder fix); recalibrate or reset the navigation filter; reduce reliance on dead reckoning; shorten the mission leg; check the DVL and INS sensors.""",

"communication_anomaly": """Fault: Communication anomaly (degraded acoustic link)
Typical signs: Packet loss rises strongly (tens of percent) and latency rises to many hundreds of milliseconds or seconds. Thrusters, heading, depth and position error stay normal. Typical causes are multipath, ambient noise, long range between nodes or a node moving out of range.
How to tell it apart: the key sign is high packet loss and latency with a healthy vehicle. This is a network problem, not a hardware fault of the vehicle.
Recommended actions: move closer to the neighbor or relay node; reduce data rate or packet size; increase retransmission timeouts; switch to store-and-forward mode; do not abort the mission only because of this fault.""",

"normal_operation": """State: Normal operation (no anomaly)
Typical signs: All thrusters are stable around the same RPM. Heading and depth are steady. Position error is small (about 1 to 2 meters). Packet loss is low (a few percent at most) and latency is about 100 to 300 milliseconds. Small random fluctuations are normal noise and are not anomalies.
Recommended actions: no anomaly detected; continue the mission and keep monitoring.""",
}


def build_index():
    KB_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in DOCS.items():
        (KB_DIR / f"{name}.md").write_text(text, encoding="utf-8")
    shutil.rmtree(DB_DIR, ignore_errors=True)   # har baar fresh index
    client = chromadb.PersistentClient(path=DB_DIR)
    col = client.get_or_create_collection("auv_kb")
    files = sorted(KB_DIR.glob("*.md"))
    col.add(
        ids=[f.stem for f in files],
        documents=[f.read_text(encoding="utf-8") for f in files],
    )
    return len(files)


def retrieve(summary, k=2):
    client = chromadb.PersistentClient(path=DB_DIR)
    col = client.get_collection("auv_kb")
    res = col.query(query_texts=[summary], n_results=k)
    return list(zip(res["ids"][0], res["documents"][0]))


if __name__ == "__main__":
    import json
    n = build_index()
    print(f"{n} documents index hue.\n")
    for sid in ["s001", "s016", "s031", "s046"]:
        s = json.load(open(f"data/summaries/{sid}.json", encoding="utf-8"))["summary"]
        hits = retrieve(s, k=2)
        print(sid, "->", [h[0] for h in hits])