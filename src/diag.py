import json
import time
import ollama
from rag import retrieve

MODEL = "mistral"
LABELS = ["none", "propulsion_failure", "navigation_drift", "communication_anomaly"]
CONFIDENCE = ["low", "medium", "high"]

SYSTEM = f"""You are a diagnostic expert for an autonomous underwater vehicle (AUV).
You receive a telemetry summary and reference documents.
Decide which single state best explains the telemetry, using ONLY the signs described in the reference documents.
Small random fluctuations are normal. If everything is normal, the answer is "none".

Allowed values for "anomaly": {LABELS}
Allowed values for "confidence": {CONFIDENCE}

Reply with ONE JSON object that has ALL FOUR keys, in this order:
{{"anomaly": "<one allowed value>", "confidence": "<low|medium|high>", "evidence": "<one short sentence with key numbers>", "recommendation": "<one short sentence from the reference documents>"}}

Example of a correct reply:
{{"anomaly": "navigation_drift", "confidence": "high", "evidence": "Position error grew from 1.1 m to 8.0 m while thrusters, heading and communication are normal.", "recommendation": "Perform a position fix and recalibrate the navigation filter."}}

Never omit a key. Never add other keys."""


def build_user_prompt(item, k):
    docs = retrieve(item["summary"], k=k)
    ref = "\n\n".join(f"[{name}]\n{text}" for name, text in docs)
    return (
        f"REFERENCE DOCUMENTS:\n{ref}\n\n"
        f"TELEMETRY SUMMARY:\n{item['summary']}\n\n"
        f"KEY NUMBERS:\n{json.dumps(item['facts'])}"
    )


def check(obj):
    """Returns (ok, reason)."""
    if not isinstance(obj, dict):
        return False, "not a JSON object"
    for key in ["anomaly", "confidence", "evidence", "recommendation"]:
        if key not in obj:
            return False, f"missing key: {key}"
    if obj["anomaly"] not in LABELS:
        return False, f"anomaly must be one of {LABELS}"
    if obj["confidence"] not in CONFIDENCE:
        return False, f"confidence must be one of {CONFIDENCE}"
    return True, ""


def call_llm(messages):
    resp = ollama.chat(
        model=MODEL,
        messages=messages,
        format="json",
        options={"temperature": 0, "num_predict": 450, "num_ctx": 2048},
        keep_alive="30m",
    )
    return resp["message"]["content"]


def diagnose(item, k=4):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_user_prompt(item, k)},
    ]
    raw = call_llm(messages)
    for attempt in range(2):             # 1 try + 1 retry
        try:
            obj = json.loads(raw)
            ok, reason = check(obj)
        except json.JSONDecodeError:
            obj, ok, reason = None, False, "invalid JSON"
        if ok:
            return {"valid": True, "retried": attempt == 1, **obj}
        if attempt == 0:
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",
                             "content": f"Your answer was invalid: {reason}. Reply again with valid JSON only."})
            raw = call_llm(messages)
    return {"valid": False, "raw": raw}


if __name__ == "__main__":
    for sid in ["s001", "s016", "s031", "s046"]:
        item = json.load(open(f"data/summaries/{sid}.json", encoding="utf-8"))
        t0 = time.time()
        out = diagnose(item, k=4)
        print(f"--- {sid}  ({time.time() - t0:.1f} s)")
        print(json.dumps(out, indent=2, ensure_ascii=False))