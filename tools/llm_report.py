#!/usr/bin/env python3
"""
Step 7: LLM incident report with evidence validation.

  python3 tools/llm_report.py --alerts reports/alerts.jsonl [--model llama3.2:3b]
                              [--no-llm] [--inject-error]

Pipeline (the LLM never sees raw CAN data and can never act):
 1. EVIDENCE   - deterministic Python summary of alerts.jsonl: per attack class
                 the alert count, vehicles, CAN IDs, time range, mean confidence,
                 and a rule-based severity.
 2. LLM DRAFT  - Llama 3 (local, via Ollama) writes the report as strict JSON
                 from that evidence only (temperature 0).
 3. VALIDATE   - every claim is checked against the evidence: attack type exists,
                 vehicles/CAN IDs/counts/severity match, no attack left out, and
                 every number in the free-text summary is a real number from the
                 evidence. Failed claims are FLAGGED in the report, not hidden.
 4. ADVISORY   - recommended actions are printed for a human analyst. The tool
                 executes nothing: 0 automatic actions.

--no-llm       : skip the LLM, build the draft from a template (works offline/CI)
--inject-error : corrupt the draft on purpose to demonstrate the validator
"""
import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone

SEVERITY_RULES = {          # deterministic, documented, auditable
    "DoS": "critical",      # bus availability lost: brakes/steering messages delayed
    "STEERING_WHEEL": "high",
    "SPEED": "high",
    "GAS": "high",
    "RPM": "high",
    "ZERO_DAY": "medium",   # unknown pattern: needs analyst triage
}
SEVERITIES = ["low", "medium", "high", "critical"]


# ---------------------------------------------------------------- 1. evidence
def build_evidence(alert_path):
    per = defaultdict(lambda: {"count": 0, "vehicles": set(), "can_ids": Counter(),
                               "t0": None, "t1": None, "prob_sum": 0.0})
    with open(alert_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            a = json.loads(line)
            e = per[a["class"]]
            e["count"] += 1
            e["vehicles"].add(int(a["vehicle"]))
            e["can_ids"][a["top_can_id"]] += 1
            t = int(a["ts_us"])
            e["t0"] = t if e["t0"] is None else min(e["t0"], t)
            e["t1"] = t if e["t1"] is None else max(e["t1"], t)
            e["prob_sum"] += float(a["prob"])
    evidence = {"attacks": {}}
    for cls, e in sorted(per.items()):
        evidence["attacks"][cls] = {
            "alert_count": e["count"],
            "vehicles": sorted(e["vehicles"]),
            "top_can_ids": [cid for cid, _ in e["can_ids"].most_common(3)],
            "first_seen_utc": iso(e["t0"]),
            "last_seen_utc": iso(e["t1"]),
            "mean_confidence": round(e["prob_sum"] / e["count"], 3),
            "severity": SEVERITY_RULES.get(cls, "medium"),
        }
    all_veh = sorted({v for a in evidence["attacks"].values() for v in a["vehicles"]})
    evidence["total_alerts"] = sum(a["alert_count"] for a in evidence["attacks"].values())
    evidence["vehicles_affected"] = all_veh
    return evidence


def iso(us):
    return datetime.fromtimestamp(us / 1e6, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- 2. LLM draft
PROMPT = """You are a vehicle-security analyst. Write an incident report using ONLY the
evidence below. Do not invent vehicles, CAN IDs, counts or attack types.
Return ONLY JSON with exactly this schema:
{{"summary": "<2-3 sentences>",
  "incidents": [{{"attack_type": "<string>", "vehicles": [<int>], "alert_count": <int>,
                  "top_can_ids": ["<hex string>"], "severity": "low|medium|high|critical",
                  "explanation": "<1-2 sentences>", "recommended_actions": ["<string>"]}}]}}
Include one incident per attack type in the evidence. Copy numbers exactly.

EVIDENCE:
{evidence}
"""


def llm_draft(evidence, model, host):
    body = json.dumps({
        "model": model,
        "prompt": PROMPT.format(evidence=json.dumps(evidence, indent=1)),
        "format": "json",          # Ollama constrains output to valid JSON
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 2048, "num_thread": 4},  # small context, bounded CPU
        "keep_alive": 0,                 # unload model from RAM right after
    }).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(json.loads(r.read())["response"])


def template_draft(evidence):
    inc = []
    for cls, a in evidence["attacks"].items():
        inc.append({"attack_type": cls, "vehicles": a["vehicles"], "alert_count": a["alert_count"],
                    "top_can_ids": a["top_can_ids"], "severity": a["severity"],
                    "explanation": f"{cls} activity detected on CAN IDs {', '.join(a['top_can_ids'])}.",
                    "recommended_actions": ["Isolate the affected ECU gateway port for inspection",
                                            "Collect a full CAN capture from the affected vehicles"]})
    return {"summary": f"{evidence['total_alerts']} alerts across "
                       f"{len(evidence['vehicles_affected'])} vehicles and "
                       f"{len(evidence['attacks'])} attack types.",
            "incidents": inc}


def inject_errors(draft, evidence):
    """Simulate typical LLM hallucinations so the validator can be demonstrated."""
    if not isinstance(draft.get("incidents"), list):
        draft["incidents"] = []
    first = next((i for i in draft["incidents"] if isinstance(i, dict)), None)
    if first is not None:
        aslist = lambda x: x if isinstance(x, list) else ([] if x is None else [x])
        cnt = first.get("alert_count")
        first["vehicles"] = aslist(first.get("vehicles")) + [99]          # vehicle that doesn't exist
        first["alert_count"] = (cnt if isinstance(cnt, int) else 0) + 17   # wrong count
        first["top_can_ids"] = aslist(first.get("top_can_ids")) + ["7FF"]  # CAN ID not in evidence
    draft["incidents"].append({"attack_type": "FUZZING", "vehicles": [1], "alert_count": 5,
                               "top_can_ids": ["123"], "severity": "high",
                               "explanation": "invented", "recommended_actions": []})
    draft["summary"] = draft.get("summary", "") + " A further 4242 alerts were seen."
    return draft


# ---------------------------------------------------------------- 3. validate
def validate(draft, evidence):
    """Returns a list of (claim, ok, reason). Pure function: easy to unit-test."""
    ev = evidence["attacks"]
    checks = []

    def check(claim, ok, why=""):
        checks.append((claim, bool(ok), why))

    if not isinstance(draft.get("incidents"), list):
        check("report has an incidents list", False, "missing or wrong type")
        return checks

    reported = set()
    for n, inc in enumerate(draft["incidents"]):
        # LLM output is UNTRUSTED: check the schema before using any field
        if not isinstance(inc, dict):
            check(f"incident #{n} is an object", False, f"got {type(inc).__name__}")
            continue
        cls = str(inc.get("attack_type", ""))
        reported.add(cls)
        if cls not in ev:
            check(f"attack type '{cls}' exists", False, "not in evidence (hallucinated)")
            continue
        e = ev[cls]
        check(f"[{cls}] attack type exists", True)

        veh_raw = inc.get("vehicles")
        if not isinstance(veh_raw, list):
            check(f"[{cls}] vehicles is a list", False,
                  f"schema violation: got {type(veh_raw).__name__} {veh_raw!r}")
        else:
            veh = [int(v) for v in veh_raw if str(v).strip().lstrip("-").isdigit()]
            junk = [v for v in veh_raw if not str(v).strip().lstrip("-").isdigit()]
            bad_v = sorted(set(veh) - set(e["vehicles"]))
            why = (f"not in evidence: {bad_v}" if bad_v else
                   f"non-numeric: {junk}" if junk else "empty" if not veh else "")
            check(f"[{cls}] vehicles {veh_raw}", not why, why)

        cnt = inc.get("alert_count")
        if isinstance(cnt, bool) or not isinstance(cnt, int):
            check(f"[{cls}] alert_count is an integer", False,
                  f"schema violation: got {type(cnt).__name__} {cnt!r}")
        else:
            check(f"[{cls}] alert_count = {cnt}", cnt == e["alert_count"],
                  "" if cnt == e["alert_count"] else f"evidence says {e['alert_count']}")

        ids_raw = inc.get("top_can_ids")
        if not isinstance(ids_raw, list):
            check(f"[{cls}] top_can_ids is a list", False,
                  f"schema violation: got {type(ids_raw).__name__} {ids_raw!r}")
        else:
            ids = [str(c).upper().replace("0X", "").zfill(3) for c in ids_raw]
            bad_id = sorted(set(ids) - set(e["top_can_ids"]))
            check(f"[{cls}] CAN IDs {ids}", not bad_id, f"not in evidence: {bad_id}" if bad_id else "")

        sev = str(inc.get("severity", "")).lower()
        check(f"[{cls}] severity = {sev}", sev == e["severity"],
              "" if sev == e["severity"] else f"rule says {e['severity']}")

    for cls in ev:
        check(f"[{cls}] is reported (completeness)", cls in reported,
              "" if cls in reported else "attack in evidence but missing from report")

    # every number in the free-text summary must come from the evidence
    allowed = {evidence["total_alerts"], len(evidence["vehicles_affected"]), len(ev)}
    for e in ev.values():
        allowed |= {e["alert_count"], len(e["vehicles"])} | set(e["vehicles"])
    nums = [int(n) for n in re.findall(r"\b\d+\b", str(draft.get("summary", "")))]
    bad_n = sorted({n for n in nums if n not in allowed})
    check(f"summary numbers {nums}", not bad_n, f"not in evidence: {bad_n}" if bad_n else "")
    return checks


# ---------------------------------------------------------------- 4. report
def render(draft, evidence, checks, source):
    ok = sum(1 for _, k, _ in checks if k)
    lines = ["# IoV Incident Report", "",
             f"*Draft by: {source}. Every claim below was checked against detector evidence.*",
             f"**Validation: {ok}/{len(checks)} claims verified"
             + (" ✅**" if ok == len(checks) else f" — {len(checks) - ok} FLAGGED ❌**"), "",
             "## Summary", draft.get("summary", ""), "", "## Incidents"]
    incidents = [i for i in draft.get("incidents", []) if isinstance(i, dict)] \
        if isinstance(draft.get("incidents"), list) else []
    for inc in incidents:
        cls = inc.get("attack_type")
        grounded = cls in evidence["attacks"]
        lines += [f"### {cls} {'' if grounded else '❌ NOT IN EVIDENCE'}",
                  f"- Severity: **{inc.get('severity')}**",
                  f"- Vehicles: {inc.get('vehicles')}",
                  f"- Alerts: {inc.get('alert_count')}",
                  f"- CAN IDs: {inc.get('top_can_ids')}",
                  f"- {inc.get('explanation', '')}"]
        if grounded:
            e = evidence["attacks"][cls]
            lines.append(f"- Seen: {e['first_seen_utc']} → {e['last_seen_utc']} UTC, "
                         f"mean confidence {e['mean_confidence']}")
        lines.append("")
    lines += ["## Recommended actions (ADVISORY — require human approval, nothing is executed)"]
    for inc in incidents:
        actions = inc.get("recommended_actions", [])
        for a in (actions if isinstance(actions, list) else [actions]):
            lines.append(f"- [{inc.get('attack_type')}] {a}")
    lines += ["", "## Validation log", "| claim | result | detail |", "|---|---|---|"]
    for claim, k, why in checks:
        lines.append(f"| {claim} | {'✅' if k else '❌'} | {why} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts", default="reports/alerts.jsonl")
    ap.add_argument("--out", default="reports/incident_report.md")
    ap.add_argument("--model", default="llama3.2:1b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--inject-error", action="store_true")
    args = ap.parse_args()

    evidence = build_evidence(args.alerts)
    if not evidence["attacks"]:
        sys.exit("no alerts in " + args.alerts)
    print(f"evidence: {evidence['total_alerts']} alerts, {len(evidence['attacks'])} attack types, "
          f"vehicles {evidence['vehicles_affected']}")

    source = "template (no LLM)"
    if args.no_llm:
        draft = template_draft(evidence)
    else:
        try:
            print(f"asking {args.model} via Ollama (CPU can take ~1-2 min)...")
            draft = llm_draft(evidence, args.model, args.host)
            source = f"{args.model} (local, Ollama)"
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            print(f"LLM unavailable ({e}); falling back to template")
            draft = template_draft(evidence)
    if args.inject_error:
        draft = inject_errors(draft, evidence)
        source += " + injected errors"

    checks = validate(draft, evidence)
    with open(args.out, "w") as fh:
        fh.write(render(draft, evidence, checks, source))
    bad = [c for c in checks if not c[1]]
    print(f"validation: {len(checks) - len(bad)}/{len(checks)} claims verified")
    for claim, _, why in bad:
        print(f"  FLAGGED: {claim} -> {why}")
    print(f"report -> {args.out}   (actions executed automatically: 0)")


if __name__ == "__main__":
    main()
