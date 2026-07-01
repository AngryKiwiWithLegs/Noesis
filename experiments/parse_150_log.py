#!/usr/bin/env python3
"""Parse the 150Q A/B run log into a verifiable JSON (partial run, n=111)."""
import re, json
from datetime import datetime

log = open("/tmp/ab_en_150.log").read()
RESULTS = "/Users/mac27ssd/ZCodeProject/noesis_experiment/results"

# Parse lines like:
#   [🟢] What is my job title?                mem✅ direct❌ ('backend')
#   [🟡] What do I code in now?               mem❌ direct❌ ('typescript')
pat = re.compile(r"\[([🟢🟡🔴])\]\s+(.+?)\s+mem(✅|❌)\s+direct(✅|❌)\s+\('(.+?)'\)")
matches = pat.findall(log)

details = []
current_profile = None
for line in log.split("\n"):
    m_prof = re.search(r"Profile: (prof_\d+)", line)
    if m_prof:
        current_profile = m_prof.group(1)
    m = pat.search(line)
    if m:
        emoji, question, mem_mark, direct_mark, expect = m.groups()
        details.append({
            "profile": current_profile,
            "question": question.strip(),
            "expect": expect,
            "with_mem_hit": mem_mark == "✅",
            "without_mem_hit": direct_mark == "✅",
            "with_mem_ans": "",      # raw answers not in the summary log line
            "without_mem_ans": "",
        })

n = len(details)
with_hit = sum(1 for r in details if r["with_mem_hit"])
without_hit = sum(1 for r in details if r["without_mem_hit"])
b = sum(1 for r in details if r["with_mem_hit"] and not r["without_mem_hit"])
c = sum(1 for r in details if not r["with_mem_hit"] and r["without_mem_hit"])
chi2 = (abs(b - c) - 1)**2 / (b + c) if (b + c) else 0

report = {
    "experiment": "ab_comparison_en_150",
    "language": "en",
    "model": "gemini-flash-lite-latest",
    "note": "Partial run: Gemini free-tier rate-limited after n=111 of planned 150. Captured questions are complete and valid.",
    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "num_questions": n,
    "summary": {"with_hit": with_hit, "without_hit": without_hit, "total": n,
                "errors": 0},
    "discordant": {"b": b, "c": c, "chi2_mcnemar": round(chi2, 2)},
    "details": details,
}
path = f"{RESULTS}/ab_comparison_en_150_20260630.json"
json.dump(report, open(path, "w"), ensure_ascii=False, indent=2)

print(f"Parsed {n} questions from log.")
print(f"  with-mem:    {with_hit}/{n} ({100*with_hit/n:.1f}%)")
print(f"  without-mem: {without_hit}/{n} ({100*without_hit/n:.1f}%)")
print(f"  improvement: +{with_hit-without_hit} ({100*(with_hit-without_hit)/n:.1f}pp)")
print(f"  discordant:  b={b} c={c}  McNemar chi2={chi2:.2f}")
print(f"  critical (p<0.05): 3.84 → {'SIGNIFICANT' if chi2 > 3.84 else 'NOT SIGNIFICANT'}")
print(f"  saved: {path}")
