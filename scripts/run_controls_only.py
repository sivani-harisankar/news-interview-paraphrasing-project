"""
run_controls_only.py  --  Re-run ONLY the control check of the LLM-as-judge.

Control = a deliberately WRONG prediction ("not a paraphrase") shown for a pair where DeBERTa and you
both said "paraphrase". A trustworthy judge should reject it. This script runs the control on ALL such
pairs (not just 10) and reports how often the judge correctly rejects it.

run:
    python run_controls_only.py
"""

import csv
import math

import llm_judge as v1
import llm_judge_predictions as jp       

OUTPUT_FILE = "judge_controls_outputs.tsv"
ONLY_FIRST_N = None          


def wilson_interval(successes, n, z=1.96):
    """95% confidence interval for a proportion (honest about small samples)."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def summarise(name, rows):
    ok = [r for r in rows if r["label_acceptable"] in ("0", "1")]
    rejected = sum(1 for r in ok if r["label_acceptable"] == "0")
    low, high = wilson_interval(rejected, len(ok))
    print("  %-8s judge correctly rejects %d of %d  (%.0f%%, 95%% interval %.0f%%-%.0f%%)" % (
        name, rejected, len(ok), 100 * rejected / len(ok) if ok else 0, 100 * low, 100 * high))


def main():
    from openai import OpenAI
    client = OpenAI()

    pairs = v1.load_rogan() + v1.load_obama()
    items = [p for p in pairs if p["human_label"] == "1" and p["model_label"] == "1"]
    print("Found", len(items), "pairs where DeBERTa and you both said paraphrase.")
    if not items:
        print("STOP: none found. Check the file names in llm_judge.py.")
        return
    if ONLY_FIRST_N:
        items = items[:ONLY_FIRST_N]
        print("TEST RUN: only", len(items), "controls.")

    columns = ["dataset", "pair_id", "guest_text", "host_text", "label_acceptable",
               "votes_label_acceptable", "runs_parsed", "explanation"]
    results = []
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(columns)
        for i, p in enumerate(items, start=1):
            res = jp.judge_one(client, p["guest_text"], p["host_text"], "No", "", "")
            row = dict(p)
            row.update(res)
            row = {k: str(row.get(k, "")).replace("\t", " ").replace("\n", " ") for k in columns}
            results.append(row)
            writer.writerow([row[c] for c in columns])
            print(i, "/", len(items), p["dataset"], p["pair_id"], "-> judge accepts the wrong 'No'?",
                  "YES (bad)" if res["label_acceptable"] == 1 else "no (good)")

    print("\nCONTROL RESULT (should be close to 100% rejected):")
    summarise("rogan", [r for r in results if r["dataset"] == "rogan"])
    summarise("obama", [r for r in results if r["dataset"] == "obama"])
    summarise("pooled", results)
    wrongly_accepted = [r for r in results if r["label_acceptable"] == "1"]
    if wrongly_accepted:
        print("\nWrong 'No' predictions the judge ACCEPTED (look at these yourself):")
        for r in wrongly_accepted:
            print(" [%s %s] %s" % (r["dataset"], r["pair_id"], r["explanation"][:200]))
    print("\nSaved to", OUTPUT_FILE)


if __name__ == "__main__":
    main()
