"""
score_judge_predictions.py  --  Summarise the LLM-as-judge verdicts on DeBERTa's predictions.

    python score_judge_predictions.py [judge_pred_outputs.tsv]
"""

import csv
import sys

INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "judge_pred_outputs.tsv"


def read_rows():
    with open(INPUT_FILE, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def prf(truth, guess):
    tp = sum(1 for t, g in zip(truth, guess) if t == 1 and g == 1)
    fp = sum(1 for t, g in zip(truth, guess) if t == 0 and g == 1)
    fn = sum(1 for t, g in zip(truth, guess) if t == 1 and g == 0)
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if tp else 0.0
    return p, r, f1


def pct(a, b):
    return "%d/%d (%.0f%%)" % (a, b, 100 * a / b) if b else "0/0"


def report(name, rows):
    real = [r for r in rows if r["is_control"] == "0"]
    ctrl = [r for r in rows if r["is_control"] == "1"]
    print("\n==== " + name + " ====")
    unparsed = [r for r in real if r["label_acceptable"] not in ("0", "1")]
    real = [r for r in real if r["label_acceptable"] in ("0", "1")]
    print("  judged DeBERTa predictions:", len(real), "| unparseable judge replies:", len(unparsed))
    if not real:
        return

    agree = [r for r in real if r["shown_label"] == r["human_label"]]       # DeBERTa = human
    wrong = [r for r in real if r["shown_label"] != r["human_label"]]       # DeBERTa != human
    acc = lambda rs: sum(1 for r in rs if r["label_acceptable"] == "1")

    print(" Label decision:")
    print("  strict agreement with human label       :", pct(len(agree), len(real)))
    print("  judge accepts DeBERTa's label (overall) :", pct(acc(real), len(real)))
    print("  -- where DeBERTa MATCHES the human     :", pct(acc(agree), len(agree)),
          " (should be high; low = judge stricter than humans)")
    print("  -- where DeBERTa DIFFERS from the human:", pct(acc(wrong), len(wrong)),
          " (these are the 'plausible disagreement' cases)")
    fps = [r for r in wrong if r["shown_label"] == "1"]
    fns = [r for r in wrong if r["shown_label"] == "0"]
    print("       false positives accepted:", pct(acc(fps), len(fps)),
          "| false negatives accepted:", pct(acc(fns), len(fns)))

    # judge-lenient F1: where the judge accepts a disagreeing prediction, treat the prediction as correct
    human = [int(r["human_label"]) for r in real]
    model = [int(r["shown_label"]) for r in real]
    lenient_truth = [int(r["shown_label"]) if r["label_acceptable"] == "1" else int(r["human_label"]) for r in real]
    p, r_, f = prf(human, model)
    p2, r2, f2 = prf(lenient_truth, model)
    print(" F1 (positive class):  strict  P=%.2f R=%.2f F1=%.2f   |   judge-lenient  P=%.2f R=%.2f F1=%.2f" % (p, r_, f, p2, r2, f2))

    pos = [r for r in real if r["shown_label"] == "1" and r["highlights_acceptable"] in ("0", "1")]
    print(" Highlights (pairs where DeBERTa said paraphrase):")
    print("  judge accepts the highlighted words:", pct(sum(1 for r in pos if r["highlights_acceptable"] == "1"), len(pos)))

    if ctrl:
        ctrl_ok = [r for r in ctrl if r["label_acceptable"] in ("0", "1")]
        rejected = sum(1 for r in ctrl_ok if r["label_acceptable"] == "0")
        print(" CONTROL (deliberately wrong 'not a paraphrase' shown for clear paraphrases):")
        print("  judge correctly REJECTS them:", pct(rejected, len(ctrl_ok)), " (should be close to 100%)")

    shown = [r for r in wrong if r["label_acceptable"] == "1"][:8]
    if shown:
        print(" Examples where DeBERTa differs from human but the judge accepts it:")
        for r in shown:
            print("  [%s %s] DeBERTa=%s human=%s | %s" % (r["dataset"], r["pair_id"], r["shown_label"],
                  r["human_label"], r["explanation"][:160]))


def main():
    rows = read_rows()
    report("Rogan / Musk (stratified sample)", [r for r in rows if r["dataset"] == "rogan"])
    report("Obama / Conan", [r for r in rows if r["dataset"] == "obama"])
    report("Both datasets pooled", rows)


if __name__ == "__main__":
    main()
