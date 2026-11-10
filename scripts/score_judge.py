#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
score_judge.py  --  Compare the human labels, DeBERTa and the LLM judge.

Run AFTER llm_judge.py has produced judge_outputs.tsv:
    python score_judge.py

Label 1 ("is a paraphrase") is the positive class, as in the paper.
"""

import csv
import re

INPUT_FILE = "judge_outputs.tsv"


def read_rows():
    with open(INPUT_FILE, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def words(text):
    """Lower-case words, punctuation removed."""
    return set(re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower().replace("’", "'")))


def jaccard(a, b):
    a, b = words(a), words(b)
    if not a or not b:
        return None
    return len(a & b) / len(a | b)


def label_scores(truth, guess):
    """truth and guess are lists of 0/1. Returns a dictionary of scores."""
    tp = sum(1 for t, g in zip(truth, guess) if t == 1 and g == 1)
    fp = sum(1 for t, g in zip(truth, guess) if t == 0 and g == 1)
    fn = sum(1 for t, g in zip(truth, guess) if t == 1 and g == 0)
    tn = sum(1 for t, g in zip(truth, guess) if t == 0 and g == 0)
    n = len(truth)
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    accuracy = (tp + tn) / n if n else float("nan")
    # Cohen's kappa = agreement beyond what chance alone would give
    p_yes = ((tp + fp) / n) * ((tp + fn) / n) if n else 0
    p_no = ((tn + fn) / n) * ((tn + fp) / n) if n else 0
    expected = p_yes + p_no
    kappa = (accuracy - expected) / (1 - expected) if n and expected != 1 else float("nan")
    return {"n": n, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": precision, "recall": recall, "F1": f1,
            "accuracy": accuracy, "kappa": kappa}


def show(title, scores):
    print("  " + title.ljust(28),
          "n=%d  TP=%d FP=%d FN=%d TN=%d  P=%.2f R=%.2f F1=%.2f  acc=%.2f  kappa=%.2f" % (
              scores["n"], scores["TP"], scores["FP"], scores["FN"], scores["TN"],
              scores["precision"], scores["recall"], scores["F1"], scores["accuracy"], scores["kappa"]))


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan"), len(values)


def report(name, rows):
    print("\n==== " + name + " ====")
    parse_fail = [r for r in rows if r["judge_label"] not in ("0", "1")]
    usable = [r for r in rows if r["judge_label"] in ("0", "1") and r["human_label"] in ("0", "1")
              and r["model_label"] in ("0", "1")]
    print("  pairs:", len(rows), "| judge replies that could not be parsed:", len(parse_fail),
          "| pairs used below:", len(usable))
    if not usable:
        return
    human = [int(r["human_label"]) for r in usable]
    judge = [int(r["judge_label"]) for r in usable]
    deberta = [int(r["model_label"]) for r in usable]

    print(" Label scores (positive class = paraphrase):")
    show("DeBERTa vs human", label_scores(human, deberta))
    show("LLM judge vs human", label_scores(human, judge))
    show("LLM judge vs DeBERTa", label_scores(deberta, judge))

    # Highlight overlap: only on pairs where BOTH sides have a highlight (like the paper)
    for who, prefix in (("DeBERTa", "model"), ("LLM judge", "judge")):
        for side in ("guest", "host"):
            scores = [jaccard(r["human_%s_highlight" % side], r["%s_%s_highlight" % (prefix, side)])
                      for r in usable]
            avg, count = mean(scores)
            if count:
                print("  Jaccard %s vs human, %s side: %.2f  (on %d pairs where both highlight)" % (
                    who, side, avg, count))


def main():
    rows = read_rows()
    report("Rogan / Musk (stratified sample - recall is inflated, trust precision more)",
           [r for r in rows if r["dataset"] == "rogan"])
    report("Obama / Conan (all pairs)", [r for r in rows if r["dataset"] == "obama"])
    report("Both datasets pooled", rows)


if __name__ == "__main__":
    main()

