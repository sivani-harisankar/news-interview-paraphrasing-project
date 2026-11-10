#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
llm_judge.py  --  Ask an OpenAI model to act as a "judge" on the pairs you have labelled.

For every guest/host pair, the judge decides:
  1. is part of the host's turn a paraphrase of part of the guest's turn?
  2. which words (in the guest turn and in the host turn)?

The answers are saved next to human labels and DeBERTa's predictions,
so score_judge.py can compare all three.

cmd prompts:
    pip install openai
    set OPENAI_API_KEY=my_key
    python llm_judge.py
"""

import csv
import json
import os
import re
import time
from collections import Counter

# ---------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------
ROGAN_LABELS_FILE = "annotation_sheet_rogan_labelled.tsv"  #40 hand-labelled Rogan pairs
ROGAN_KEY_FILE = "answer_key.tsv"                          # DeBERTa's output for those 40 pairs
OBAMA_LABELS_FILE = "annotation_sheet_conan_labelled.tsv"  #39 hand-labelled Obama pairs
OBAMA_FILE = "conan_obama_predictions.tsv"                 # DeBERTa's output for the 39 Obama pairs
OUTPUT_FILE = "judge_outputs.tsv"

MODEL_NAME = "gpt-4o"      #model key can use
NUMBER_OF_RUNS = 3         #the paper took 3 samples per pair and a majority vote
TEMPERATURE = 1.0          #same as the paper
ONLY_FIRST_N = None           # TEST RUN: only judge 3 pairs, None = judge all of them.

# ---------------------------------------------------------------
# THE PROMPT
# ---------------------------------------------------------------
INSTRUCTIONS = """You are an expert linguistic annotator working on news interviews.

You will see two consecutive turns: first the GUEST's turn, then the HOST's turn.
Decide whether the host's turn contains a context-dependent paraphrase of the guest's turn.

A context-dependent paraphrase means: part of the host's turn restates content from part of
the guest's turn, so that in this conversation the two parts convey the same meaning, even if
they would not count as paraphrases out of context (for example "you've been doing this for a
while now" restating "my 20th time in doing it"). The host's words can differ from the guest's
or repeat them. It does NOT count if the host only talks about the same topic, reacts to the
guest, or moves on to a new question without restating something the guest said.

Reply with JSON only, using exactly these keys:
{
  "explanation": "one or two sentences, thinking step by step",
  "paraphrase": true or false,
  "guest_highlight": "the paraphrased words copied exactly from the GUEST turn (empty string if none)",
  "host_highlight": "the paraphrased words copied exactly from the HOST turn (empty string if none)"
}
If paraphrase is false, both highlights must be empty strings."""

# Worked examples shown to the judge before each real pair.
# This first one is the README example. ADD MORE (the paper used 8) taken from the TRAINING
# split of the authors' data -- never from the test set, and never from your own 79 pairs.
FEW_SHOT_EXAMPLES = [
    {
        "guest": "my 20th time in doing it as a television commentator",
        "host": "you've been doing this for a while now",
        "answer": {
            "explanation": "The host restates the guest's long experience doing it in other words.",
            "paraphrase": True,
            "guest_highlight": "20th time in doing it",
            "host_highlight": "you've been doing this for a while now",
        },
    },
]


def make_user_message(guest_text, host_text):
    return "GUEST: " + guest_text + "\nHOST: " + host_text


def build_messages(guest_text, host_text):
    messages = [{"role": "system", "content": INSTRUCTIONS}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": make_user_message(ex["guest"], ex["host"])})
        messages.append({"role": "assistant", "content": json.dumps(ex["answer"])})
    messages.append({"role": "user", "content": make_user_message(guest_text, host_text)})
    return messages


# ---------------------------------------------------------------
# STEP 1: load the pairs
# ---------------------------------------------------------------
def read_tsv(path):
    # files saved from Excel are often cp1252 instead of utf-8, so try both
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            with open(path, encoding=encoding, newline="") as f:
                return list(csv.DictReader(f, delimiter="\t"))
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not read " + path)


def clean(value):
    return (value or "").strip()


def load_rogan():
    labels = read_tsv(ROGAN_LABELS_FILE)
    key = {clean(r["pair_id"]): r for r in read_tsv(ROGAN_KEY_FILE)}
    rows = []
    for r in labels:
        k = key.get(clean(r["pair_id"]), {})
        rows.append({
            "dataset": "rogan",
            "pair_id": clean(r["pair_id"]),
            "guest_text": clean(r["guest_text"]),
            "host_text": clean(r["host_text"]),
            "human_label": clean(r.get("human_label")),
            "human_guest_highlight": clean(r.get("human_guest_highlight")),
            "human_host_highlight": clean(r.get("human_host_highlight")),
            "model_label": clean(k.get("model_paraphrase")),
            "model_guest_highlight": clean(k.get("guest_highlight")),
            "model_host_highlight": clean(k.get("host_highlight")),
        })
    return rows


def load_obama():
    labels = {clean(r["pair_id"]): r for r in read_tsv(OBAMA_LABELS_FILE)}
    rows = []
    for r in read_tsv(OBAMA_FILE):
        lab = labels.get(clean(r["pair_id"]), {})
        rows.append({
            "dataset": "obama",
            "pair_id": clean(r["pair_id"]),
            "guest_text": clean(r["guest_text"]),
            "host_text": clean(r["host_text"]),
            "human_label": clean(lab.get("human_label")),
            "human_guest_highlight": clean(lab.get("human_guest_highlight")),
            "human_host_highlight": clean(lab.get("human_host_highlight")),
            "model_label": clean(r.get("model_paraphrase")),
            "model_guest_highlight": clean(r.get("guest_highlight")),
            "model_host_highlight": clean(r.get("host_highlight")),
        })
    return rows


# ---------------------------------------------------------------
# STEP 2: ask the judge
# ---------------------------------------------------------------
def ask_once(client, messages):
    """One call to the model. Returns the raw text of its reply."""
    try:
        reply = client.chat.completions.create(
            model=MODEL_NAME, messages=messages,
            temperature=TEMPERATURE, response_format={"type": "json_object"})
    except Exception as err:
        # some newer models refuse a custom temperature: try again without it
        if "temperature" in str(err).lower():
            reply = client.chat.completions.create(
                model=MODEL_NAME, messages=messages,
                response_format={"type": "json_object"})
        else:
            raise
    return reply.choices[0].message.content


def parse_reply(text):
    """Turn the raw reply into (label, guest_highlight, host_highlight).
    Returns None if the reply cannot be parsed (the paper had 17% of these for GPT-4)."""
    try:
        data = json.loads(text)
        value = data["paraphrase"]
        if isinstance(value, str):
            value = value.strip().lower() in ("true", "yes", "1")
        label = 1 if value else 0
        guest_hl = str(data.get("guest_highlight", "") or "").strip()
        host_hl = str(data.get("host_highlight", "") or "").strip()
        if label == 0:
            guest_hl, host_hl = "", ""
        return label, guest_hl, host_hl
    except Exception:
        return None


def judge_pair(client, guest_text, host_text):
    messages = build_messages(guest_text, host_text)
    parsed, raws = [], []
    for _ in range(NUMBER_OF_RUNS):
        try:
            raw = ask_once(client, messages)
        except Exception as err:
            raw = "API_ERROR: " + str(err)
            time.sleep(2)
        raws.append(raw.replace("\t", " ").replace("\n", " "))
        result = parse_reply(raw)
        if result is not None:
            parsed.append(result)

    if not parsed:
        return {"judge_label": "", "judge_guest_highlight": "", "judge_host_highlight": "",
                "votes_for_paraphrase": 0, "runs_parsed": 0, "raw": raws}

    votes = Counter(label for label, _, _ in parsed)
    # majority vote; on a tie (only possible if a run failed) we say "not a paraphrase"
    winner = 1 if votes[1] > votes[0] else 0
    # highlights come from the first run that agreed with the majority
    for label, g, h in parsed:
        if label == winner:
            guest_hl, host_hl = g, h
            break
    return {"judge_label": winner, "judge_guest_highlight": guest_hl, "judge_host_highlight": host_hl,
            "votes_for_paraphrase": votes[1], "runs_parsed": len(parsed), "raw": raws}


# ---------------------------------------------------------------
# STEP 3: run everything and save
# ---------------------------------------------------------------
def main():
    from openai import OpenAI          #reads key from the OPENAI_API_KEY variable
    client = OpenAI()

    pairs = load_rogan() + load_obama()
    pairs = [p for p in pairs if p["human_label"] in ("0", "1")]   # only pairs you labelled
    print("Loaded", len(pairs), "labelled pairs.")
    if not pairs:
        print("STOP: no pairs have a human_label of 0 or 1. Check the label files in the SETTINGS section.")
        return
    if ONLY_FIRST_N:
        pairs = pairs[:ONLY_FIRST_N]
        print("TEST RUN: judging only the first", len(pairs), "pairs.")

    columns = ["dataset", "pair_id", "guest_text", "host_text",
               "human_label", "human_guest_highlight", "human_host_highlight",
               "model_label", "model_guest_highlight", "model_host_highlight",
               "judge_label", "judge_guest_highlight", "judge_host_highlight",
               "votes_for_paraphrase", "runs_parsed", "raw_responses"]

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(columns)
        for i, p in enumerate(pairs, start=1):
            result = judge_pair(client, p["guest_text"], p["host_text"])
            row = dict(p)
            row.update(result)
            row["raw_responses"] = " ||| ".join(result["raw"])
            writer.writerow([str(row.get(c, "")).replace("\t", " ").replace("\n", " ") for c in columns])
            print(i, "/", len(pairs), p["dataset"], p["pair_id"], "-> judge says", result["judge_label"],
                  "(human", p["human_label"] + ")")
    print("Done! Saved to", OUTPUT_FILE)


if __name__ == "__main__":
    main()

